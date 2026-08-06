from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.deep_utils import (
    TorchBundlePredictMixin,
    concat_data,
    cycle_batch_indices,
    ensure_float_tensor,
    ensure_model_bundle,
    ensure_model_device,
    extract_logits,
    get_torch_device,
    get_torch_len,
    num_batches,
    sharpen_probs,
)
from modssc.inductive.methods.helpers.match_trainer import (
    MatchStepResult,
    run_fixed_step_match,
    uses_fixed_step_match,
)
from modssc.inductive.methods.helpers.ssl_augmentation import ssl_batch_views
from modssc.inductive.methods.utils import (
    detect_backend,
    ensure_1d_labels_torch,
    ensure_torch_data,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec

logger = logging.getLogger(__name__)


_sharpen = sharpen_probs


@dataclass(frozen=True)
class FlexMatchSpec:
    """Specification for FlexMatch (torch-only)."""

    model_bundle: TorchModelBundle | None = None
    lambda_u: float = 1.0
    p_cutoff: float = 0.95
    temperature: float = 0.5
    mu: int = 7
    hard_label: bool = True
    thresh_warmup: bool = True
    use_cat: bool = False
    batch_size: int = 64
    max_epochs: int = 1
    detach_target: bool = True
    max_steps: int | None = field(default=None, kw_only=True)
    training_mode: str = field(default="epochs", kw_only=True)
    reference_implementation: str = field(default="standardized", kw_only=True)
    sampler_mode: str = field(default="replacement", kw_only=True)
    sampler_shuffle_buffer: int = field(default=8192, kw_only=True)
    augmentation_profile: str = field(default="", kw_only=True)
    interleave_bn: bool = field(default=False, kw_only=True)
    evaluation_interval_steps: int = field(default=5000, kw_only=True)
    checkpoint_interval_steps: int = field(default=5000, kw_only=True)
    reporting_policy: str = field(default="best_historical_checkpoint", kw_only=True)
    reporting_window_checkpoints: int = field(default=20, kw_only=True)
    allow_short_run: bool = field(default=False, kw_only=True)


class FlexMatchMethod(TorchBundlePredictMixin, InductiveMethod):
    """FlexMatch with classwise adaptive thresholds (torch-only)."""

    info = MethodInfo(
        method_id="flexmatch",
        name="FlexMatch",
        year=2021,
        family="pseudo-label",
        supports_gpu=True,
        paper_title="FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling",
        paper_pdf="https://arxiv.org/pdf/2110.08263",
        official_code="https://github.com/TorchSSL/TorchSSL",
    )

    def __init__(self, spec: FlexMatchSpec | None = None) -> None:
        self.spec = spec or FlexMatchSpec()
        self._bundle: TorchModelBundle | None = None
        self._backend: str | None = None
        self._selected_label: Any | None = None
        self._classwise_acc: Any | None = None
        self._ulb_size: int | None = None
        self.diagnostics_: dict[str, Any] = {}

    def _init_state(self, *, n_classes: int, device: Any) -> None:
        torch = optional_import("torch", extra="inductive-torch")
        ulb_size = self._ulb_size
        if ulb_size is None:
            raise InductiveValidationError("Unlabeled pool size is missing.")
        self._selected_label = torch.full((int(ulb_size),), -1, dtype=torch.long, device=device)
        self._classwise_acc = torch.zeros((int(n_classes),), device=device)

    def _update_classwise_acc(self) -> None:
        torch = optional_import("torch", extra="inductive-torch")
        if self._selected_label is None or self._classwise_acc is None:
            raise InductiveValidationError("FlexMatch state not initialized.")
        sel_cpu = (self._selected_label + 1).detach().cpu()
        counts = torch.bincount(sel_cpu, minlength=int(self._classwise_acc.numel()) + 1)
        counts = counts.to(self._classwise_acc.device)
        if bool(self.spec.thresh_warmup):
            denom = counts.max().clamp_min(1.0)
            self._classwise_acc = counts[1:].to(self._classwise_acc.dtype) / denom
        else:
            counts_pos = counts[1:]
            denom = counts_pos.max()
            if denom <= 0:
                self._classwise_acc = torch.zeros_like(self._classwise_acc)
            else:
                self._classwise_acc = counts_pos.to(self._classwise_acc.dtype) / denom

    def _get_idx_u(self, data: Any, *, device: Any, n_u: int) -> Any:
        torch = optional_import("torch", extra="inductive-torch")
        if data.meta is None:
            self._ulb_size = int(n_u)
            return torch.arange(int(n_u), dtype=torch.long, device=device)
        if not isinstance(data.meta, Mapping):
            raise InductiveValidationError("FlexMatch requires data.meta to be a mapping.")
        idx_u = data.meta.get("idx_u")
        if idx_u is None:
            idx_u = data.meta.get("unlabeled_idx")
        if idx_u is None:
            idx_u = data.meta.get("unlabeled_indices")
        if idx_u is None:
            self._ulb_size = int(n_u)
            return torch.arange(int(n_u), dtype=torch.long, device=device)
        if not isinstance(idx_u, torch.Tensor):
            raise InductiveValidationError("meta.idx_u must be a torch.Tensor.")
        if idx_u.dtype != torch.int64:
            raise InductiveValidationError("meta.idx_u must be int64.")
        if idx_u.ndim != 1:
            raise InductiveValidationError("meta.idx_u must be 1D.")
        if int(idx_u.shape[0]) != int(n_u):
            raise InductiveValidationError("meta.idx_u must match X_u size.")
        if idx_u.device != device:
            raise InductiveValidationError("meta.idx_u must be on the same device as X_u.")

        ulb_size = data.meta.get("ulb_size") or data.meta.get("unlabeled_size")
        if ulb_size is None:
            if int(idx_u.min().item()) != 0 or int(idx_u.max().item()) != int(n_u) - 1:
                raise InductiveValidationError(
                    "meta.idx_u must be contiguous 0..n_u-1 or provide meta.ulb_size."
                )
            uniq = torch.unique(idx_u)
            if int(uniq.numel()) != int(n_u):
                raise InductiveValidationError("meta.idx_u must contain unique indices.")
            ulb_size = int(n_u)
        else:
            if not isinstance(ulb_size, int):
                raise InductiveValidationError("meta.ulb_size must be an int.")
            if ulb_size < int(n_u):
                raise InductiveValidationError("meta.ulb_size must be >= len(idx_u).")
            if int(idx_u.max().item()) >= int(ulb_size):
                raise InductiveValidationError("meta.idx_u entries must be < meta.ulb_size.")

        self._ulb_size = int(ulb_size)
        return idx_u

    def _paper_step(
        self,
        logits_l: Any,
        logits_uw: Any,
        logits_us: Any,
        y_lb: Any,
        idx_u: Any,
    ) -> MatchStepResult:
        torch = optional_import("torch", extra="inductive-torch")
        if self._selected_label is None or self._classwise_acc is None:
            self._init_state(n_classes=int(logits_l.shape[1]), device=logits_l.device)
        self._update_classwise_acc()
        assert self._selected_label is not None and self._classwise_acc is not None

        sup_loss = torch.nn.functional.cross_entropy(logits_l, y_lb)
        probs = torch.softmax(logits_uw.detach(), dim=1)
        max_probs, pseudo = probs.max(dim=1)
        threshold = float(self.spec.p_cutoff) * (
            self._classwise_acc[pseudo] / (2.0 - self._classwise_acc[pseudo])
        )
        mask = (max_probs >= threshold).to(logits_us.dtype)
        select = max_probs >= float(self.spec.p_cutoff)
        self._update_selected_labels(
            idx_u=idx_u,
            pseudo=pseudo,
            select=select,
        )
        loss_u = torch.nn.functional.cross_entropy(logits_us, pseudo, reduction="none")
        unsup_loss = (loss_u * mask).mean()
        return MatchStepResult(
            loss=sup_loss + float(self.spec.lambda_u) * unsup_loss,
            accepted=float(mask.sum().item()),
            unlabeled=int(mask.numel()),
            diagnostics={
                "supervised_loss": float(sup_loss.detach().item()),
                "unsupervised_loss": float(unsup_loss.detach().item()),
                "mask_rate": float(mask.mean().item()),
                "selected_at_base_threshold": int(select.sum().item()),
                "threshold_mean": float(threshold.mean().item()),
            },
        )

    def _update_selected_labels(
        self,
        *,
        idx_u: Any,
        pseudo: Any,
        select: Any,
    ) -> None:
        """Apply accepted CPL labels with deterministic duplicate semantics.

        TorchSSL samples the unlabeled loader with replacement, so one batch can
        contain the same pool index more than once.  PyTorch explicitly leaves
        ``index_put_(accumulate=False)`` undefined for duplicate indices.  Stable
        sorting makes the intended sequential rule explicit: the last accepted
        occurrence in batch order wins.  The final write contains unique indices
        and remains entirely on the accelerator.
        """

        if self._selected_label is None:
            raise InductiveValidationError("FlexMatch state not initialized.")
        torch = optional_import("torch", extra="inductive-torch")
        selected_indices = idx_u[select]
        if int(selected_indices.numel()) == 0:
            return
        selected_pseudo = pseudo[select]
        order = selected_indices.argsort(stable=True)
        sorted_indices = selected_indices[order]
        sorted_pseudo = selected_pseudo[order]
        keep_last = torch.ones_like(sorted_indices, dtype=torch.bool)
        keep_last[:-1] = sorted_indices[:-1] != sorted_indices[1:]
        unique_indices = sorted_indices[keep_last]
        unique_pseudo = sorted_pseudo[keep_last]
        self._selected_label.index_copy_(0, unique_indices, unique_pseudo)

    def _paper_state(self) -> dict[str, Any]:
        if self._selected_label is None or self._classwise_acc is None:
            raise InductiveValidationError("FlexMatch paper state is not initialized.")
        return {
            "selected_label": self._selected_label,
            "classwise_acc": self._classwise_acc,
            "ulb_size": self._ulb_size,
        }

    def _load_paper_state(self, state: Mapping[str, Any]) -> None:
        torch = optional_import("torch", extra="inductive-torch")
        if self.spec.model_bundle is None:
            raise InductiveValidationError("FlexMatch checkpoint requires a model bundle.")
        device = next(self.spec.model_bundle.model.parameters()).device
        selected = state.get("selected_label")
        classwise = state.get("classwise_acc")
        ulb_size = state.get("ulb_size")
        if not isinstance(selected, torch.Tensor) or not isinstance(classwise, torch.Tensor):
            raise InductiveValidationError("FlexMatch checkpoint state is invalid.")
        if not isinstance(ulb_size, int) or int(selected.numel()) != int(ulb_size):
            raise InductiveValidationError("FlexMatch checkpoint pool size is invalid.")
        self._selected_label = selected.to(device=device, dtype=torch.int64)
        self._classwise_acc = classwise.to(device=device)
        self._ulb_size = int(ulb_size)

    def _paper_trace(self) -> dict[str, Any]:
        selected_count = (
            int((self._selected_label >= 0).sum().item()) if self._selected_label is not None else 0
        )
        return {
            "classwise_acc": (
                self._classwise_acc.detach().cpu().tolist()
                if self._classwise_acc is not None
                else None
            ),
            "selected_count": selected_count,
            "selected_label_count": selected_count,
        }

    def _validate_fixed_step_contract(self) -> None:
        expected = {
            "lambda_u": (float(self.spec.lambda_u), 1.0),
            "p_cutoff": (float(self.spec.p_cutoff), 0.95),
            "temperature": (float(self.spec.temperature), 0.5),
            "mu": (int(self.spec.mu), 7),
            "batch_size": (int(self.spec.batch_size), 64),
        }
        changed = [name for name, (actual, target) in expected.items() if actual != target]
        if (
            changed
            or not self.spec.hard_label
            or not self.spec.thresh_warmup
            or not self.spec.use_cat
            or not self.spec.detach_target
        ):
            raise InductiveValidationError(
                "FlexMatch fixed-step contract changed a frozen hyperparameter: "
                + ", ".join(changed or ["boolean training contract"])
            )

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> FlexMatchMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "params lambda_u=%s p_cutoff=%s temperature=%s mu=%s hard_label=%s thresh_warmup=%s "
            "use_cat=%s batch_size=%s max_epochs=%s max_steps=%s detach_target=%s "
            "has_model_bundle=%s "
            "device=%s seed=%s",
            self.spec.lambda_u,
            self.spec.p_cutoff,
            self.spec.temperature,
            self.spec.mu,
            self.spec.hard_label,
            self.spec.thresh_warmup,
            self.spec.use_cat,
            self.spec.batch_size,
            self.spec.max_epochs,
            self.spec.max_steps,
            self.spec.detach_target,
            bool(self.spec.model_bundle),
            device,
            seed,
        )
        if uses_fixed_step_match(self.spec):
            self._validate_fixed_step_contract()
            if getattr(data, "X_u_w", None) is None:
                raise InductiveValidationError("FlexMatch fixed-step training requires X_u_w.")
            self._ulb_size = int(get_torch_len(data.X_u_w))
            run_fixed_step_match(
                self,
                data,
                device=device,
                seed=seed,
                method_id=self.info.method_id,
                step_fn=self._paper_step,
                state_getter=self._paper_state,
                state_loader=self._load_paper_state,
                trace_getter=self._paper_trace,
            )
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self
        if data is None:
            raise InductiveValidationError("data must not be None.")

        backend = detect_backend(data.X_l)
        if backend != "torch":
            raise InductiveValidationError("FlexMatch requires torch tensors (torch backend).")

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u_w is None or ds.X_u_s is None:
            raise InductiveValidationError("FlexMatch requires X_u_w and X_u_s.")

        X_l = ds.X_l
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        X_u_w = ds.X_u_w
        X_u_s = ds.X_u_s
        logger.info(
            "FlexMatch sizes: n_labeled=%s n_unlabeled=%s",
            int(get_torch_len(X_l)),
            int(get_torch_len(X_u_w)),
        )

        if int(get_torch_len(X_l)) == 0:
            raise InductiveValidationError("X_l must be non-empty.")
        if int(get_torch_len(X_u_w)) == 0 or int(get_torch_len(X_u_s)) == 0:
            raise InductiveValidationError("X_u_w and X_u_s must be non-empty.")
        if int(get_torch_len(X_u_w)) != int(get_torch_len(X_u_s)):
            raise InductiveValidationError("X_u_w and X_u_s must have the same number of rows.")

        ensure_float_tensor(X_l, name="X_l")
        ensure_float_tensor(X_u_w, name="X_u_w")
        ensure_float_tensor(X_u_s, name="X_u_s")

        if y_l.dtype != torch.int64:
            raise InductiveValidationError("y_l must be int64 for torch cross entropy.")

        if self.spec.model_bundle is None:
            raise InductiveValidationError("model_bundle must be provided for FlexMatch.")
        bundle = ensure_model_bundle(self.spec.model_bundle)
        model = bundle.model
        optimizer = bundle.optimizer
        ensure_model_device(model, device=get_torch_device(X_l))

        if int(self.spec.batch_size) <= 0:
            raise InductiveValidationError("batch_size must be >= 1.")
        if int(self.spec.mu) < 1:
            raise InductiveValidationError("mu must be >= 1.")
        if int(self.spec.max_epochs) <= 0:
            raise InductiveValidationError("max_epochs must be >= 1.")
        if self.spec.max_steps is not None and int(self.spec.max_steps) <= 0:
            raise InductiveValidationError("max_steps must be >= 1 when provided.")
        if float(self.spec.lambda_u) < 0:
            raise InductiveValidationError("lambda_u must be >= 0.")
        if not (0.0 <= float(self.spec.p_cutoff) <= 1.0):
            raise InductiveValidationError("p_cutoff must be in [0, 1].")
        if float(self.spec.temperature) <= 0:
            raise InductiveValidationError("temperature must be > 0.")

        idx_u_all = self._get_idx_u(
            ds, device=get_torch_device(X_u_w), n_u=int(get_torch_len(X_u_w))
        )

        batch_size = int(self.spec.batch_size)
        unlabeled_batch_size = batch_size * int(self.spec.mu)
        steps_l = num_batches(int(get_torch_len(X_l)), batch_size)
        steps_u = num_batches(int(get_torch_len(X_u_w)), unlabeled_batch_size)
        steps_per_epoch = max(int(steps_l), int(steps_u))
        target_steps = (
            int(self.spec.max_steps)
            if self.spec.max_steps is not None
            else int(self.spec.max_epochs) * steps_per_epoch
        )
        epochs_to_run = (target_steps + steps_per_epoch - 1) // steps_per_epoch

        gen_l = torch.Generator().manual_seed(int(seed))
        gen_u = torch.Generator().manual_seed(int(seed) + 1)

        model.train()
        optimization_steps = 0
        accepted_total = 0
        unlabeled_total = 0
        for epoch in range(epochs_to_run):
            iter_l_idx = cycle_batch_indices(
                int(get_torch_len(X_l)),
                batch_size=batch_size,
                generator=gen_l,
                device=get_torch_device(X_l),
                steps=steps_per_epoch,
            )
            iter_u_idx = cycle_batch_indices(
                int(get_torch_len(X_u_w)),
                batch_size=unlabeled_batch_size,
                generator=gen_u,
                device=get_torch_device(X_u_w),
                steps=steps_per_epoch,
            )
            for step, (idx_l, idx_u) in enumerate(zip(iter_l_idx, iter_u_idx, strict=False)):
                global_step = epoch * steps_per_epoch + step
                if global_step >= target_steps:
                    break
                x_lb, x_uw, x_us = ssl_batch_views(
                    ds,
                    X_l=X_l,
                    X_u_w=X_u_w,
                    X_u_s=X_u_s,
                    idx_l=idx_l,
                    idx_u=idx_u,
                    optimization_step=global_step,
                )
                y_lb = y_l[idx_l]
                idx_global = idx_u_all[idx_u]

                if bool(self.spec.use_cat):
                    inputs = concat_data([x_lb, x_uw, x_us])
                    logits = extract_logits(model(inputs))
                    if int(logits.ndim) != 2:
                        raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                    num_lb = int(get_torch_len(x_lb))
                    num_u = int(get_torch_len(x_uw))
                    expected = num_lb + num_u + int(get_torch_len(x_us))
                    if int(logits.shape[0]) != expected:
                        raise InductiveValidationError(
                            "Concatenated logits batch size does not match inputs."
                        )
                    logits_l = logits[:num_lb]
                    logits_uw = logits[num_lb : num_lb + num_u]
                    logits_us = logits[num_lb + num_u :]
                else:
                    logits_l = extract_logits(model(x_lb))
                    logits_us = extract_logits(model(x_us))
                    with torch.no_grad():
                        logits_uw = extract_logits(model(x_uw))

                if int(logits_l.ndim) != 2 or int(logits_uw.ndim) != 2 or int(logits_us.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                if logits_uw.shape != logits_us.shape:
                    raise InductiveValidationError("Unlabeled logits shape mismatch.")
                if logits_uw.shape[1] != logits_l.shape[1]:
                    raise InductiveValidationError("Logits must agree on class dimension.")
                if y_lb.min().item() < 0 or y_lb.max().item() >= int(logits_l.shape[1]):
                    raise InductiveValidationError("y_l labels must be within [0, n_classes).")

                if self._classwise_acc is None or self._selected_label is None:
                    self._init_state(
                        n_classes=int(logits_l.shape[1]),
                        device=get_torch_device(X_u_w),
                    )

                sup_loss = torch.nn.functional.cross_entropy(logits_l, y_lb)
                logits_uw_target = (
                    logits_uw.detach() if bool(self.spec.detach_target) else logits_uw
                )
                probs_uw = torch.softmax(logits_uw_target, dim=1)
                max_probs, max_idx = probs_uw.max(dim=1)

                class_acc = self._classwise_acc[max_idx]
                thresh = float(self.spec.p_cutoff) * (class_acc / (2.0 - class_acc))
                mask = (max_probs >= thresh).to(logits_us.dtype)
                accepted_total += int(mask.sum().item())
                unlabeled_total += int(mask.numel())

                pseudo_soft = _sharpen(probs_uw, temperature=float(self.spec.temperature))
                if bool(self.spec.hard_label):
                    pseudo = pseudo_soft.argmax(dim=1)
                    loss_u = torch.nn.functional.cross_entropy(logits_us, pseudo, reduction="none")
                else:
                    log_probs = torch.nn.functional.log_softmax(logits_us, dim=1)
                    loss_u = -(pseudo_soft * log_probs).sum(dim=1)

                unsup_loss = (loss_u * mask).mean()

                if step == 0:
                    mask_mean = float(mask.mean().item()) if int(mask.numel()) else 0.0
                    logger.debug(
                        "FlexMatch epoch=%s p_cutoff=%s thresh_warmup=%s class_acc_mean=%.3f "
                        "thresh_mean=%.3f mask_mean=%.3f",
                        epoch,
                        self.spec.p_cutoff,
                        self.spec.thresh_warmup,
                        float(class_acc.mean().item()),
                        float(thresh.mean().item()),
                        mask_mean,
                    )

                select = max_probs >= float(self.spec.p_cutoff)
                if int(select.sum().item()) > 0:
                    self._selected_label[idx_global[select]] = max_idx[select]
                    self._update_classwise_acc()

                loss = sup_loss + float(self.spec.lambda_u) * unsup_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimization_steps += 1

        self._bundle = bundle
        self._backend = backend
        self.diagnostics_ = {
            "optimization_steps": optimization_steps,
            "target_steps": target_steps,
            "accepted_pseudo_labels": accepted_total,
            "unlabeled_predictions": unlabeled_total,
            "acceptance_rate": accepted_total / max(unlabeled_total, 1),
            "selected_label_count": int((self._selected_label >= 0).sum().item())
            if self._selected_label is not None
            else 0,
        }
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self
