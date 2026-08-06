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
    init_alignment_uniform,
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
class SoftMatchSpec:
    """Specification for SoftMatch (torch-only)."""

    model_bundle: TorchModelBundle | None = None
    lambda_u: float = 1.0
    temperature: float = 0.5
    ema_p: float = 0.999
    n_sigma: float = 2.0
    per_class: bool = False
    dist_align: bool = True
    dist_uniform: bool = True
    hard_label: bool = True
    use_cat: bool = False
    mu: int = 7
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


class SoftMatchMethod(TorchBundlePredictMixin, InductiveMethod):
    """SoftMatch with truncated Gaussian weighting (torch-only)."""

    info = MethodInfo(
        method_id="softmatch",
        name="SoftMatch",
        year=2023,
        family="pseudo-label",
        supports_gpu=True,
        paper_title="SoftMatch: Addressing the Quantity-Quality Tradeoff in Semi-supervised Learning",
        paper_pdf="https://openreview.net/pdf?id=ymt1zQXBDiF",
        official_code="https://github.com/Hhhhhhao/SoftMatch",
    )

    def __init__(self, spec: SoftMatchSpec | None = None) -> None:
        self.spec = spec or SoftMatchSpec()
        self._bundle: TorchModelBundle | None = None
        self._backend: str | None = None
        self._p_model: Any | None = None
        self._p_target: Any | None = None
        self._prob_max_mu_t: Any | None = None
        self._prob_max_var_t: Any | None = None
        self.diagnostics_: dict[str, Any] = {}

    def _dist_align(self, probs_u: Any, probs_l: Any) -> Any:
        if not bool(self.spec.dist_align):
            return probs_u
        if self._p_model is None or self._p_target is None:
            self._p_model, self._p_target = init_alignment_uniform(
                n_classes=int(probs_u.shape[1]),
                device=probs_u.device,
            )
        assert self._p_model is not None and self._p_target is not None
        m = float(self.spec.ema_p)
        self._p_model = self._p_model * m + (1.0 - m) * probs_u.mean(dim=0)
        if bool(self.spec.dist_uniform):
            target = self._p_target
        else:
            # The pinned TorchSSL SoftMatch implementation updates the target
            # distribution from labeled predictions before applying DA.
            self._p_target = self._p_target * m + (1.0 - m) * probs_l.mean(dim=0)
            target = self._p_target
        ratio = target / self._p_model.clamp_min(1e-6)
        aligned = probs_u * ratio
        aligned = aligned / aligned.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return aligned

    def _init_stats(self, *, n_classes: int, device: Any) -> None:
        torch = optional_import("torch", extra="inductive-torch")
        if bool(self.spec.per_class):
            self._prob_max_mu_t = torch.full(
                (int(n_classes),), 1.0 / float(n_classes), device=device
            )
            self._prob_max_var_t = torch.ones((int(n_classes),), device=device)
        else:
            self._prob_max_mu_t = torch.tensor(1.0 / float(n_classes), device=device)
            self._prob_max_var_t = torch.tensor(1.0, device=device)

    def _update_stats(self, max_probs: Any, max_idx: Any) -> None:
        torch = optional_import("torch", extra="inductive-torch")
        if self._prob_max_mu_t is None or self._prob_max_var_t is None:
            raise InductiveValidationError("SoftMatch stats not initialized.")

        m = float(self.spec.ema_p)
        if not bool(self.spec.per_class):
            mu = max_probs.mean()
            var = (
                max_probs.var(unbiased=True)
                if int(max_probs.numel()) > 1
                else max_probs.new_tensor(0.0)
            )
            self._prob_max_mu_t = self._prob_max_mu_t * m + (1.0 - m) * mu
            self._prob_max_var_t = self._prob_max_var_t * m + (1.0 - m) * var
        else:
            num_classes = int(self._prob_max_mu_t.numel())
            mu = torch.zeros((num_classes,), device=max_probs.device)
            var = torch.ones((num_classes,), device=max_probs.device)
            for c in range(num_classes):
                cls_probs = max_probs[max_idx == c]
                if int(cls_probs.numel()) > 1:
                    mu[c] = cls_probs.mean()
                    var[c] = cls_probs.var(unbiased=True)
            self._prob_max_mu_t = self._prob_max_mu_t * m + (1.0 - m) * mu
            self._prob_max_var_t = self._prob_max_var_t * m + (1.0 - m) * var

    def _paper_step(
        self,
        logits_l: Any,
        logits_uw: Any,
        logits_us: Any,
        y_lb: Any,
        _idx_u: Any,
    ) -> MatchStepResult:
        torch = optional_import("torch", extra="inductive-torch")
        sup_loss = torch.nn.functional.cross_entropy(logits_l, y_lb)
        probs_raw = torch.softmax(logits_uw.detach(), dim=1)
        probs_l = torch.softmax(logits_l.detach(), dim=1)
        if self._prob_max_mu_t is None or self._prob_max_var_t is None:
            self._init_stats(n_classes=int(logits_l.shape[1]), device=logits_l.device)
        raw_max_probs, raw_max_idx = probs_raw.max(dim=1)
        # The TorchSSL implementation that produced the Section 4.1 results
        # estimates the Gaussian on raw weak predictions.  Its alignment is
        # then used only for the sample weight; the hard pseudo-label remains
        # the argmax of the original prediction.
        self._update_stats(raw_max_probs, raw_max_idx)
        probs_aligned = self._dist_align(probs_raw, probs_l)
        max_probs, max_idx = probs_aligned.max(dim=1)
        assert self._prob_max_mu_t is not None and self._prob_max_var_t is not None
        if self.spec.per_class:
            mu = self._prob_max_mu_t[max_idx]
            variance = self._prob_max_var_t[max_idx]
        else:
            mu = self._prob_max_mu_t
            variance = self._prob_max_var_t
        denominator = (2.0 * variance / (float(self.spec.n_sigma) ** 2)).clamp_min(1e-12)
        distance = torch.clamp(max_probs - mu, max=0.0)
        weight = torch.exp(-(distance**2) / denominator)
        pseudo = probs_raw.argmax(dim=1)
        loss_u = torch.nn.functional.cross_entropy(logits_us, pseudo, reduction="none")
        unsup_loss = (loss_u * weight).mean()
        return MatchStepResult(
            loss=sup_loss + float(self.spec.lambda_u) * unsup_loss,
            accepted=float(weight.sum().item()),
            unlabeled=int(weight.numel()),
            diagnostics={
                "supervised_loss": float(sup_loss.detach().item()),
                "unsupervised_loss": float(unsup_loss.detach().item()),
                "mean_weight": float(weight.mean().item()),
                "prob_max_mu": float(mu.mean().item()),
                "prob_max_variance": float(variance.mean().item()),
            },
        )

    def _paper_state(self) -> dict[str, Any]:
        values = (
            self._p_model,
            self._p_target,
            self._prob_max_mu_t,
            self._prob_max_var_t,
        )
        if any(value is None for value in values):
            raise InductiveValidationError("SoftMatch paper state is not initialized.")
        return {
            "p_model": self._p_model,
            "p_target": self._p_target,
            "prob_max_mu_t": self._prob_max_mu_t,
            "prob_max_var_t": self._prob_max_var_t,
        }

    def _load_paper_state(self, state: Mapping[str, Any]) -> None:
        torch = optional_import("torch", extra="inductive-torch")
        if self.spec.model_bundle is None:
            raise InductiveValidationError("SoftMatch checkpoint requires a model bundle.")
        device = next(self.spec.model_bundle.model.parameters()).device
        names = ("p_model", "p_target", "prob_max_mu_t", "prob_max_var_t")
        values = [state.get(name) for name in names]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise InductiveValidationError("SoftMatch checkpoint state is invalid.")
        self._p_model = values[0].to(device=device)
        self._p_target = values[1].to(device=device)
        self._prob_max_mu_t = values[2].to(device=device)
        self._prob_max_var_t = values[3].to(device=device)

    def _paper_trace(self) -> dict[str, Any]:
        return {
            "p_model": (
                self._p_model.detach().cpu().tolist() if self._p_model is not None else None
            ),
            "p_target": (
                self._p_target.detach().cpu().tolist() if self._p_target is not None else None
            ),
            "prob_max_mu_t": (
                self._prob_max_mu_t.detach().cpu().tolist()
                if self._prob_max_mu_t is not None
                else None
            ),
            "prob_max_var_t": (
                self._prob_max_var_t.detach().cpu().tolist()
                if self._prob_max_var_t is not None
                else None
            ),
        }

    def _validate_fixed_step_contract(self) -> None:
        expected = {
            "lambda_u": (float(self.spec.lambda_u), 1.0),
            "temperature": (float(self.spec.temperature), 0.5),
            "ema_p": (float(self.spec.ema_p), 0.999),
            "n_sigma": (float(self.spec.n_sigma), 2.0),
            "mu": (int(self.spec.mu), 7),
            "batch_size": (int(self.spec.batch_size), 64),
        }
        changed = [name for name, (actual, target) in expected.items() if actual != target]
        if (
            changed
            or self.spec.per_class
            or not self.spec.dist_align
            or self.spec.dist_uniform
            or not self.spec.hard_label
            or not self.spec.use_cat
            or not self.spec.detach_target
        ):
            raise InductiveValidationError(
                "SoftMatch fixed-step contract changed a frozen hyperparameter: "
                + ", ".join(changed or ["boolean training contract"])
            )

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> SoftMatchMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "params lambda_u=%s temperature=%s ema_p=%s n_sigma=%s per_class=%s dist_align=%s "
            "dist_uniform=%s hard_label=%s use_cat=%s mu=%s batch_size=%s max_epochs=%s "
            "max_steps=%s "
            "detach_target=%s has_model_bundle=%s device=%s seed=%s",
            self.spec.lambda_u,
            self.spec.temperature,
            self.spec.ema_p,
            self.spec.n_sigma,
            self.spec.per_class,
            self.spec.dist_align,
            self.spec.dist_uniform,
            self.spec.hard_label,
            self.spec.use_cat,
            self.spec.mu,
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
            raise InductiveValidationError("SoftMatch requires torch tensors (torch backend).")

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u_w is None or ds.X_u_s is None:
            raise InductiveValidationError("SoftMatch requires X_u_w and X_u_s.")

        X_l = ds.X_l
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        X_u_w = ds.X_u_w
        X_u_s = ds.X_u_s
        logger.info(
            "SoftMatch sizes: n_labeled=%s n_unlabeled=%s",
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
            raise InductiveValidationError("model_bundle must be provided for SoftMatch.")
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
        if float(self.spec.temperature) <= 0:
            raise InductiveValidationError("temperature must be > 0.")
        if float(self.spec.n_sigma) <= 0:
            raise InductiveValidationError("n_sigma must be > 0.")
        if not (0.0 <= float(self.spec.ema_p) < 1.0):
            raise InductiveValidationError("ema_p must be in [0, 1).")

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
        effective_weight_total = 0.0
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

                sup_loss = torch.nn.functional.cross_entropy(logits_l, y_lb)

                logits_uw_target = (
                    logits_uw.detach() if bool(self.spec.detach_target) else logits_uw
                )
                probs_u_raw = torch.softmax(logits_uw_target, dim=1)
                probs_l = torch.softmax(logits_l.detach(), dim=1)
                probs_u = self._dist_align(probs_u_raw, probs_l)

                max_probs, max_idx = probs_u.max(dim=1)
                if self._prob_max_mu_t is None or self._prob_max_var_t is None:
                    self._init_stats(n_classes=int(logits_l.shape[1]), device=logits_l.device)
                self._update_stats(max_probs, max_idx)

                assert self._prob_max_mu_t is not None and self._prob_max_var_t is not None
                if bool(self.spec.per_class):
                    mu = self._prob_max_mu_t[max_idx]
                    var = self._prob_max_var_t[max_idx]
                else:
                    mu = self._prob_max_mu_t
                    var = self._prob_max_var_t
                denom = (2.0 * var / (float(self.spec.n_sigma) ** 2)).clamp_min(1e-12)
                diff = torch.clamp(max_probs - mu, max=0.0)
                weight = torch.exp(-(diff**2) / denom)
                effective_weight_total += float(weight.sum().item())
                unlabeled_total += int(weight.numel())

                pseudo_soft = _sharpen(probs_u_raw, temperature=float(self.spec.temperature))
                if bool(self.spec.hard_label):
                    pseudo = pseudo_soft.argmax(dim=1)
                    loss_u = torch.nn.functional.cross_entropy(logits_us, pseudo, reduction="none")
                else:
                    log_probs = torch.nn.functional.log_softmax(logits_us, dim=1)
                    loss_u = -(pseudo_soft * log_probs).sum(dim=1)

                unsup_loss = (loss_u * weight).mean()

                if step == 0:
                    mu_mean = float(mu.mean().item())
                    var_mean = float(var.mean().item())
                    weight_mean = float(weight.mean().item()) if int(weight.numel()) else 0.0
                    if self._p_model is not None and self._p_target is not None:
                        p_model_min = float(self._p_model.min().item())
                        p_model_max = float(self._p_model.max().item())
                        p_target_min = float(self._p_target.min().item())
                        p_target_max = float(self._p_target.max().item())
                    else:
                        p_model_min = p_model_max = p_target_min = p_target_max = 0.0
                    logger.debug(
                        "SoftMatch epoch=%s mu_mean=%.3f var_mean=%.3f weight_mean=%.3f "
                        "dist_align=%s dist_uniform=%s p_model[min=%.3f max=%.3f] "
                        "p_target[min=%.3f max=%.3f]",
                        epoch,
                        mu_mean,
                        var_mean,
                        weight_mean,
                        self.spec.dist_align,
                        self.spec.dist_uniform,
                        p_model_min,
                        p_model_max,
                        p_target_min,
                        p_target_max,
                    )

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
            "effective_pseudo_label_weight": effective_weight_total,
            "unlabeled_predictions": unlabeled_total,
            "mean_pseudo_label_weight": effective_weight_total / max(unlabeled_total, 1),
        }
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self
