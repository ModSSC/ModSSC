from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Literal

from modssc.capabilities import MethodCapabilities
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
from modssc.inductive.model_binding import ModelBindingSpec
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec
from modssc.runtime.contracts import MethodExecutionContract
from modssc.runtime.method_contracts import (
    fallback_method_execution_contract,
    with_inductive_input_roles,
)

logger = logging.getLogger(__name__)


_sharpen = sharpen_probs


@dataclass(frozen=True)
class FixMatchSpec:
    """Specification for FixMatch (torch-only)."""

    model_bundle: TorchModelBundle | None = None
    lambda_u: float = 1.0
    p_cutoff: float = 0.95
    temperature: float = 0.5
    mu: int = 7
    hard_label: bool = True
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
    evaluation_interval_steps: int = field(default=1024, kw_only=True)
    evaluation_tail_interval_steps: int | None = field(default=None, kw_only=True)
    evaluation_tail_start_fraction: float | None = field(default=None, kw_only=True)
    checkpoint_interval_steps: int = field(default=1024, kw_only=True)
    reporting_policy: str = field(default="median_last_checkpoints", kw_only=True)
    reporting_window_checkpoints: int = field(default=20, kw_only=True)
    allow_short_run: bool = field(default=False, kw_only=True)


class FixMatchMethod(TorchBundlePredictMixin, InductiveMethod):
    """FixMatch pseudo-labeling with weak/strong augmentation (torch-only)."""

    info = MethodInfo(
        method_id="fixmatch",
        name="FixMatch",
        year=2020,
        family="pseudo-label",
        supports_gpu=True,
        paper_title="FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence",
        paper_pdf="https://arxiv.org/pdf/2001.07685",
        official_code="https://github.com/google-research/fixmatch",
        capabilities=MethodCapabilities(
            regime="inductive",
            requires_unlabeled=True,
            requires_weak_augmentation=True,
            min_strong_augmentations=1,
            required_classifier_outputs=frozenset({"logits"}),
            backends=frozenset({"torch"}),
            supports_checkpointing=True,
        ),
        model_binding=ModelBindingSpec.single(),
    )

    @classmethod
    def execution_contract(
        cls,
        spec: FixMatchSpec,
        capabilities: MethodCapabilities,
        model_binding: Any | None = None,
    ) -> MethodExecutionContract:
        fixed_steps = uses_fixed_step_match(spec)
        feature_roles = ("fit.X_l", "fit.X_u_w", "fit.X_u_s.0")
        contract = with_inductive_input_roles(
            fallback_method_execution_contract(cls, capabilities, model_binding),
            feature_roles=feature_roles,
            row_groups=(
                ("fit.X_l", "fit.y_l"),
                ("fit.X_u_w", "fit.X_u_s.0"),
            ),
        )
        return replace(
            contract,
            components=tuple(
                replace(
                    requirement,
                    outputs=frozenset({"logits"}),
                    input_roles=feature_roles,
                    requires_ema=fixed_steps,
                    requires_scheduler=fixed_steps,
                    scheduler_types=(frozenset({"LambdaLR"}) if fixed_steps else frozenset()),
                )
                for requirement in contract.components
            ),
        )

    def __init__(self, spec: FixMatchSpec | None = None) -> None:
        self.spec = spec or FixMatchSpec()
        self._bundle: TorchModelBundle | None = None
        self._backend: str | None = None
        self.diagnostics_: dict[str, Any] = {}

    @property
    def unlabeled_index_space(self) -> Literal["local", "source"]:
        """Index space required by the active training protocol."""

        return "local" if uses_fixed_step_match(self.spec) else "source"

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
        probs = torch.softmax(logits_uw.detach(), dim=1)
        max_probs, pseudo = probs.max(dim=1)
        mask = (max_probs >= float(self.spec.p_cutoff)).to(logits_us.dtype)
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
                "confidence_mean": float(max_probs.mean().item()),
            },
        )

    @staticmethod
    def _paper_state() -> dict[str, Any]:
        return {}

    @staticmethod
    def _load_paper_state(state: Mapping[str, Any]) -> None:
        if dict(state):
            raise InductiveValidationError("FixMatch paper state must be empty.")

    def _paper_trace(self) -> dict[str, Any]:
        return {"confidence_threshold": float(self.spec.p_cutoff)}

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
            or not self.spec.use_cat
            or not self.spec.detach_target
        ):
            raise InductiveValidationError(
                "FixMatch fixed-step contract changed a frozen hyperparameter: "
                + ", ".join(changed or ["boolean training contract"])
            )

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> FixMatchMethod:
        start = perf_counter()
        self.evaluation_metric_sets_ = {}
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "params lambda_u=%s p_cutoff=%s temperature=%s mu=%s hard_label=%s use_cat=%s "
            "batch_size=%s max_epochs=%s max_steps=%s detach_target=%s "
            "has_model_bundle=%s device=%s seed=%s",
            self.spec.lambda_u,
            self.spec.p_cutoff,
            self.spec.temperature,
            self.spec.mu,
            self.spec.hard_label,
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
            raise InductiveValidationError("FixMatch requires torch tensors (torch backend).")

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u_w is None or ds.X_u_s is None:
            raise InductiveValidationError("FixMatch requires X_u_w and X_u_s.")

        X_l = ds.X_l
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        X_u_w = ds.X_u_w
        X_u_s = ds.X_u_s
        logger.info(
            "FixMatch sizes: n_labeled=%s n_unlabeled=%s",
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
            raise InductiveValidationError("model_bundle must be provided for FixMatch.")
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
                probs_uw = torch.softmax(logits_uw, dim=1)
                if bool(self.spec.detach_target):
                    probs_uw = probs_uw.detach()
                mask = (probs_uw.max(dim=1).values >= float(self.spec.p_cutoff)).to(logits_us.dtype)
                accepted_total += int(mask.sum().item())
                unlabeled_total += int(mask.numel())

                pseudo_soft = _sharpen(probs_uw, temperature=float(self.spec.temperature))
                if bool(self.spec.hard_label):
                    pseudo = pseudo_soft.argmax(dim=1)
                    loss_u = torch.nn.functional.cross_entropy(logits_us, pseudo, reduction="none")
                else:
                    log_probs = torch.nn.functional.log_softmax(logits_us, dim=1)
                    loss_u = -(pseudo_soft * log_probs).sum(dim=1)

                if int(mask.numel()) == 0:
                    unsup_loss = torch.zeros((), device=logits_us.device)
                else:
                    unsup_loss = (loss_u * mask).mean()

                if step == 0:
                    mask_mean = float(mask.mean().item()) if int(mask.numel()) else 0.0
                    logger.debug(
                        "FixMatch epoch=%s p_cutoff=%s mask_mean=%.3f sup_loss=%.4f unsup_loss=%.4f",
                        epoch,
                        self.spec.p_cutoff,
                        mask_mean,
                        float(sup_loss.item()),
                        float(unsup_loss.item()),
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
            "accepted_pseudo_labels": accepted_total,
            "unlabeled_predictions": unlabeled_total,
            "acceptance_rate": accepted_total / max(unlabeled_total, 1),
            "confidence_threshold": float(self.spec.p_cutoff),
        }
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self
