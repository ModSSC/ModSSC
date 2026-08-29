from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.deep.match_primitives import (
    FixedSSLBatchSampler,
    deinterleave_batch,
    interleave_batch,
)
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.helpers.ssl_augmentation import ssl_batch_views
from modssc.inductive.methods.helpers.torch_support import (
    concat_data,
    ensure_float_tensor,
    ensure_model_bundle,
    ensure_model_device,
    extract_logits,
    get_torch_device,
    get_torch_len,
    optimizer_step_with_bundle,
    slice_data,
)
from modssc.inductive.methods.utils import (
    detect_backend,
    ensure_1d_labels_torch,
    ensure_torch_data,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec
from modssc.runtime.checkpoint import CheckpointError, CheckpointStore
from modssc.runtime.continuation import PlannedContinuation, continuation_requested
from modssc.runtime.execution import ExecutionContext

MATCH_REFERENCE_TARGET_STEPS = 1 << 20

MatchTrainingMode = Literal["epochs", "fixed_steps"]
MatchReferenceImplementation = Literal["google_fixmatch", "torchssl"]
MatchReportingPolicy = Literal["median_last_checkpoints", "best_historical_checkpoint"]


@dataclass(frozen=True)
class MatchTrainerConfig:
    """Generic executable contract for the shared fixed-step Match trainer."""

    reference_implementation: MatchReferenceImplementation
    sampler_mode: Literal["replacement", "shuffle_repeat"]
    sampler_shuffle_buffer: int
    augmentation_profile: Literal["google_fixmatch_ra", "torchssl_ra"]
    interleave_bn: bool
    evaluation_interval_steps: int
    evaluation_tail_interval_steps: int | None
    evaluation_tail_start_fraction: float | None
    checkpoint_interval_steps: int
    reporting_policy: MatchReportingPolicy
    reporting_window_checkpoints: int
    allow_short_run: bool


@dataclass(frozen=True)
class MatchStepResult:
    """One method-specific contribution to the shared Match training loop."""

    loss: Any
    accepted: float
    unlabeled: int
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class MatchTrainingResult:
    """State returned by an uninterrupted or resumed paper training run."""

    optimization_steps: int
    target_steps: int
    accepted: float
    unlabeled: int
    evaluation_history: tuple[dict[str, Any], ...]
    paper_metrics: Mapping[str, Any]
    checkpoint_history: tuple[dict[str, Any], ...]
    resumed_from_step: int
    best_historical_checkpoint: Mapping[str, Any] | None = None


StepFunction = Callable[[Any, Any, Any, Any, Any], MatchStepResult]
StateGetter = Callable[[], Mapping[str, Any]]
StateLoader = Callable[[Mapping[str, Any]], None]
TraceGetter = Callable[[], Mapping[str, Any]]


def uses_fixed_step_match(spec: Any) -> bool:
    """Return whether a method requests the generic resumable Match trainer."""

    mode = str(getattr(spec, "training_mode", "epochs"))
    if mode not in ("epochs", "fixed_steps"):
        raise InductiveValidationError("training_mode must be 'epochs' or 'fixed_steps'.")
    return mode == "fixed_steps"


def _trainer_config(spec: Any) -> MatchTrainerConfig:
    reference = str(getattr(spec, "reference_implementation", "standardized"))
    if reference not in ("google_fixmatch", "torchssl"):
        raise InductiveValidationError(
            "fixed_steps requires reference_implementation='google_fixmatch' or 'torchssl'."
        )
    sampler_mode = str(getattr(spec, "sampler_mode", "replacement"))
    if sampler_mode not in ("replacement", "shuffle_repeat"):
        raise InductiveValidationError("sampler_mode must be 'replacement' or 'shuffle_repeat'.")
    augmentation_profile = str(getattr(spec, "augmentation_profile", ""))
    if augmentation_profile not in ("google_fixmatch_ra", "torchssl_ra"):
        raise InductiveValidationError(
            "augmentation_profile must be 'google_fixmatch_ra' or 'torchssl_ra'."
        )
    reporting_policy = str(getattr(spec, "reporting_policy", ""))
    if reporting_policy not in ("median_last_checkpoints", "best_historical_checkpoint"):
        raise InductiveValidationError(
            "reporting_policy must be 'median_last_checkpoints' or 'best_historical_checkpoint'."
        )
    shuffle_buffer = int(getattr(spec, "sampler_shuffle_buffer", 8192))
    evaluation_interval = int(getattr(spec, "evaluation_interval_steps", 0))
    raw_tail_interval = getattr(spec, "evaluation_tail_interval_steps", None)
    raw_tail_fraction = getattr(spec, "evaluation_tail_start_fraction", None)
    tail_interval = None if raw_tail_interval is None else int(raw_tail_interval)
    tail_fraction = None if raw_tail_fraction is None else float(raw_tail_fraction)
    checkpoint_interval = int(getattr(spec, "checkpoint_interval_steps", 0))
    reporting_window = int(getattr(spec, "reporting_window_checkpoints", 20))
    if shuffle_buffer <= 0:
        raise InductiveValidationError("sampler_shuffle_buffer must be positive.")
    if evaluation_interval <= 0:
        raise InductiveValidationError("evaluation_interval_steps must be positive.")
    if (tail_interval is None) != (tail_fraction is None):
        raise InductiveValidationError(
            "evaluation tail interval and start fraction must be configured together."
        )
    if tail_interval is not None and tail_interval <= 0:
        raise InductiveValidationError("evaluation_tail_interval_steps must be positive.")
    if tail_fraction is not None and not 0.0 < tail_fraction < 1.0:
        raise InductiveValidationError(
            "evaluation_tail_start_fraction must be strictly between 0 and 1."
        )
    if checkpoint_interval <= 0:
        raise InductiveValidationError("checkpoint_interval_steps must be positive.")
    if reporting_window <= 0:
        raise InductiveValidationError("reporting_window_checkpoints must be positive.")
    return MatchTrainerConfig(
        reference_implementation=reference,  # type: ignore[arg-type]
        sampler_mode=sampler_mode,  # type: ignore[arg-type]
        sampler_shuffle_buffer=shuffle_buffer,
        augmentation_profile=augmentation_profile,  # type: ignore[arg-type]
        interleave_bn=bool(getattr(spec, "interleave_bn", False)),
        evaluation_interval_steps=evaluation_interval,
        evaluation_tail_interval_steps=tail_interval,
        evaluation_tail_start_fraction=tail_fraction,
        checkpoint_interval_steps=checkpoint_interval,
        reporting_policy=reporting_policy,  # type: ignore[arg-type]
        reporting_window_checkpoints=reporting_window,
        allow_short_run=bool(getattr(spec, "allow_short_run", False)),
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _match_contract_error(field: str, actual: Any, expected: Any) -> None:
    raise InductiveValidationError(
        "Match runtime violates the configured reference contract: "
        f"{field}={actual!r}, expected {expected!r}."
    )


def _validate_match_bundle_contract(
    *,
    config: MatchTrainerConfig,
    bundle: TorchModelBundle,
    batch_size: int,
    mu: int,
    sampler_contract: Mapping[str, Any],
    augmentation_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Authenticate the executable Match stack, not merely its YAML label."""

    torch = optional_import("torch", extra="inductive-torch")
    from modssc.inductive.deep.wide_resnet import WideResNetCifar

    reference = config.reference_implementation

    if int(batch_size) != 64:
        _match_contract_error("batch_size", batch_size, 64)
    if int(mu) != 7:
        _match_contract_error("mu", mu, 7)
    if sampler_contract.get("mode") != config.sampler_mode:
        _match_contract_error(
            "sampler.mode",
            sampler_contract.get("mode"),
            config.sampler_mode,
        )
    if not isinstance(bundle.model, WideResNetCifar):
        _match_contract_error(
            "model.type",
            f"{type(bundle.model).__module__}.{type(bundle.model).__qualname__}",
            "modssc.inductive.deep.wide_resnet.WideResNetCifar",
        )
    model = bundle.model
    expected_model = {
        "depth": 28,
        "widen_factor": 2,
        "in_channels": 3,
        "num_classes": 10,
        "reference_implementation": reference,
    }
    actual_model = {
        "depth": int(model.depth),
        "widen_factor": int(model.widen_factor),
        "in_channels": int(model.in_channels),
        "num_classes": int(model.classifier.out_features),
        "reference_implementation": str(model.reference_implementation),
    }
    for field, expected in expected_model.items():
        if actual_model[field] != expected:
            _match_contract_error(f"model.{field}", actual_model[field], expected)

    batch_norms = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    if not batch_norms:
        _match_contract_error("model.batch_norm_count", 0, "> 0")
    if any(float(module.momentum) != 0.001 for module in batch_norms):
        _match_contract_error(
            "model.bn_momentum",
            sorted({float(module.momentum) for module in batch_norms}),
            0.001,
        )
    if any(float(module.eps) != 0.001 for module in batch_norms):
        _match_contract_error(
            "model.bn_eps",
            sorted({float(module.eps) for module in batch_norms}),
            0.001,
        )

    expected_mean = None
    expected_std = None
    if reference == "torchssl":
        expected_mean = [0.4913725490196078, 0.4823529411764706, 0.44666666666666666]
        expected_std = [0.24705882352941178, 0.24352941176470588, 0.2615686274509804]
    actual_mean = (
        None if model.input_mean is None else model.input_mean.detach().cpu().reshape(-1).tolist()
    )
    actual_std = (
        None if model.input_std is None else model.input_std.detach().cpu().reshape(-1).tolist()
    )
    if expected_mean is None:
        if actual_mean is not None or actual_std is not None:
            _match_contract_error("model.input_normalization", (actual_mean, actual_std), None)
    else:
        if actual_mean is None or not np.allclose(actual_mean, expected_mean, rtol=0.0, atol=1e-7):
            _match_contract_error("model.input_mean", actual_mean, expected_mean)
        if actual_std is None or not np.allclose(actual_std, expected_std, rtol=0.0, atol=1e-7):
            _match_contract_error("model.input_std", actual_std, expected_std)

    if not isinstance(bundle.optimizer, torch.optim.SGD):
        _match_contract_error("optimizer.type", type(bundle.optimizer).__name__, "SGD")
    optimizer_defaults = bundle.optimizer.defaults
    expected_optimizer = {
        "lr": 0.03,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 0.0005,
        "decay_bias_and_norm": False,
    }
    for field in ("lr", "momentum"):
        if not math.isclose(
            float(optimizer_defaults[field]),
            expected_optimizer[field],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            _match_contract_error(
                f"optimizer.{field}", optimizer_defaults[field], expected_optimizer[field]
            )
    if bool(optimizer_defaults["nesterov"]) is not True:
        _match_contract_error("optimizer.nesterov", optimizer_defaults["nesterov"], True)
    group_decay = sorted(
        {float(group.get("weight_decay", 0.0)) for group in bundle.optimizer.param_groups}
    )
    if group_decay != [0.0, 0.0005]:
        _match_contract_error("optimizer.parameter_group_weight_decay", group_decay, [0.0, 0.0005])

    if bundle.scheduler is None or not isinstance(
        bundle.scheduler, torch.optim.lr_scheduler.LambdaLR
    ):
        _match_contract_error(
            "scheduler.type",
            None if bundle.scheduler is None else type(bundle.scheduler).__name__,
            "LambdaLR",
        )
    if bundle.ema_model is None or type(bundle.ema_model) is not type(model):
        _match_contract_error(
            "ema_model.type",
            None if bundle.ema_model is None else type(bundle.ema_model).__name__,
            type(model).__name__,
        )
    if set(bundle.ema_model.state_dict()) != set(model.state_dict()):
        _match_contract_error("ema_model.state_keys", "mismatch", "model state keys")
    if any(parameter.requires_grad for parameter in bundle.ema_model.parameters()):
        _match_contract_error("ema_model.requires_grad", True, False)

    meta = bundle.meta
    if not isinstance(meta, Mapping):
        _match_contract_error("bundle.meta", meta, "authenticated mapping")
    expected_initialization = (
        "google_fixmatch_variance_scaling"
        if reference == "google_fixmatch"
        else "torchssl_kaiming_normal"
    )
    expected_meta = {
        "contract_schema_version": 1,
        "classifier_id": "wide_resnet_cifar",
        "depth": 28,
        "widen_factor": 2,
        "in_channels": 3,
        "num_classes": 10,
        "bn_momentum": 0.001,
        "bn_eps": 0.001,
        "initialization": expected_initialization,
        "optimizer": "sgd",
        "lr": 0.03,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 0.0005,
        "scheduler": "cosine",
        "scheduler_step_unit": "optimizer_step",
        "max_steps": MATCH_REFERENCE_TARGET_STEPS,
        "cosine_cycles": 7.0 / 16.0,
        "ema_decay": 0.999,
        "predict_with_ema": True,
        "decay_bias_and_norm": False,
        "reference_implementation": reference,
        "ema_strategy": "parameters_only_copy_buffers",
    }
    for field, expected in expected_meta.items():
        actual = meta.get(field)
        equal = (
            math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
            if isinstance(expected, float) and isinstance(actual, (int, float))
            else actual == expected
        )
        if not equal:
            _match_contract_error(f"bundle.meta.{field}", actual, expected)

    contract = {
        "schema_version": 2,
        "training_mode": "fixed_steps",
        "reference_stack": reference,
        "architecture": actual_model,
        "initialization": expected_initialization,
        "optimizer": expected_optimizer,
        "scheduler": {
            "name": "cosine",
            "step_unit": "optimizer_step",
            "max_steps": MATCH_REFERENCE_TARGET_STEPS,
            "cycles": 7.0 / 16.0,
        },
        "ema": {
            "decay": 0.999,
            "strategy": "parameters_only_copy_buffers",
            "predict_with_ema": True,
        },
        "batches": {"labeled": 64, "unlabeled": 448, "mu": 7},
        "augmentation": dict(augmentation_contract),
        "augmentation_sha256": _canonical_sha256(augmentation_contract),
        "interleave_bn": config.interleave_bn,
        "evaluation_interval_steps": config.evaluation_interval_steps,
        "evaluation_tail_interval_steps": config.evaluation_tail_interval_steps,
        "evaluation_tail_start_fraction": config.evaluation_tail_start_fraction,
        "checkpoint_interval_steps": config.checkpoint_interval_steps,
        "reporting_policy": config.reporting_policy,
        "reporting_window_checkpoints": config.reporting_window_checkpoints,
        "sampler": dict(sampler_contract),
    }
    return contract, _canonical_sha256(contract)


def _tensor_group_sha256(values: Mapping[str, Any]) -> str:
    """Authenticate named tensor values without relying on serialization metadata."""

    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _cpu_state(value: Any) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        # Optimizer state dictionaries use integer parameter identifiers.  They
        # must remain integers so ``Optimizer.load_state_dict`` can remap them
        # to the freshly constructed model parameters on resume.
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    return value


def _checkpoint_identity(
    *,
    method_id: str,
    config: MatchTrainerConfig,
    data: Any,
    augmentation_contract: Mapping[str, Any],
) -> dict[str, Any]:
    meta = data.meta if isinstance(getattr(data, "meta", None), Mapping) else {}
    identity = {
        # Version 4 makes the independently recoverable TorchSSL ``model_best``
        # state part of the continuation contract. A version-3 checkpoint can
        # reproduce the scalar history but cannot authenticate the model that
        # attained it, so it must not silently resume under the stronger
        # replication contract.
        "schema_version": 4,
        "method_id": str(method_id),
        "training_mode": "fixed_steps",
        "reference_implementation": config.reference_implementation,
        "sampler_mode": config.sampler_mode,
        "augmentation_profile": config.augmentation_profile,
        "augmentation_runtime": dict(augmentation_contract),
        "augmentation_runtime_sha256": _canonical_sha256(augmentation_contract),
        "interleave_bn": config.interleave_bn,
        "evaluation_interval_steps": config.evaluation_interval_steps,
        "evaluation_tail_interval_steps": config.evaluation_tail_interval_steps,
        "evaluation_tail_start_fraction": config.evaluation_tail_start_fraction,
        "checkpoint_interval_steps": config.checkpoint_interval_steps,
        "reporting_policy": config.reporting_policy,
        "reporting_window_checkpoints": config.reporting_window_checkpoints,
        "dataset_fingerprint": meta.get("dataset_fingerprint"),
        "split_fingerprint": meta.get("split_fingerprint"),
        "partition_sha256": meta.get("partition_sha256"),
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def _resolve_match_augmenter(
    *,
    config: MatchTrainerConfig,
    data: Any,
) -> tuple[Any, dict[str, Any]]:
    """Resolve and authenticate the augmenter that fixed-step Match will run."""

    from modssc.data_augmentation.cifar_reference import (
        CIFAR_REFERENCE_AUGMENTER_ID,
        CIFAR_REFERENCE_CONTRACT_SCHEMA_VERSION,
        CifarReferenceAugmentation,
        cifar_reference_runtime_identity,
    )

    meta = data.meta if isinstance(getattr(data, "meta", None), Mapping) else {}
    configured = meta.get("online_augmentation")
    if configured is None:
        augmentation_seed = meta.get("augmentation_seed", 0)
        if not isinstance(augmentation_seed, int) or isinstance(augmentation_seed, bool):
            _match_contract_error("augmentation.seed", augmentation_seed, "integer")
        augmenter = CifarReferenceAugmentation(
            profile=config.augmentation_profile,
            seed=augmentation_seed,
        )
    else:
        augmenter = configured

    if type(augmenter) is not CifarReferenceAugmentation:
        actual_type = f"{type(augmenter).__module__}.{type(augmenter).__qualname__}"
        _match_contract_error(
            "augmentation.type",
            actual_type,
            ("modssc.data_augmentation.cifar_reference.CifarReferenceAugmentation"),
        )
    actual_profile = getattr(augmenter, "profile", None)
    if actual_profile != config.augmentation_profile:
        _match_contract_error(
            "augmentation.profile",
            actual_profile,
            config.augmentation_profile,
        )

    actual_seed = getattr(augmenter, "seed", None)
    if not isinstance(actual_seed, int) or isinstance(actual_seed, bool):
        _match_contract_error("augmentation.seed", actual_seed, "integer")
    identity = augmenter.runtime_identity()
    if not isinstance(identity, Mapping):
        _match_contract_error("augmentation.runtime_identity", identity, "mapping")
    canonical_identity = cifar_reference_runtime_identity(
        profile=config.augmentation_profile,
        seed=actual_seed,
    )
    expected = {
        "schema_version": CIFAR_REFERENCE_CONTRACT_SCHEMA_VERSION,
        "augmenter_id": CIFAR_REFERENCE_AUGMENTER_ID,
        "profile": config.augmentation_profile,
        "seed": actual_seed,
    }
    actual_config = identity.get("config")
    actual = {
        "schema_version": identity.get("schema_version"),
        "augmenter_id": identity.get("augmenter_id"),
        "profile": actual_config.get("profile") if isinstance(actual_config, Mapping) else None,
        "seed": actual_config.get("seed") if isinstance(actual_config, Mapping) else None,
    }
    for field, expected_value in expected.items():
        if actual[field] != expected_value:
            _match_contract_error(
                f"augmentation.runtime_identity.{field}",
                actual[field],
                expected_value,
            )
    if identity != canonical_identity:
        _match_contract_error(
            "augmentation.runtime_identity",
            identity,
            canonical_identity,
        )
    try:
        _canonical_sha256(identity)
    except (TypeError, ValueError) as exc:
        raise InductiveValidationError(
            "Match augmentation runtime identity must be JSON-serializable."
        ) from exc
    return augmenter, dict(identity)


class _CheckpointStore:
    """Thin Match serializer over the method-agnostic native checkpoint store."""

    def __init__(
        self,
        *,
        identity: Mapping[str, Any],
        context: ExecutionContext | None,
    ) -> None:
        self.identity = dict(identity)
        self.history: list[dict[str, Any]] = []
        self.context = context
        self.store = (
            CheckpointStore.from_context(context)
            if context is not None and context.resume_policy != "never"
            else None
        )

    @property
    def enabled(self) -> bool:
        return self.store is not None

    @staticmethod
    def _serialize(payload: Any) -> bytes:
        torch = optional_import("torch", extra="inductive-torch")
        stream = io.BytesIO()
        torch.save(payload, stream)
        return stream.getvalue()

    @staticmethod
    def _deserialize(payload: bytes) -> Any:
        torch = optional_import("torch", extra="inductive-torch")
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)

    def save(self, payload: Mapping[str, Any], *, step: int, reason: str) -> None:
        if self.store is None:
            return
        try:
            checkpoint_payload = dict(payload)
            checkpoint_payload["checkpoint_history"] = list(self.history)
            record = self.store.save(
                checkpoint_payload,
                step=int(step),
                reason=str(reason),
                serializer=self._serialize,
            )
            self.store.prune(keep_last=1)
        except (CheckpointError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise InductiveValidationError("Match checkpoint save failed.") from exc
        self.history.append(
            {
                "step": record.step,
                "reason": record.reason,
                "payload_sha256": record.payload_sha256,
            }
        )

    def load(self) -> dict[str, Any] | None:
        if self.store is None or self.context is None:
            return None
        try:
            loaded = self.store.load_from_context(
                self.context,
                deserializer=self._deserialize,
            )
        except (CheckpointError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise InductiveValidationError("Match checkpoint load failed.") from exc
        if loaded is None:
            return None
        payload = loaded.payload
        if not isinstance(payload, dict) or payload.get("identity") != self.identity:
            raise InductiveValidationError("Match checkpoint payload identity is incompatible.")
        raw_history = payload.pop("checkpoint_history", [])
        if isinstance(raw_history, list):
            self.history = [dict(item) for item in raw_history if isinstance(item, Mapping)]
        current_event = {
            "step": loaded.record.step,
            "reason": loaded.record.reason,
            "payload_sha256": loaded.record.payload_sha256,
        }
        if not self.history or self.history[-1] != current_event:
            self.history.append(current_event)
        return payload


def _rng_state(torch: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(torch: Any, state: Mapping[str, Any]) -> None:
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise InductiveValidationError("Match checkpoint RNG state is invalid.") from exc


def _bundle_state(bundle: TorchModelBundle) -> dict[str, Any]:
    state: dict[str, Any] = {
        "model": _cpu_state(bundle.model.state_dict()),
        "optimizer": _cpu_state(bundle.optimizer.state_dict()),
        "ema_model": (
            _cpu_state(bundle.ema_model.state_dict()) if bundle.ema_model is not None else None
        ),
        "scheduler": (
            _cpu_state(bundle.scheduler.state_dict()) if bundle.scheduler is not None else None
        ),
        "scaler": _cpu_state(bundle.scaler.state_dict()) if bundle.scaler is not None else None,
    }
    return state


def _clone_tensor_state(value: Any) -> Any:
    """Recursively detach, move to CPU, and clone retained tensor state."""

    torch = optional_import("torch", extra="inductive-torch")
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_tensor_state(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tensor_state(item) for item in value)
    if isinstance(value, list):
        return [_clone_tensor_state(item) for item in value]
    return value


def _capture_best_historical_checkpoint(
    bundle: TorchModelBundle,
    *,
    event: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the bundle that produced one TorchSSL best-test observation.

    Torch ``state_dict`` tensors share storage with a live CPU model. The
    recursive ``_cpu_state`` conversion used for immediate serialization is
    therefore not sufficient for a snapshot retained while training
    continues; clone the complete bundle before keeping it.
    """

    step = int(event["step"])
    accuracy = float(event["test_accuracy"])
    error_percent = float(event["test_error_percent"])
    if step <= 0 or not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise InductiveValidationError("Match best historical checkpoint event is invalid.")
    if not math.isclose(
        error_percent,
        100.0 * (1.0 - accuracy),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise InductiveValidationError("Match best historical checkpoint error is invalid.")

    frozen = _clone_tensor_state(_bundle_state(bundle))
    model_state = frozen.get("model")
    ema_state = frozen.get("ema_model")
    if not isinstance(model_state, Mapping):
        raise InductiveValidationError("Match best historical model state is invalid.")
    if ema_state is not None and not isinstance(ema_state, Mapping):
        raise InductiveValidationError("Match best historical EMA state is invalid.")
    return {
        "schema_version": 1,
        "step": step,
        "test_accuracy": accuracy,
        "test_error_percent": error_percent,
        "model_sha256": _tensor_group_sha256(model_state),
        "ema_model_sha256": (None if ema_state is None else _tensor_group_sha256(ema_state)),
        "bundle": frozen,
    }


def _validate_best_historical_checkpoint(
    raw: Any,
    *,
    history: list[dict[str, Any]],
    reporting_policy: MatchReportingPolicy,
) -> dict[str, Any] | None:
    """Validate a retained best bundle against its authenticated metric history."""

    eligible = [item for item in history if item.get("historical_eligible") is True]
    expected: dict[str, Any] | None = None
    for event in eligible:
        accuracy = float(event["test_accuracy"])
        if expected is None or accuracy > float(expected["test_accuracy"]):
            expected = event

    if reporting_policy != "best_historical_checkpoint":
        if raw is not None:
            raise InductiveValidationError(
                "Match checkpoint unexpectedly contains a best historical bundle."
            )
        return None
    if expected is None:
        if raw is not None:
            raise InductiveValidationError(
                "Match checkpoint best historical bundle has no eligible evaluation."
            )
        return None
    if not isinstance(raw, Mapping):
        raise InductiveValidationError("Match checkpoint best historical bundle is missing.")

    required = {
        "schema_version",
        "step",
        "test_accuracy",
        "test_error_percent",
        "model_sha256",
        "ema_model_sha256",
        "bundle",
    }
    if set(raw) != required or raw.get("schema_version") != 1:
        raise InductiveValidationError("Match checkpoint best historical schema is invalid.")
    try:
        step = int(raw["step"])
        accuracy = float(raw["test_accuracy"])
        error_percent = float(raw["test_error_percent"])
    except (TypeError, ValueError) as exc:
        raise InductiveValidationError(
            "Match checkpoint best historical metadata is invalid."
        ) from exc
    if step != int(expected["step"]) or not math.isclose(
        accuracy,
        float(expected["test_accuracy"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise InductiveValidationError(
            "Match checkpoint best historical bundle disagrees with evaluation history."
        )
    if (
        not math.isfinite(accuracy)
        or not 0.0 <= accuracy <= 1.0
        or not math.isclose(
            error_percent,
            100.0 * (1.0 - accuracy),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise InductiveValidationError("Match checkpoint best historical metric is invalid.")

    bundle = raw["bundle"]
    if not isinstance(bundle, Mapping):
        raise InductiveValidationError("Match checkpoint best historical state is invalid.")
    model_state = bundle.get("model")
    ema_state = bundle.get("ema_model")
    if not isinstance(model_state, Mapping):
        raise InductiveValidationError("Match checkpoint best historical model is invalid.")
    if ema_state is not None and not isinstance(ema_state, Mapping):
        raise InductiveValidationError("Match checkpoint best historical EMA is invalid.")
    model_sha256 = _tensor_group_sha256(model_state)
    ema_sha256 = None if ema_state is None else _tensor_group_sha256(ema_state)
    if raw["model_sha256"] != model_sha256 or raw["ema_model_sha256"] != ema_sha256:
        raise InductiveValidationError("Match checkpoint best historical state digest is invalid.")
    return dict(raw)


def _best_historical_checkpoint_summary(
    checkpoint: Mapping[str, Any] | None,
    *,
    durable: bool,
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    return {
        "schema_version": int(checkpoint["schema_version"]),
        "step": int(checkpoint["step"]),
        "test_accuracy": float(checkpoint["test_accuracy"]),
        "test_error_percent": float(checkpoint["test_error_percent"]),
        "model_sha256": str(checkpoint["model_sha256"]),
        "ema_model_sha256": checkpoint["ema_model_sha256"],
        "storage": "native_checkpoint_payload" if durable else "fitted_method_memory",
        "active_model_role": "terminal_model",
    }


def _load_bundle_state(bundle: TorchModelBundle, state: Mapping[str, Any]) -> None:
    try:
        bundle.model.load_state_dict(state["model"])
        bundle.optimizer.load_state_dict(state["optimizer"])
        if bundle.ema_model is not None:
            if state.get("ema_model") is None:
                raise KeyError("ema_model")
            bundle.ema_model.load_state_dict(state["ema_model"])
        elif state.get("ema_model") is not None:
            raise InductiveValidationError("Checkpoint contains an unexpected EMA model.")
        if bundle.scheduler is not None:
            if state.get("scheduler") is None:
                raise KeyError("scheduler")
            bundle.scheduler.load_state_dict(state["scheduler"])
        elif state.get("scheduler") is not None:
            raise InductiveValidationError("Checkpoint contains an unexpected scheduler.")
        if bundle.scaler is not None:
            if state.get("scaler") is None:
                raise KeyError("scaler")
            bundle.scaler.load_state_dict(state["scaler"])
        elif state.get("scaler") is not None:
            raise InductiveValidationError("Checkpoint contains an unexpected scaler.")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        if isinstance(exc, InductiveValidationError):
            raise
        raise InductiveValidationError("Match checkpoint model state is invalid.") from exc


def _tensor_indices(indices: Any, *, device: Any) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    if isinstance(indices, torch.Tensor):
        return indices.to(device=device, dtype=torch.int64)
    return torch.as_tensor(np.asarray(indices, dtype=np.int64), device=device, dtype=torch.int64)


def _reference_batch_views(
    *,
    augmentation_profile: str,
    data: Any,
    X_l: Any,
    X_u_w: Any,
    X_u_s: Any,
    idx_l: Any,
    idx_u: Any,
    step: int,
    augmenter: Any | None = None,
) -> tuple[Any, Any, Any]:
    meta = data.meta if isinstance(getattr(data, "meta", None), Mapping) else {}
    if int(getattr(X_l, "ndim", 0)) != 4:
        return ssl_batch_views(
            data,
            X_l=X_l,
            X_u_w=X_u_w,
            X_u_s=X_u_s,
            idx_l=idx_l,
            idx_u=idx_u,
            optimization_step=int(step),
        )

    from modssc.data_augmentation.cifar_reference import CifarReferenceAugmentation

    configured = meta.get("online_augmentation")
    augmentation_seed = int(getattr(configured, "seed", meta.get("augmentation_seed", 0)))
    if augmenter is None:
        augmenter = CifarReferenceAugmentation(
            profile=augmentation_profile,
            seed=augmentation_seed,
        )
    if not hasattr(augmenter, "apply_batch"):
        return ssl_batch_views(
            data,
            X_l=X_l,
            X_u_w=X_u_w,
            X_u_s=X_u_s,
            idx_l=idx_l,
            idx_u=idx_u,
            optimization_step=int(step),
        )

    x_lb = slice_data(X_l, idx_l)
    X_u = getattr(data, "X_u", None)
    if X_u is None:
        X_u = X_u_w
    x_u = slice_data(X_u, idx_u)

    idx_l_all = meta.get("source_idx_l", meta.get("idx_l"))
    idx_u_all = meta.get("source_idx_u", meta.get("idx_u"))
    if idx_l_all is None or idx_u_all is None:
        raise InductiveValidationError("Paper Match augmentation requires source sample indices.")
    sample_l = slice_data(idx_l_all, idx_l)
    sample_u = slice_data(idx_u_all, idx_u)
    return (
        augmenter.apply_batch(
            x_lb,
            sample_ids=sample_l,
            step=int(step),
            view="labeled_weak",
        ),
        augmenter.apply_batch(
            x_u,
            sample_ids=sample_u,
            step=int(step),
            view="unlabeled_weak",
        ),
        augmenter.apply_batch(
            x_u,
            sample_ids=sample_u,
            step=int(step),
            view="unlabeled_strong",
        ),
    )


def _forward_match(
    model: Any,
    *,
    x_lb: Any,
    x_uw: Any,
    x_us: Any,
    interleave_bn: bool,
    mu: int,
) -> tuple[Any, Any, Any]:
    if interleave_bn:
        inputs = concat_data([x_lb, x_uw, x_us])
        inputs = interleave_batch(inputs, groups=2 * int(mu) + 1)
        logits = extract_logits(model(inputs))
        logits = deinterleave_batch(logits, groups=2 * int(mu) + 1)
    else:
        logits = extract_logits(model(concat_data([x_lb, x_uw, x_us])))
    if int(logits.ndim) != 2:
        raise InductiveValidationError("Model logits must be 2D (batch, classes).")
    num_lb = int(get_torch_len(x_lb))
    num_u = int(get_torch_len(x_uw))
    expected = num_lb + num_u + int(get_torch_len(x_us))
    if int(logits.shape[0]) != expected:
        raise InductiveValidationError("Concatenated Match logits do not preserve batch size.")
    logits_l = logits[:num_lb]
    logits_uw = logits[num_lb : num_lb + num_u]
    logits_us = logits[num_lb + num_u :]
    if logits_uw.shape != logits_us.shape:
        raise InductiveValidationError("Unlabeled Match logits shape mismatch.")
    if int(logits_l.shape[1]) != int(logits_uw.shape[1]):
        raise InductiveValidationError("Match logits disagree on class dimension.")
    return logits_l, logits_uw, logits_us


def _evaluation_payload(data: Any) -> tuple[Any, Any] | None:
    meta = data.meta if isinstance(getattr(data, "meta", None), Mapping) else {}
    splits = meta.get("evaluation_splits")
    payload = splits.get("test") if isinstance(splits, Mapping) else None
    if not isinstance(payload, Mapping):
        return None
    X = payload.get("X")
    y = payload.get("y")
    if X is None or y is None:
        return None
    return X, y


def _accuracy(
    bundle: TorchModelBundle,
    *,
    X: Any,
    y: Any,
    batch_size: int = 1024,
) -> float:
    torch = optional_import("torch", extra="inductive-torch")
    model = bundle.ema_model if bundle.ema_model is not None else bundle.model
    was_training = bool(model.training)
    model.eval()
    correct = 0
    total = int(get_torch_len(X))
    if total <= 0:
        raise InductiveValidationError("Paper evaluation test split is empty.")
    with torch.no_grad():
        for start in range(0, total, int(batch_size)):
            stop = min(start + int(batch_size), total)
            logits = extract_logits(model(slice_data(X, slice(start, stop))))
            target = slice_data(y, slice(start, stop))
            if not isinstance(target, torch.Tensor):
                target = torch.as_tensor(target, device=logits.device, dtype=torch.int64)
            else:
                target = target.to(device=logits.device, dtype=torch.int64)
            correct += int((logits.argmax(dim=1) == target).sum().item())
    model.train(was_training)
    return float(correct) / float(total)


def _paper_metrics(
    *,
    reporting_policy: MatchReportingPolicy,
    reporting_window_checkpoints: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    if not history:
        return {
            "historical_paper_metric": None,
            "fixed_terminal_metric": None,
            "selection_uses_test": False,
        }
    terminal = float(history[-1]["test_accuracy"])
    if reporting_policy == "median_last_checkpoints":
        accuracies = [float(item["test_accuracy"]) for item in history]
        window = accuracies[-int(reporting_window_checkpoints) :]
        historical_accuracy = float(np.median(np.asarray(window, dtype=np.float64)))
        policy = f"median_last_{int(reporting_window_checkpoints)}_checkpoints"
        selection_uses_test = False
    else:
        historical = [
            float(item["test_accuracy"])
            for item in history
            if item.get("historical_eligible") is True
        ]
        if not historical:
            raise InductiveValidationError(
                "best historical checkpoint evaluation requires an eligible event."
            )
        historical_accuracy = max(historical)
        policy = "best_test_checkpoint_historical_only"
        selection_uses_test = True
    return {
        "historical_policy": policy,
        "historical_paper_metric": {
            "test_accuracy": historical_accuracy,
            "test_error_percent": 100.0 * (1.0 - historical_accuracy),
        },
        "fixed_terminal_metric": {
            "test_accuracy": terminal,
            "test_error_percent": 100.0 * (1.0 - terminal),
        },
        "selection_uses_test": selection_uses_test,
        "benchmark_eligible_metric": "fixed_terminal_metric",
    }


def _match_evaluation_metric_sets(
    paper_metrics: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Expose paper and terminal statistics as explicit native metric sets."""

    metric_sets: dict[str, dict[str, Any]] = {}
    terminal = paper_metrics.get("fixed_terminal_metric")
    if isinstance(terminal, Mapping):
        metric_sets["terminal"] = {
            "test": {
                "accuracy": float(terminal["test_accuracy"]),
                "error_percent": float(terminal["test_error_percent"]),
                "role": "terminal_checkpoint",
                "benchmark_eligible": (
                    paper_metrics.get("benchmark_eligible_metric") == "fixed_terminal_metric"
                ),
            }
        }
    historical = paper_metrics.get("historical_paper_metric")
    if isinstance(historical, Mapping):
        metric_sets["reported"] = {
            "test": {
                "accuracy": float(historical["test_accuracy"]),
                "error_percent": float(historical["test_error_percent"]),
                "role": "paper_reported_historical_statistic",
                "policy": str(paper_metrics.get("historical_policy", "unknown")),
                "selection_uses_test": bool(paper_metrics.get("selection_uses_test", False)),
                "benchmark_eligible": False,
            }
        }
    return metric_sets


def _continuation_requested() -> bool:
    return continuation_requested()


def _forced_continuation_step(*, allow_short_run: bool, target_steps: int) -> int | None:
    raw = os.environ.get("MODSSC_FORCE_CONTINUATION_STEP")
    if raw is None:
        return None
    if not allow_short_run:
        raise InductiveValidationError(
            "MODSSC_FORCE_CONTINUATION_STEP requires allow_short_run=true."
        )
    try:
        step = int(raw)
    except ValueError as exc:
        raise InductiveValidationError(
            "MODSSC_FORCE_CONTINUATION_STEP must be an integer."
        ) from exc
    if step <= 0 or step >= int(target_steps):
        raise InductiveValidationError(
            "MODSSC_FORCE_CONTINUATION_STEP must be inside the configured run."
        )
    return step


@contextmanager
def _paper_deterministic_runtime(torch: Any) -> Iterator[dict[str, Any]]:
    """Scope and describe the deterministic numeric runtime for one paper run."""

    deterministic_algorithms = bool(torch.are_deterministic_algorithms_enabled())
    deterministic_warn_only = bool(torch.is_deterministic_algorithms_warn_only_enabled())
    cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
    cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    candidates = (
        ("global", getattr(torch, "backends", None)),
        ("cuda_matmul", getattr(getattr(torch.backends, "cuda", None), "matmul", None)),
        ("cudnn", getattr(torch.backends, "cudnn", None)),
        ("cudnn_conv", getattr(getattr(torch.backends, "cudnn", None), "conv", None)),
    )
    precision_targets = tuple(
        (name, owner, "fp32_precision")
        for name, owner in candidates
        if owner is not None and hasattr(owner, "fp32_precision")
    )
    legacy_tf32_targets = tuple(
        (name, owner, "allow_tf32")
        for name, owner in candidates
        if owner is not None
        and not hasattr(owner, "fp32_precision")
        and hasattr(owner, "allow_tf32")
    )
    precision_state = {
        name: getattr(owner, attribute) for name, owner, attribute in precision_targets
    }
    legacy_tf32_state = {
        name: bool(getattr(owner, attribute)) for name, owner, attribute in legacy_tf32_targets
    }
    get_matmul_precision = getattr(torch, "get_float32_matmul_precision", None)
    set_matmul_precision = getattr(torch, "set_float32_matmul_precision", None)
    matmul_precision_state = (
        get_matmul_precision()
        if callable(get_matmul_precision) and callable(set_matmul_precision)
        else None
    )
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        for _, owner, attribute in precision_targets:
            setattr(owner, attribute, "ieee")
        for _, owner, attribute in legacy_tf32_targets:
            setattr(owner, attribute, False)
        if matmul_precision_state is not None:
            set_matmul_precision("highest")
        yield {
            "schema_version": 1,
            "deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
            "float32_precision": {
                name: getattr(owner, attribute) for name, owner, attribute in precision_targets
            },
            "legacy_allow_tf32": {
                name: bool(getattr(owner, attribute))
                for name, owner, attribute in legacy_tf32_targets
            },
            "matmul_precision": (
                get_matmul_precision() if matmul_precision_state is not None else None
            ),
        }
    finally:
        for name, owner, attribute in precision_targets:
            setattr(owner, attribute, precision_state[name])
        for name, owner, attribute in legacy_tf32_targets:
            setattr(owner, attribute, legacy_tf32_state[name])
        if matmul_precision_state is not None:
            set_matmul_precision(matmul_precision_state)
        torch.use_deterministic_algorithms(
            deterministic_algorithms,
            warn_only=deterministic_warn_only,
        )
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if cublas_workspace_config is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config


def _historical_evaluation(
    *,
    global_step: int,
    target_steps: int,
    interval: int,
    tail_interval: int | None = 1000,
    tail_start_fraction: float | None = 0.8,
) -> bool:
    """Apply the TorchSSL historical evaluation cadence.

    TorchSSL evaluates after the update while its zero-based ``self.it`` is
    still unchanged.  Once the incremented counter is beyond 80 percent of
    training, subsequent iterations use a 1000-step cadence.  Keeping this
    off-by-one explicit is required for the historical best-test statistic.
    """

    step = int(global_step)
    target = int(target_steps)
    if not 0 <= step < target:
        return False
    use_tail = (
        tail_interval is not None
        and tail_start_fraction is not None
        and step > float(tail_start_fraction) * target
    )
    active_interval = int(tail_interval) if use_tail else int(interval)
    return step % active_interval == 0


def _raise_continuation() -> None:
    raise PlannedContinuation()


def _run_fixed_step_match(
    owner: Any,
    data: Any,
    *,
    device: DeviceSpec,
    seed: int,
    method_id: str,
    step_fn: StepFunction,
    state_getter: StateGetter,
    state_loader: StateLoader,
    trace_getter: TraceGetter | None = None,
    _enforce_reference_contract: bool = True,
) -> MatchTrainingResult:
    """Run one explicit fixed-step contract through the resumable Match loop."""

    spec = owner.spec
    config = _trainer_config(spec)
    if data is None:
        raise InductiveValidationError("data must not be None.")
    backend = detect_backend(data.X_l)
    if backend != "torch":
        raise InductiveValidationError("Fixed-step Match training requires torch tensors.")
    ds = ensure_torch_data(data, device=device)
    torch = optional_import("torch", extra="inductive-torch")
    if ds.X_u_w is None or ds.X_u_s is None:
        raise InductiveValidationError("Fixed-step Match training requires weak and strong views.")
    X_l = ds.X_l
    y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
    X_u_w = ds.X_u_w
    X_u_s = ds.X_u_s
    n_l = int(get_torch_len(X_l))
    n_u = int(get_torch_len(X_u_w))
    if n_l <= 0 or n_u <= 0:
        raise InductiveValidationError("Match labeled and unlabeled pools must be non-empty.")
    if n_u != int(get_torch_len(X_u_s)):
        raise InductiveValidationError("Match weak and strong pools must have equal size.")
    ensure_float_tensor(X_l, name="X_l")
    ensure_float_tensor(X_u_w, name="X_u_w")
    ensure_float_tensor(X_u_s, name="X_u_s")
    if y_l.dtype != torch.int64:
        raise InductiveValidationError("y_l must be int64 for fixed-step Match training.")

    if spec.model_bundle is None:
        raise InductiveValidationError(
            "model_bundle must be provided for fixed-step Match training."
        )
    bundle = ensure_model_bundle(spec.model_bundle)
    ensure_model_device(bundle.model, device=get_torch_device(X_l))

    batch_size = int(spec.batch_size)
    mu = int(spec.mu)
    target_steps = int(spec.max_steps) if spec.max_steps is not None else 0
    if batch_size != 64 or mu != 7:
        raise InductiveValidationError("The Match reference contract requires batches 64/448.")
    if config.allow_short_run:
        if target_steps <= 0 or target_steps > MATCH_REFERENCE_TARGET_STEPS:
            raise InductiveValidationError("Short Match max_steps must be inside (0, 2^20].")
    elif target_steps != MATCH_REFERENCE_TARGET_STEPS:
        raise InductiveValidationError(
            "The full Match reference contract requires exactly 2^20 optimization steps."
        )
    if not _enforce_reference_contract and not config.allow_short_run:
        raise InductiveValidationError(
            "The Match reference contract can only be disabled for an explicit short run."
        )
    forced_continuation_step = _forced_continuation_step(
        allow_short_run=config.allow_short_run,
        target_steps=target_steps,
    )

    reference = config.reference_implementation
    reference_augmenter = None
    if int(getattr(X_l, "ndim", 0)) == 4:
        reference_augmenter, augmentation_contract = _resolve_match_augmenter(
            config=config,
            data=ds,
        )
    else:
        if _enforce_reference_contract:
            raise InductiveValidationError(
                "The Match reference contract requires 4D CIFAR image tensors."
            )
        meta = ds.meta if isinstance(getattr(ds, "meta", None), Mapping) else {}
        if meta.get("online_augmentation") is not None:
            raise InductiveValidationError(
                "The internal non-image Match harness accepts only precomputed SSL views."
            )
        augmentation_contract = {
            "schema_version": 1,
            "augmenter_id": "internal.precomputed_ssl_views",
            "implementation": "modssc.inductive.types.InductiveDataset",
            "config": {"weak": "X_u_w", "strong": "X_u_s"},
        }
    sampler = FixedSSLBatchSampler(
        n_l,
        n_u,
        labeled_batch_size=batch_size,
        unlabeled_batch_size=batch_size * mu,
        seed=int(seed),
        mode=config.sampler_mode,
        shuffle_buffer=config.sampler_shuffle_buffer,
    )
    sampler_contract = sampler.contract()
    if _enforce_reference_contract:
        match_contract, match_contract_sha256 = _validate_match_bundle_contract(
            config=config,
            bundle=bundle,
            batch_size=batch_size,
            mu=mu,
            sampler_contract=sampler_contract,
            augmentation_contract=augmentation_contract,
        )
    else:
        match_contract = {
            "schema_version": 2,
            "training_mode": "fixed_steps",
            "reference_stack": reference,
            "reference_bundle_validation": "internal_diagnostic_harness",
            "batches": {
                "labeled": batch_size,
                "unlabeled": batch_size * mu,
                "mu": mu,
            },
            "augmentation": dict(augmentation_contract),
            "augmentation_sha256": _canonical_sha256(augmentation_contract),
            "sampler": dict(sampler_contract),
        }
        match_contract_sha256 = _canonical_sha256(match_contract)
    identity = _checkpoint_identity(
        method_id=method_id,
        config=config,
        data=ds,
        augmentation_contract=augmentation_contract,
    )
    checkpoint_store = _CheckpointStore(
        identity=identity,
        context=getattr(ds, "execution_context", None),
    )
    checkpoint_payload = checkpoint_store.load()
    evaluation_history: list[dict[str, Any]] = []
    best_historical_checkpoint: dict[str, Any] | None = None
    numeric_probe: dict[str, Any] = {}
    accepted = 0.0
    unlabeled = 0
    start_step = 0
    if checkpoint_payload is not None:
        _load_bundle_state(bundle, checkpoint_payload["bundle"])
        sampler.load_state_dict(checkpoint_payload["sampler"])
        state_loader(checkpoint_payload["method_state"])
        _restore_rng_state(torch, checkpoint_payload["rng"])
        start_step = int(checkpoint_payload["next_step"])
        accepted = float(checkpoint_payload.get("accepted", 0.0))
        unlabeled = int(checkpoint_payload.get("unlabeled", 0))
        raw_history = checkpoint_payload.get("evaluation_history", [])
        if not isinstance(raw_history, list):
            raise InductiveValidationError("Match checkpoint evaluation history is invalid.")
        evaluation_history = [dict(item) for item in raw_history]
        best_historical_checkpoint = _validate_best_historical_checkpoint(
            checkpoint_payload.get("best_historical_checkpoint"),
            history=evaluation_history,
            reporting_policy=config.reporting_policy,
        )
        if start_step < 0 or start_step > target_steps:
            raise InductiveValidationError("Match checkpoint step is outside the configured run.")
        numeric_probe = dict(checkpoint_payload.get("numeric_probe", {}))
    else:
        random.seed(int(seed))
        np.random.seed(int(seed) % (1 << 32))
        torch.manual_seed(int(seed))
        numeric_probe["initial_model_sha256"] = _tensor_group_sha256(bundle.model.state_dict())

    eval_interval = config.evaluation_interval_steps
    checkpoint_interval = config.checkpoint_interval_steps
    if config.allow_short_run and target_steps < checkpoint_interval:
        checkpoint_interval = max(1, target_steps // 2)
    paper_eval = _evaluation_payload(ds)
    bundle.model.train()

    def make_checkpoint(*, next_step: int) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "identity": identity,
            "next_step": int(next_step),
            "target_steps": int(target_steps),
            "bundle": _bundle_state(bundle),
            "sampler": _cpu_state(sampler.state_dict()),
            "method_state": _cpu_state(state_getter()),
            "rng": _rng_state(torch),
            "accepted": float(accepted),
            "unlabeled": int(unlabeled),
            "evaluation_history": list(evaluation_history),
            "best_historical_checkpoint": best_historical_checkpoint,
            "numeric_probe": dict(numeric_probe),
        }

    for global_step in range(start_step, target_steps):
        batch = sampler.next_batch()
        idx_l = _tensor_indices(batch.labeled, device=get_torch_device(X_l))
        idx_u = _tensor_indices(batch.unlabeled, device=get_torch_device(X_u_w))
        x_lb, x_uw, x_us = _reference_batch_views(
            augmentation_profile=config.augmentation_profile,
            data=ds,
            X_l=X_l,
            X_u_w=X_u_w,
            X_u_s=X_u_s,
            idx_l=idx_l,
            idx_u=idx_u,
            step=global_step,
            augmenter=reference_augmenter,
        )
        y_lb = y_l[idx_l]
        logits_l, logits_uw, logits_us = _forward_match(
            bundle.model,
            x_lb=x_lb,
            x_uw=x_uw,
            x_us=x_us,
            interleave_bn=config.interleave_bn,
            mu=mu,
        )
        if y_lb.min().item() < 0 or y_lb.max().item() >= int(logits_l.shape[1]):
            raise InductiveValidationError("Match labels must be within [0, n_classes).")
        step_result = step_fn(logits_l, logits_uw, logits_us, y_lb, idx_u)
        if not isinstance(step_result, MatchStepResult):
            raise InductiveValidationError("Match step hook returned an invalid result.")
        if int(step_result.unlabeled) != batch_size * mu:
            raise InductiveValidationError("Match step did not consume exactly 448 examples.")
        if global_step == 0:
            numeric_probe.update(
                {
                    "batch_indices_sha256": _tensor_group_sha256(
                        {"labeled": idx_l, "unlabeled": idx_u}
                    ),
                    "augmented_inputs_sha256": _tensor_group_sha256(
                        {"labeled": x_lb, "unlabeled_strong": x_us, "unlabeled_weak": x_uw}
                    ),
                    "logits_sha256": _tensor_group_sha256(
                        {
                            "labeled": logits_l,
                            "unlabeled_strong": logits_us,
                            "unlabeled_weak": logits_uw,
                        }
                    ),
                    "loss": float(step_result.loss.detach().item()),
                    "accepted": float(step_result.accepted),
                    "step_diagnostics": dict(step_result.diagnostics),
                }
            )
            if method_id == "fixmatch":
                probabilities = torch.softmax(logits_uw.detach(), dim=1)
                confidence, pseudo_label = probabilities.max(dim=1)
                numeric_probe["pseudo_label_mask_sha256"] = _tensor_group_sha256(
                    {
                        "mask": confidence >= 0.95,
                        "pseudo_label": pseudo_label,
                    }
                )
        bundle.optimizer.zero_grad()
        step_result.loss.backward()
        optimizer_step_with_bundle(bundle)
        accepted += float(step_result.accepted)
        unlabeled += int(step_result.unlabeled)
        completed_step = global_step + 1

        historical_eligible = (
            completed_step % eval_interval == 0
            if config.reporting_policy == "median_last_checkpoints"
            else _historical_evaluation(
                global_step=global_step,
                target_steps=target_steps,
                interval=eval_interval,
                tail_interval=config.evaluation_tail_interval_steps,
                tail_start_fraction=config.evaluation_tail_start_fraction,
            )
        )
        should_evaluate = paper_eval is not None and (
            historical_eligible
            or (
                config.reporting_policy == "median_last_checkpoints"
                and completed_step == target_steps
            )
        )
        if should_evaluate:
            X_test, y_test = paper_eval
            event: dict[str, Any] = {
                "step": int(completed_step),
                "test_accuracy": _accuracy(bundle, X=X_test, y=y_test),
                "historical_eligible": bool(historical_eligible),
            }
            event["test_error_percent"] = 100.0 * (1.0 - float(event["test_accuracy"]))
            event["step_diagnostics"] = dict(step_result.diagnostics)
            if trace_getter is not None:
                event["method_state"] = dict(trace_getter())
            evaluation_history.append(event)
            if (
                config.reporting_policy == "best_historical_checkpoint"
                and historical_eligible
                and (
                    best_historical_checkpoint is None
                    or float(event["test_accuracy"])
                    > float(best_historical_checkpoint["test_accuracy"])
                )
            ):
                best_historical_checkpoint = _capture_best_historical_checkpoint(
                    bundle,
                    event=event,
                )

        periodic = completed_step % checkpoint_interval == 0
        forced_continuation = (
            forced_continuation_step is not None
            and start_step < forced_continuation_step
            and completed_step >= forced_continuation_step
        )
        continuation = (
            _continuation_requested() or forced_continuation
        ) and completed_step < target_steps
        defer_complete_checkpoint = (
            config.reporting_policy == "best_historical_checkpoint"
            and completed_step == target_steps
        )
        if (periodic or continuation or completed_step == target_steps) and not (
            defer_complete_checkpoint
        ):
            reason = (
                "planned_continuation"
                if continuation
                else ("complete" if completed_step == target_steps else "periodic")
            )
            checkpoint_store.save(
                make_checkpoint(next_step=completed_step),
                step=completed_step,
                reason=reason,
            )
        if continuation:
            _raise_continuation()

    if config.reporting_policy == "best_historical_checkpoint":
        terminal_already_recorded = any(
            item.get("terminal_evaluation") is True and int(item.get("step", -1)) == target_steps
            for item in evaluation_history
        )
        if paper_eval is not None and not terminal_already_recorded:
            X_test, y_test = paper_eval
            terminal_event: dict[str, Any] = {
                "step": int(target_steps),
                "test_accuracy": _accuracy(bundle, X=X_test, y=y_test),
                "historical_eligible": False,
                "terminal_evaluation": True,
            }
            terminal_event["test_error_percent"] = 100.0 * (
                1.0 - float(terminal_event["test_accuracy"])
            )
            if trace_getter is not None:
                terminal_event["method_state"] = dict(trace_getter())
            evaluation_history.append(terminal_event)
        best_historical_checkpoint = _validate_best_historical_checkpoint(
            best_historical_checkpoint,
            history=evaluation_history,
            reporting_policy=config.reporting_policy,
        )
        if start_step < target_steps or not terminal_already_recorded:
            checkpoint_store.save(
                make_checkpoint(next_step=target_steps),
                step=target_steps,
                reason="complete",
            )

    metrics = _paper_metrics(
        reporting_policy=config.reporting_policy,
        reporting_window_checkpoints=config.reporting_window_checkpoints,
        history=evaluation_history,
    )
    best_historical_summary = _best_historical_checkpoint_summary(
        best_historical_checkpoint,
        durable=checkpoint_store.enabled,
    )
    owner.evaluation_metric_sets_ = _match_evaluation_metric_sets(metrics)
    result = MatchTrainingResult(
        optimization_steps=target_steps,
        target_steps=target_steps,
        accepted=accepted,
        unlabeled=unlabeled,
        evaluation_history=tuple(evaluation_history),
        paper_metrics=metrics,
        checkpoint_history=tuple(checkpoint_store.history),
        resumed_from_step=start_step,
        best_historical_checkpoint=best_historical_summary,
    )
    owner._bundle = bundle
    owner.best_historical_checkpoint_ = best_historical_checkpoint
    owner._backend = backend
    final_trace = dict(trace_getter()) if trace_getter is not None else {}
    owner.diagnostics_ = {
        "optimization_steps": result.optimization_steps,
        "target_steps": result.target_steps,
        "accepted_pseudo_labels": result.accepted,
        "unlabeled_predictions": result.unlabeled,
        "acceptance_rate": result.accepted / max(result.unlabeled, 1),
        "training_mode": "fixed_steps",
        "reference_stack": reference,
        "augmentation_profile": config.augmentation_profile,
        "interleave_bn": config.interleave_bn,
        "sampler_contract": sampler_contract,
        "match_contract": match_contract,
        "match_contract_sha256": match_contract_sha256,
        "batch_size_labeled": batch_size,
        "batch_size_unlabeled": batch_size * mu,
        "resumed_from_step": result.resumed_from_step,
        "checkpoint_policy": {
            "checkpoint_interval_steps": checkpoint_interval,
            "evaluation_interval_steps": eval_interval,
            "evaluation_tail_interval_steps": config.evaluation_tail_interval_steps,
            "evaluation_tail_start_fraction": config.evaluation_tail_start_fraction,
            "checkpoint_root_configured": checkpoint_store.enabled,
            "history": list(result.checkpoint_history),
        },
        "evaluation_history": list(result.evaluation_history),
        "paper_metrics": dict(result.paper_metrics),
        "best_historical_checkpoint": best_historical_summary,
        "method_state": final_trace,
        "numeric_probe": numeric_probe,
        "effective_pseudo_label_weight": result.accepted,
        "mean_pseudo_label_weight": result.accepted / max(result.unlabeled, 1),
    }
    owner.diagnostics_.update(final_trace)
    return result


def run_fixed_step_match(
    owner: Any,
    data: Any,
    *,
    device: DeviceSpec,
    seed: int,
    method_id: str,
    step_fn: StepFunction,
    state_getter: StateGetter,
    state_loader: StateLoader,
    trace_getter: TraceGetter | None = None,
    _enforce_reference_contract: bool = True,
) -> MatchTrainingResult:
    """Run one fixed-step Match contract without leaking global state.

    ``_enforce_reference_contract=False`` is an internal testing seam for the
    checkpoint engine. It is accepted only when ``allow_short_run`` is true;
    all method entry points retain strict reference-bundle validation.
    """

    torch = optional_import("torch", extra="inductive-torch")
    with _paper_deterministic_runtime(torch) as runtime_contract:
        result = _run_fixed_step_match(
            owner,
            data,
            device=device,
            seed=seed,
            method_id=method_id,
            step_fn=step_fn,
            state_getter=state_getter,
            state_loader=state_loader,
            trace_getter=trace_getter,
            _enforce_reference_contract=_enforce_reference_contract,
        )
        runtime_contract["rng_initialization"] = (
            "python_numpy_torch_from_model_seed_on_fresh_run;"
            "authenticated_checkpoint_state_on_resume"
        )
        owner.diagnostics_["numeric_runtime_contract"] = runtime_contract
        return result


__all__ = [
    "MATCH_REFERENCE_TARGET_STEPS",
    "MatchStepResult",
    "MatchTrainerConfig",
    "MatchTrainingResult",
    "run_fixed_step_match",
    "uses_fixed_step_match",
]
