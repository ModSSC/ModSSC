from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.deep_utils import (
    TorchBundlePredictMixin,
    concat_data,
    cycle_batch_indices,
    cycle_batches,
    ensure_float_tensor,
    ensure_model_bundle,
    ensure_model_device,
    extract_features,
    extract_logits,
    freeze_batchnorm,
    get_torch_device,
    get_torch_len,
    num_batches,
    sharpen_probs,
    should_freeze_batchnorm,
    slice_data,
)
from modssc.inductive.methods.utils import (
    detect_backend,
    ensure_1d_labels_torch,
    ensure_torch_data,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec

logger = logging.getLogger(__name__)


_sharpen = sharpen_probs


def _mixup(
    X: Any,
    y: Any,
    *,
    alpha: float,
    generator: Any,
):
    torch = optional_import("torch", extra="inductive-torch")
    if not isinstance(y, torch.Tensor):
        raise InductiveValidationError("mixup expects torch.Tensor labels.")
    is_dict = isinstance(X, dict)
    base = X["x"] if is_dict and "x" in X else X
    if not isinstance(base, torch.Tensor):
        raise InductiveValidationError("mixup expects torch.Tensor inputs.")
    if int(base.shape[0]) != int(y.shape[0]):
        raise InductiveValidationError("mixup X and y must have the same first dimension.")
    batch = int(base.shape[0])
    if batch == 0:
        raise InductiveValidationError("mixup requires non-empty batch.")
    if alpha <= 0:
        lam = torch.ones((batch,), device=base.device, dtype=base.dtype)
    else:
        dist = torch.distributions.Beta(float(alpha), float(alpha))
        lam = dist.sample((batch,)).to(device=base.device, dtype=base.dtype)
        lam = torch.max(lam, 1.0 - lam)

    perm = torch.randperm(batch, generator=generator, device="cpu")
    if base.device.type != "cpu":
        perm = perm.to(device=base.device)
    X2 = base[perm]
    y2 = y[perm]

    view = [batch] + [1] * (int(base.dim()) - 1)
    lam_x = lam.view(*view)
    mixed_x = lam_x * base + (1.0 - lam_x) * X2
    mixed_y = lam.view(batch, 1) * y + (1.0 - lam.view(batch, 1)) * y2
    if is_dict:
        out = dict(X)
        out["x"] = mixed_x
        return out, mixed_y
    return mixed_x, mixed_y


def _forward_head(bundle: TorchModelBundle, *, features: Any) -> Any:
    meta = bundle.meta or {}
    head = None
    if isinstance(meta, Mapping):
        head = meta.get("forward_head") or meta.get("head")
    if callable(head):
        return head(features)
    model = bundle.model
    try:
        return model(features, only_fc=True)
    except TypeError as exc:
        raise InductiveValidationError(
            "mixup_manifold requires bundle.meta['forward_head'] (callable) or "
            "a model that accepts only_fc=True."
        ) from exc


def _bundle_prefers_manifold_mixup(bundle: TorchModelBundle) -> bool:
    meta = bundle.meta or {}
    if not isinstance(meta, Mapping):
        return False
    return bool(meta.get("prefer_manifold_mixup"))


def _bundle_supports_discrete_inputs(bundle: TorchModelBundle) -> bool:
    meta = bundle.meta or {}
    if isinstance(meta, Mapping) and meta.get("input_space") == "token_ids":
        return True
    model = bundle.model
    get_emb_fn = getattr(model, "get_input_embeddings", None)
    if callable(get_emb_fn):
        return True
    if hasattr(model, "module"):
        return callable(getattr(model.module, "get_input_embeddings", None))
    return False


def _ensure_mixmatch_tensor_input(
    x: Any,
    *,
    name: str,
    allow_discrete: bool,
) -> None:
    torch = optional_import("torch", extra="inductive-torch")
    if allow_discrete:
        if isinstance(x, dict):
            ensure_float_tensor(x, name=name)
            return
        if not isinstance(x, torch.Tensor):
            raise InductiveValidationError(f"{name} must be a torch.Tensor.")
        valid = {
            torch.float32,
            torch.float64,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if x.dtype not in valid:
            raise InductiveValidationError(
                f"{name} must be floating point or integer token ids for embedding-based models."
            )
        return
    ensure_float_tensor(x, name=name)


def _forward_features(bundle: TorchModelBundle, X: Any) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    meta = bundle.meta or {}
    if isinstance(meta, Mapping):
        forward = meta.get("forward_features") or meta.get("feature_extractor")
        if callable(forward):
            features = forward(X)
            if not isinstance(features, torch.Tensor):
                raise InductiveValidationError(
                    "mixup_manifold forward_features must return torch.Tensor."
                )
            return features
    out = bundle.model(X)
    if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], torch.Tensor):
        return out[1]
    return extract_features(out)


@dataclass(frozen=True)
class MixMatchSpec:
    """Specification for MixMatch (torch-only)."""

    model_bundle: TorchModelBundle | None = None
    lambda_u: float = 1.0
    temperature: float = 0.5
    mixup_alpha: float = 0.5
    unsup_warm_up: float = 0.4
    mixup_manifold: bool = False
    freeze_bn: bool = False
    batch_size: int = 64
    max_epochs: int = 1


class MixMatchMethod(TorchBundlePredictMixin, InductiveMethod):
    """MixMatch consistency with MixUp (torch-only, uses two augmentations)."""

    info = MethodInfo(
        method_id="mixmatch",
        name="MixMatch",
        year=2019,
        family="mixup",
        supports_gpu=True,
        paper_title="MixMatch: A Holistic Approach to Semi-Supervised Learning",
        paper_pdf="https://arxiv.org/abs/1905.02249",
        official_code="",
    )

    def __init__(self, spec: MixMatchSpec | None = None) -> None:
        self.spec = spec or MixMatchSpec()
        self._bundle: TorchModelBundle | None = None
        self._backend: str | None = None

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> MixMatchMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "params lambda_u=%s temperature=%s mixup_alpha=%s unsup_warm_up=%s mixup_manifold=%s "
            "freeze_bn=%s batch_size=%s max_epochs=%s has_model_bundle=%s device=%s seed=%s",
            self.spec.lambda_u,
            self.spec.temperature,
            self.spec.mixup_alpha,
            self.spec.unsup_warm_up,
            self.spec.mixup_manifold,
            self.spec.freeze_bn,
            self.spec.batch_size,
            self.spec.max_epochs,
            bool(self.spec.model_bundle),
            device,
            seed,
        )
        if data is None:
            raise InductiveValidationError("data must not be None.")

        backend = detect_backend(data.X_l)
        if backend != "torch":
            raise InductiveValidationError("MixMatch requires torch tensors (torch backend).")

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u_w is None or ds.X_u_s is None:
            raise InductiveValidationError("MixMatch requires X_u_w and X_u_s.")

        X_l = ds.X_l
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        X_u_w = ds.X_u_w
        X_u_s = ds.X_u_s
        logger.info(
            "MixMatch sizes: n_labeled=%s n_unlabeled=%s",
            int(get_torch_len(X_l)),
            int(get_torch_len(X_u_w)),
        )

        if int(get_torch_len(X_l)) == 0:
            raise InductiveValidationError("X_l must be non-empty.")
        if int(get_torch_len(X_u_w)) == 0 or int(get_torch_len(X_u_s)) == 0:
            raise InductiveValidationError("X_u_w and X_u_s must be non-empty.")
        if int(get_torch_len(X_u_w)) != int(get_torch_len(X_u_s)):
            raise InductiveValidationError("X_u_w and X_u_s must have the same number of rows.")

        if y_l.dtype != torch.int64:
            raise InductiveValidationError("y_l must be int64 for torch cross entropy.")

        if self.spec.model_bundle is None:
            raise InductiveValidationError("model_bundle must be provided for MixMatch.")
        bundle = ensure_model_bundle(self.spec.model_bundle)
        model = bundle.model
        optimizer = bundle.optimizer
        ensure_model_device(model, device=get_torch_device(X_l))
        use_manifold_mixup = bool(self.spec.mixup_manifold) or _bundle_prefers_manifold_mixup(
            bundle
        )
        allow_discrete_inputs = use_manifold_mixup and _bundle_supports_discrete_inputs(bundle)

        if int(self.spec.batch_size) <= 0:
            raise InductiveValidationError("batch_size must be >= 1.")
        if int(self.spec.max_epochs) <= 0:
            raise InductiveValidationError("max_epochs must be >= 1.")
        if float(self.spec.lambda_u) < 0:
            raise InductiveValidationError("lambda_u must be >= 0.")
        if float(self.spec.temperature) <= 0:
            raise InductiveValidationError("temperature must be > 0.")
        if float(self.spec.mixup_alpha) < 0:
            raise InductiveValidationError("mixup_alpha must be >= 0.")
        if float(self.spec.unsup_warm_up) < 0:
            raise InductiveValidationError("unsup_warm_up must be >= 0.")

        _ensure_mixmatch_tensor_input(X_l, name="X_l", allow_discrete=allow_discrete_inputs)
        _ensure_mixmatch_tensor_input(X_u_w, name="X_u_w", allow_discrete=allow_discrete_inputs)
        _ensure_mixmatch_tensor_input(X_u_s, name="X_u_s", allow_discrete=allow_discrete_inputs)

        steps_l = num_batches(int(get_torch_len(X_l)), int(self.spec.batch_size))
        steps_u = num_batches(int(get_torch_len(X_u_w)), int(self.spec.batch_size))
        steps_per_epoch = max(int(steps_l), int(steps_u))
        total_steps = int(self.spec.max_epochs) * steps_per_epoch
        if float(self.spec.unsup_warm_up) <= 0:
            warmup_steps = 0
        else:
            warmup_steps = int(max(1, round(float(self.spec.unsup_warm_up) * total_steps)))

        gen_l = torch.Generator().manual_seed(int(seed))
        gen_u = torch.Generator().manual_seed(int(seed) + 1)

        step_idx = 0
        model.train()
        for epoch in range(int(self.spec.max_epochs)):
            iter_l = cycle_batches(
                X_l,
                y_l,
                batch_size=int(self.spec.batch_size),
                generator=gen_l,
                steps=steps_per_epoch,
            )
            iter_u_idx = cycle_batch_indices(
                int(get_torch_len(X_u_w)),
                batch_size=int(self.spec.batch_size),
                generator=gen_u,
                device=get_torch_device(X_u_w),
                steps=steps_per_epoch,
            )
            for step, ((x_lb, y_lb), idx_u) in enumerate(zip(iter_l, iter_u_idx, strict=False)):
                x_uw = slice_data(X_u_w, idx_u)
                x_us = slice_data(X_u_s, idx_u)

                freeze_bn = should_freeze_batchnorm(
                    x_lb,
                    x_uw,
                    x_us,
                    enabled=bool(self.spec.freeze_bn),
                )
                with torch.no_grad(), freeze_batchnorm(model, enabled=freeze_bn):
                    logits_uw = extract_logits(model(x_uw))
                    logits_us = extract_logits(model(x_us))

                if logits_uw.shape != logits_us.shape:
                    raise InductiveValidationError("Unlabeled logits shape mismatch.")
                if int(logits_uw.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")

                probs_u = (torch.softmax(logits_uw, dim=1) + torch.softmax(logits_us, dim=1)) / 2.0
                pseudo_u = _sharpen(probs_u, temperature=float(self.spec.temperature)).detach()

                n_classes = int(pseudo_u.shape[1])
                if y_lb.min().item() < 0 or y_lb.max().item() >= n_classes:
                    raise InductiveValidationError("y_l labels must be within [0, n_classes).")
                y_lb_onehot = torch.nn.functional.one_hot(y_lb, num_classes=n_classes).to(
                    pseudo_u.dtype
                )
                targets = torch.cat([y_lb_onehot, pseudo_u, pseudo_u], dim=0)

                if use_manifold_mixup:
                    with freeze_batchnorm(model, enabled=freeze_bn):
                        feat_lb = _forward_features(bundle, x_lb)
                        feat_uw = _forward_features(bundle, x_uw)
                        feat_us = _forward_features(bundle, x_us)
                    inputs = torch.cat([feat_lb, feat_uw, feat_us], dim=0)
                    mixed_x, mixed_y = _mixup(
                        inputs, targets, alpha=float(self.spec.mixup_alpha), generator=gen_l
                    )
                    logits_all = extract_logits(_forward_head(bundle, features=mixed_x))
                else:
                    inputs = concat_data([x_lb, x_uw, x_us])
                    mixed_x, mixed_y = _mixup(
                        inputs, targets, alpha=float(self.spec.mixup_alpha), generator=gen_l
                    )
                    logits_all = extract_logits(model(mixed_x))

                if int(logits_all.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")

                num_lb = int(get_torch_len(x_lb))
                logits_l = logits_all[:num_lb]
                logits_u = logits_all[num_lb:]

                log_probs_l = torch.nn.functional.log_softmax(logits_l, dim=1)
                sup_loss = -(mixed_y[:num_lb] * log_probs_l).sum(dim=1).mean()

                probs_u_mixed = torch.softmax(logits_u, dim=1)
                unsup_loss = ((probs_u_mixed - mixed_y[num_lb:]) ** 2).mean()

                warm = 1.0 if warmup_steps <= 0 else min(float(step_idx) / float(warmup_steps), 1.0)
                loss = sup_loss + float(self.spec.lambda_u) * unsup_loss * float(warm)

                if step == 0:
                    logger.debug(
                        "MixMatch epoch=%s warm=%.3f sup_loss=%.4f unsup_loss=%.4f",
                        epoch,
                        float(warm),
                        float(sup_loss.item()),
                        float(unsup_loss.item()),
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_idx += 1

        self._bundle = bundle
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self
