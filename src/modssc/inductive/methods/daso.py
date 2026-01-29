from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.deep_utils import (
    concat_data,
    cycle_batch_indices,
    cycle_batches,
    ensure_float_tensor,
    ensure_model_bundle,
    ensure_model_device,
    extract_features,
    extract_logits,
    get_torch_device,
    get_torch_len,
    num_batches,
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


def _flatten_features(features: Any, *, name: str, batch: int | None = None) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    if not isinstance(features, torch.Tensor):
        raise InductiveValidationError(f"{name} must be a torch.Tensor.")
    if int(features.ndim) == 0:
        raise InductiveValidationError(f"{name} must include a batch dimension.")
    if int(features.ndim) == 1:
        features = features.unsqueeze(0)
    elif int(features.ndim) > 2:
        features = features.view(int(features.shape[0]), -1)
    if batch is not None and int(features.shape[0]) != int(batch):
        raise InductiveValidationError(f"{name} batch dimension mismatch.")
    return features


def _cosine_similarity(x: Any, y: Any) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise InductiveValidationError("cosine similarity requires torch.Tensor inputs.")
    x_norm = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    y_norm = y / y.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return x_norm @ y_norm.t()


def _forward_logits_features(bundle: TorchModelBundle, X: Any) -> tuple[Any, Any]:
    torch = optional_import("torch", extra="inductive-torch")
    meta = bundle.meta or {}
    forward_features = None
    forward_head = None
    if isinstance(meta, Mapping):
        forward_features = meta.get("forward_features") or meta.get("feature_extractor")
        forward_head = meta.get("forward_head") or meta.get("head")

    if callable(forward_features) and callable(forward_head):
        feats = forward_features(X)
        if not isinstance(feats, torch.Tensor):
            raise InductiveValidationError("forward_features must return torch.Tensor.")
        logits = extract_logits(forward_head(feats))
        return logits, feats

    out = bundle.model(X)
    feats = None
    logits = None
    if isinstance(out, Mapping):
        if "feat" in out:
            feats = extract_features(out)
        if "logits" in out:
            logits = extract_logits(out)
    else:
        logits = extract_logits(out)
        if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], torch.Tensor):
            feats = out[1]

    if feats is None and callable(forward_features):
        feats = forward_features(X)
        if not isinstance(feats, torch.Tensor):
            raise InductiveValidationError("forward_features must return torch.Tensor.")

    if logits is None:
        if feats is not None and callable(forward_head):
            logits = extract_logits(forward_head(feats))
        else:
            raise InductiveValidationError(
                "Model output must include logits or provide meta['forward_head']."
            )

    if feats is None:
        feats = logits
    return logits, feats


def _forward_features(
    bundle: TorchModelBundle, X: Any, *, model_override: Any | None = None
) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    model = model_override or bundle.model
    meta = bundle.meta or {}
    if model_override is None and isinstance(meta, Mapping):
        forward = meta.get("forward_features") or meta.get("feature_extractor")
        if callable(forward):
            feats = forward(X)
            if not isinstance(feats, torch.Tensor):
                raise InductiveValidationError("forward_features must return torch.Tensor.")
            return feats
    if model_override is not None and isinstance(meta, Mapping):
        forward = meta.get("forward_features_ema") or meta.get("feature_extractor_ema")
        if callable(forward):
            feats = forward(X)
            if not isinstance(feats, torch.Tensor):
                raise InductiveValidationError("forward_features_ema must return torch.Tensor.")
            return feats

    out = model(X)
    if isinstance(out, Mapping) and "feat" in out:
        return extract_features(out)
    if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], torch.Tensor):
        return out[1]
    try:
        return extract_logits(out)
    except InductiveValidationError as exc:
        raise InductiveValidationError(
            "DASO requires feature representations via output['feat'], a (logits, feat) "
            "tuple, or meta['forward_features']."
        ) from exc


def _check_ema(student: Any, teacher: Any) -> None:
    if teacher is student:
        raise InductiveValidationError("ema_model must be distinct from model.")
    s_params = list(student.parameters())
    t_params = list(teacher.parameters())
    if len(s_params) != len(t_params):
        raise InductiveValidationError("ema_model must match model parameter count.")
    for sp, tp in zip(s_params, t_params, strict=True):
        if sp.shape != tp.shape:
            raise InductiveValidationError("ema_model parameter shapes must match model.")
        if sp.device != tp.device:
            raise InductiveValidationError("ema_model must be on the same device as model.")


def _init_ema(student: Any, teacher: Any) -> None:
    try:
        teacher.load_state_dict(student.state_dict(), strict=True)
    except Exception as exc:  # pragma: no cover - defensive
        raise InductiveValidationError(
            "ema_model must be initialized with the same architecture as model."
        ) from exc


def _update_ema(student: Any, teacher: Any, *, decay: float) -> None:
    for t_param, s_param in zip(teacher.parameters(), student.parameters(), strict=True):
        t_param.data.mul_(float(decay)).add_(s_param.data, alpha=1.0 - float(decay))


@dataclass(frozen=True)
class DASOSpec:
    """Specification for DASO (torch-only)."""

    model_bundle: TorchModelBundle | None = None
    lambda_u: float = 1.0
    lambda_align: float = 1.0
    p_cutoff: float = 0.95
    t_proto: float = 0.05
    t_dist: float = 1.5
    queue_size: int = 256
    pretrain_steps: int = 5000
    dist_update_period: int = 100
    ema_decay: float = 0.999
    use_ema: bool = True
    dist_aware: bool = True
    interp_alpha: float = 0.5
    hard_label: bool = True
    use_cat: bool = False
    batch_size: int = 64
    max_epochs: int = 1
    detach_target: bool = True


class DASOMethod(InductiveMethod):
    """DASO with distribution-aware blending and semantic alignment (torch-only)."""

    info = MethodInfo(
        method_id="daso",
        name="DASO",
        year=2021,
        family="pseudo-label",
        supports_gpu=True,
        paper_title="DASO: Distribution-Aware Semantics-Oriented Pseudo-label for Imbalanced Semi-Supervised Learning",
        paper_pdf="https://arxiv.org/abs/2106.10909",
        official_code="https://github.com/ytaek-oh/daso",
    )

    def __init__(self, spec: DASOSpec | None = None) -> None:
        self.spec = spec or DASOSpec()
        self._bundle: TorchModelBundle | None = None
        self._backend: str | None = None
        self._ema_model: Any | None = None

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> DASOMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "params lambda_u=%s lambda_align=%s p_cutoff=%s t_proto=%s t_dist=%s queue_size=%s "
            "pretrain_steps=%s dist_update_period=%s ema_decay=%s use_ema=%s dist_aware=%s "
            "interp_alpha=%s hard_label=%s use_cat=%s batch_size=%s max_epochs=%s "
            "detach_target=%s has_model_bundle=%s device=%s seed=%s",
            self.spec.lambda_u,
            self.spec.lambda_align,
            self.spec.p_cutoff,
            self.spec.t_proto,
            self.spec.t_dist,
            self.spec.queue_size,
            self.spec.pretrain_steps,
            self.spec.dist_update_period,
            self.spec.ema_decay,
            self.spec.use_ema,
            self.spec.dist_aware,
            self.spec.interp_alpha,
            self.spec.hard_label,
            self.spec.use_cat,
            self.spec.batch_size,
            self.spec.max_epochs,
            self.spec.detach_target,
            bool(self.spec.model_bundle),
            device,
            seed,
        )
        if data is None:
            raise InductiveValidationError("data must not be None.")

        backend = detect_backend(data.X_l)
        if backend != "torch":
            raise InductiveValidationError("DASO requires torch tensors (torch backend).")

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u_w is None or ds.X_u_s is None:
            raise InductiveValidationError("DASO requires X_u_w and X_u_s.")

        X_l = ds.X_l
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        X_u_w = ds.X_u_w
        X_u_s = ds.X_u_s
        logger.info(
            "DASO sizes: n_labeled=%s n_unlabeled=%s",
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
            raise InductiveValidationError("model_bundle must be provided for DASO.")
        bundle = ensure_model_bundle(self.spec.model_bundle)
        model = bundle.model
        optimizer = bundle.optimizer
        ensure_model_device(model, device=get_torch_device(X_l))

        if int(self.spec.batch_size) <= 0:
            raise InductiveValidationError("batch_size must be >= 1.")
        if int(self.spec.max_epochs) <= 0:
            raise InductiveValidationError("max_epochs must be >= 1.")
        if float(self.spec.lambda_u) < 0:
            raise InductiveValidationError("lambda_u must be >= 0.")
        if float(self.spec.lambda_align) < 0:
            raise InductiveValidationError("lambda_align must be >= 0.")
        if not (0.0 <= float(self.spec.p_cutoff) <= 1.0):
            raise InductiveValidationError("p_cutoff must be in [0, 1].")
        if float(self.spec.t_proto) <= 0:
            raise InductiveValidationError("t_proto must be > 0.")
        if float(self.spec.t_dist) <= 0:
            raise InductiveValidationError("t_dist must be > 0.")
        if int(self.spec.queue_size) <= 0:
            raise InductiveValidationError("queue_size must be >= 1.")
        if int(self.spec.pretrain_steps) < 0:
            raise InductiveValidationError("pretrain_steps must be >= 0.")
        if int(self.spec.dist_update_period) <= 0:
            raise InductiveValidationError("dist_update_period must be >= 1.")
        if not (0.0 <= float(self.spec.ema_decay) <= 1.0):
            raise InductiveValidationError("ema_decay must be in [0, 1].")
        if not (0.0 <= float(self.spec.interp_alpha) <= 1.0):
            raise InductiveValidationError("interp_alpha must be in [0, 1].")

        ema_model = None
        if bool(self.spec.use_ema):
            ema_model = bundle.ema_model or copy.deepcopy(model)
            _check_ema(model, ema_model)
            _init_ema(model, ema_model)
            for p in ema_model.parameters():
                p.requires_grad_(False)
            ema_model.eval()
        self._ema_model = ema_model

        steps_l = num_batches(int(get_torch_len(X_l)), int(self.spec.batch_size))
        steps_u = num_batches(int(get_torch_len(X_u_w)), int(self.spec.batch_size))
        steps_per_epoch = max(int(steps_l), int(steps_u))

        gen_l = torch.Generator().manual_seed(int(seed))
        gen_u = torch.Generator().manual_seed(int(seed) + 1)

        queue: list[Any] | None = None
        m_hat: Any | None = None
        dist_accum: Any | None = None

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

                if bool(self.spec.use_cat):
                    inputs = concat_data([x_lb, x_uw, x_us])
                    logits_all, feats_all = _forward_logits_features(bundle, inputs)
                    if int(logits_all.ndim) != 2:
                        raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                    num_lb = int(get_torch_len(x_lb))
                    num_u = int(get_torch_len(x_uw))
                    expected = num_lb + num_u + int(get_torch_len(x_us))
                    if int(logits_all.shape[0]) != expected:
                        raise InductiveValidationError(
                            "Concatenated logits batch size does not match inputs."
                        )
                    feats_all = _flatten_features(feats_all, name="feat_all", batch=expected)
                    logits_l = logits_all[:num_lb]
                    logits_uw = logits_all[num_lb : num_lb + num_u]
                    logits_us = logits_all[num_lb + num_u :]
                    feats_uw = feats_all[num_lb : num_lb + num_u]
                    feats_us = feats_all[num_lb + num_u :]
                else:
                    logits_l, _ = _forward_logits_features(bundle, x_lb)
                    if bool(self.spec.detach_target):
                        with torch.no_grad():
                            logits_uw, feats_uw = _forward_logits_features(bundle, x_uw)
                    else:
                        logits_uw, feats_uw = _forward_logits_features(bundle, x_uw)
                    logits_us, feats_us = _forward_logits_features(bundle, x_us)
                    feats_uw = _flatten_features(
                        feats_uw, name="feat_uw", batch=int(get_torch_len(x_uw))
                    )
                    feats_us = _flatten_features(
                        feats_us, name="feat_us", batch=int(get_torch_len(x_us))
                    )

                if int(logits_l.ndim) != 2 or int(logits_uw.ndim) != 2 or int(logits_us.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                if logits_uw.shape != logits_us.shape:
                    raise InductiveValidationError("Unlabeled logits shape mismatch.")
                if logits_uw.shape[1] != logits_l.shape[1]:
                    raise InductiveValidationError("Logits must agree on class dimension.")
                if feats_uw.shape != feats_us.shape:
                    raise InductiveValidationError("Unlabeled feature shape mismatch.")

                n_classes = int(logits_l.shape[1])
                if y_lb.min().item() < 0 or y_lb.max().item() >= n_classes:
                    raise InductiveValidationError("y_l labels must be within [0, n_classes).")

                feat_dim = int(feats_uw.shape[1])
                if queue is None:
                    queue = [
                        torch.empty((0, feat_dim), device=feats_uw.device, dtype=feats_uw.dtype)
                        for _ in range(n_classes)
                    ]
                if m_hat is None:
                    m_hat = torch.full(
                        (int(n_classes),),
                        1.0 / float(n_classes),
                        device=feats_uw.device,
                        dtype=feats_uw.dtype,
                    )
                    dist_accum = torch.zeros_like(m_hat)

                with torch.no_grad():
                    feat_lb = _forward_features(bundle, x_lb, model_override=ema_model)
                    feat_lb = _flatten_features(
                        feat_lb, name="feat_lb", batch=int(get_torch_len(x_lb))
                    ).detach()
                    if int(feat_lb.shape[1]) != feat_dim:
                        raise InductiveValidationError("Prototype feature dimension mismatch.")

                for k in range(n_classes):
                    mask = y_lb == k
                    if int(mask.sum()) == 0:
                        continue
                    cls_feats = feat_lb[mask]
                    queue_k = queue[k]
                    queue_k = torch.cat([queue_k, cls_feats], dim=0)
                    if int(queue_k.shape[0]) > int(self.spec.queue_size):
                        queue_k = queue_k[-int(self.spec.queue_size) :]
                    queue[k] = queue_k

                prototypes: list[Any] = []
                for k in range(n_classes):
                    queue_k = queue[k]
                    if int(queue_k.numel()) == 0:
                        proto = feats_uw.new_zeros((feat_dim,))
                    else:
                        proto = queue_k.mean(dim=0)
                    prototypes.append(proto)
                C = torch.stack(prototypes, dim=0)

                sup_loss = torch.nn.functional.cross_entropy(logits_l, y_lb)

                p_hat = torch.softmax(logits_uw, dim=1)
                q_hat = torch.softmax(
                    _cosine_similarity(feats_uw, C) / float(self.spec.t_proto), dim=1
                )

                if bool(self.spec.detach_target):
                    p_hat = p_hat.detach()
                    q_hat = q_hat.detach()

                pretraining = step_idx < int(self.spec.pretrain_steps)
                if pretraining:
                    p_hat_prime = p_hat
                else:
                    if bool(self.spec.dist_aware):
                        v = m_hat.clamp_min(1e-12).pow(1.0 / float(self.spec.t_dist))
                        v = v / v.max().clamp_min(1e-12)
                        k_prime = p_hat.argmax(dim=1)
                        v_k = v[k_prime].unsqueeze(1)
                        p_hat_prime = (1.0 - v_k) * p_hat + v_k * q_hat
                    else:
                        alpha = float(self.spec.interp_alpha)
                        p_hat_prime = (1.0 - alpha) * p_hat + alpha * q_hat

                mask = (p_hat_prime.max(dim=1).values >= float(self.spec.p_cutoff)).to(
                    logits_us.dtype
                )

                if bool(self.spec.hard_label):
                    pseudo = p_hat_prime.argmax(dim=1)
                    loss_u = torch.nn.functional.cross_entropy(logits_us, pseudo, reduction="none")
                else:
                    log_probs = torch.nn.functional.log_softmax(logits_us, dim=1)
                    loss_u = -(p_hat_prime * log_probs).sum(dim=1)

                if int(mask.numel()) == 0:
                    unsup_loss = torch.zeros((), device=logits_us.device)
                else:
                    unsup_loss = (loss_u * mask).sum() / mask.sum().clamp_min(1.0)

                if pretraining or float(self.spec.lambda_align) == 0.0:
                    align_loss = torch.zeros((), device=logits_us.device)
                else:
                    q_s = torch.softmax(
                        _cosine_similarity(feats_us, C) / float(self.spec.t_proto), dim=1
                    )
                    log_q_s = torch.log(q_s.clamp_min(1e-12))
                    align_loss = -(q_hat * log_q_s).sum(dim=1).mean()

                loss = (
                    sup_loss
                    + float(self.spec.lambda_u) * unsup_loss
                    + float(self.spec.lambda_align) * align_loss
                )

                if step == 0:
                    mask_mean = float(mask.mean().item()) if int(mask.numel()) else 0.0
                    logger.debug(
                        "DASO epoch=%s pretrain=%s mask_mean=%.3f sup_loss=%.4f "
                        "unsup_loss=%.4f align_loss=%.4f",
                        epoch,
                        pretraining,
                        mask_mean,
                        float(sup_loss.item()),
                        float(unsup_loss.item()),
                        float(align_loss.item()),
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if ema_model is not None:
                    _update_ema(model, ema_model, decay=float(self.spec.ema_decay))

                if dist_accum is not None:
                    preds = p_hat_prime.argmax(dim=1)
                    dist_accum += torch.bincount(preds, minlength=int(n_classes)).to(
                        dist_accum.dtype
                    )
                    if (step_idx + 1) % int(self.spec.dist_update_period) == 0:
                        total = dist_accum.sum()
                        if float(total) > 0:
                            m_hat = dist_accum / total
                        dist_accum.zero_()

                step_idx += 1

        self._bundle = bundle
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, X: Any) -> Any:
        if self._bundle is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if backend != "torch":
            raise InductiveValidationError("predict_proba requires torch tensors.")
        torch = optional_import("torch", extra="inductive-torch")

        # Support Dict or Tensor
        if not isinstance(X, torch.Tensor) and not isinstance(X, dict):
            raise InductiveValidationError("predict_proba requires torch.Tensor or dict inputs.")

        model = self._bundle.model
        was_training = model.training
        model.eval()

        # Batched inference
        batch_size = int(self.spec.batch_size)
        from .deep_utils import extract_logits, slice_data

        n_samples = int(X["x"].shape[0]) if isinstance(X, dict) else int(X.shape[0])

        all_logits = []
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                if isinstance(X, dict):
                    idx = torch.arange(start, end, device=X["x"].device)
                    batch_X = slice_data(X, idx)
                else:
                    batch_X = X[start:end]

                logits_batch = extract_logits(model(batch_X))
                if int(logits_batch.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                all_logits.append(logits_batch)

            if not all_logits:
                # Handle empty case
                logits = torch.empty(
                    (0, 0), device=X["x"].device if isinstance(X, dict) else X.device
                )
            else:
                logits = torch.cat(all_logits, dim=0)

            probs = torch.softmax(logits, dim=1)

        if was_training:
            model.train()
        return probs

    def predict(self, X: Any) -> Any:
        proba = self.predict_proba(X)
        return proba.argmax(dim=1)
