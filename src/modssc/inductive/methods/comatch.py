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
    concat_data,
    cycle_batch_indices,
    cycle_batches,
    ensure_float_tensor,
    ensure_model_bundle,
    ensure_model_device,
    num_batches,
)
from modssc.inductive.methods.utils import (
    detect_backend,
    ensure_1d_labels_torch,
    ensure_torch_data,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec

logger = logging.getLogger(__name__)


def _extract_logits_and_features(output: Any) -> tuple[Any, Any]:
    torch = optional_import("torch", extra="inductive-torch")
    if isinstance(output, Mapping):
        logits = output.get("logits")
        feat = output.get("feat")
        if logits is None or feat is None:
            raise InductiveValidationError(
                "CoMatch requires model output mapping with keys 'logits' and 'feat'."
            )
        if not isinstance(logits, torch.Tensor) or not isinstance(feat, torch.Tensor):
            raise InductiveValidationError("CoMatch logits/feat must be torch.Tensor outputs.")
        return logits, feat
    if isinstance(output, tuple) and len(output) >= 2:
        logits, feat = output[0], output[1]
        if not isinstance(logits, torch.Tensor) or not isinstance(feat, torch.Tensor):
            raise InductiveValidationError("CoMatch logits/feat must be torch.Tensor outputs.")
        return logits, feat
    if isinstance(output, torch.Tensor):
        raise InductiveValidationError(
            "CoMatch requires model outputs to include features ('feat')."
        )
    raise InductiveValidationError("CoMatch model output must be a mapping or tuple.")


def _compute_prob(logits: Any, *, temperature: float) -> Any:
    if temperature <= 0:
        raise InductiveValidationError("temperature must be > 0.")
    torch = optional_import("torch", extra="inductive-torch")
    return torch.softmax(logits / float(temperature), dim=1)


def _l2_normalize(feats: Any, *, eps: float = 1e-12) -> Any:
    optional_import("torch", extra="inductive-torch")
    denom = feats.norm(p=2, dim=1, keepdim=True).clamp_min(float(eps))
    return feats / denom


def _row_normalize(mat: Any) -> Any:
    denom = mat.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return mat / denom


def _build_pseudo_graph(probs: Any, *, threshold: float) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    q = probs @ probs.t()
    eye = torch.eye(int(q.shape[0]), device=q.device, dtype=q.dtype)
    q = q * (1.0 - eye) + eye
    if threshold > 0:
        mask = (q >= float(threshold)).to(q.dtype)
        q = q * mask
    return _row_normalize(q)


def _contrastive_loss(feats_s0: Any, feats_s1: Any, q: Any, *, temperature: float) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    sim = torch.exp((feats_s0 @ feats_s1.t()) / float(temperature))
    sim = _row_normalize(sim)
    q = q.to(sim.dtype)
    return -(torch.log(sim + 1e-7) * q).sum(dim=1).mean()


def _one_hot(labels: Any, *, n_classes: int) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    return torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=int(n_classes)).to(
        dtype=torch.float32
    )


@dataclass(frozen=True)
class CoMatchSpec:
    """Specification for CoMatch (torch-only)."""

    model_bundle: TorchModelBundle | None = None
    lambda_u: float = 1.0
    lambda_c: float = 1.0
    p_cutoff: float = 0.95
    temperature: float = 0.5
    contrast_p_cutoff: float = 0.8
    smoothing_alpha: float = 0.9
    queue_size: int = 2560
    min_queue_fill: int = 0
    da_len: int = 256
    dist_align: bool = True
    dist_uniform: bool = True
    hard_label: bool = True
    use_cat: bool = False
    batch_size: int = 64
    max_epochs: int = 1
    detach_target: bool = True


class CoMatchMethod(InductiveMethod):
    """CoMatch with memory-smoothed pseudo-labels and contrastive graph loss (torch-only)."""

    info = MethodInfo(
        method_id="comatch",
        name="CoMatch",
        year=2021,
        family="pseudo-label",
        supports_gpu=True,
        paper_title="CoMatch: Semi-supervised Learning with Contrastive Graph Regularization",
        paper_pdf="https://arxiv.org/abs/2011.11183",
        official_code="https://github.com/salesforce/CoMatch/",
    )

    def __init__(self, spec: CoMatchSpec | None = None) -> None:
        self.spec = spec or CoMatchSpec()
        self._bundle: TorchModelBundle | None = None
        self._backend: str | None = None
        self._queue_feats: Any | None = None
        self._queue_probs: Any | None = None
        self._queue_ptr: int = 0
        self._queue_count: int = 0
        self._da_queue: Any | None = None
        self._da_ptr: int = 0
        self._da_count: int = 0
        self._da_target: Any | None = None

    def _init_memory(self, *, feat_dim: int, n_classes: int, device: Any) -> None:
        if int(self.spec.queue_size) <= 0:
            return
        torch = optional_import("torch", extra="inductive-torch")
        self._queue_feats = torch.zeros((int(self.spec.queue_size), int(feat_dim)), device=device)
        self._queue_probs = torch.zeros((int(self.spec.queue_size), int(n_classes)), device=device)
        self._queue_ptr = 0
        self._queue_count = 0

    def _update_memory(self, feats: Any, probs: Any) -> None:
        if self._queue_feats is None or self._queue_probs is None:
            return
        n = int(feats.shape[0])
        if n == 0:
            return
        if n > int(self._queue_feats.shape[0]):
            feats = feats[: int(self._queue_feats.shape[0])]
            probs = probs[: int(self._queue_probs.shape[0])]
            n = int(feats.shape[0])
        size = int(self._queue_feats.shape[0])
        end = self._queue_ptr + n
        if end <= size:
            self._queue_feats[self._queue_ptr : end] = feats
            self._queue_probs[self._queue_ptr : end] = probs
        else:
            first = size - self._queue_ptr
            self._queue_feats[self._queue_ptr :] = feats[:first]
            self._queue_probs[self._queue_ptr :] = probs[:first]
            remaining = n - first
            self._queue_feats[:remaining] = feats[first:]
            self._queue_probs[:remaining] = probs[first:]
        self._queue_ptr = (self._queue_ptr + n) % size
        self._queue_count = min(size, self._queue_count + n)

    def _memory_smooth(self, probs: Any, feats: Any) -> Any:
        if self._queue_feats is None or self._queue_probs is None:
            return probs
        if self._queue_count == 0:
            return probs
        if int(self.spec.min_queue_fill) > 0 and self._queue_count < int(self.spec.min_queue_fill):
            return probs
        alpha = float(self.spec.smoothing_alpha)
        if alpha >= 1.0:
            return probs
        queue_feats = self._queue_feats[: self._queue_count]
        queue_probs = self._queue_probs[: self._queue_count]
        sim = (feats @ queue_feats.t()) / float(self.spec.temperature)
        affinity = _row_normalize(sim.exp())
        smoothed = alpha * probs + (1.0 - alpha) * (affinity @ queue_probs)
        return smoothed

    def _init_da(self, *, n_classes: int, device: Any) -> None:
        if int(self.spec.da_len) <= 0:
            return
        torch = optional_import("torch", extra="inductive-torch")
        self._da_queue = torch.zeros((int(self.spec.da_len), int(n_classes)), device=device)
        self._da_ptr = 0
        self._da_count = 0
        self._da_target = torch.full((int(n_classes),), 1.0 / float(n_classes), device=device)

    def _dist_align(self, probs_u: Any, *, probs_l: Any | None) -> Any:
        if not bool(self.spec.dist_align) or int(self.spec.da_len) <= 0:
            return probs_u
        if self._da_queue is None:
            self._init_da(n_classes=int(probs_u.shape[1]), device=probs_u.device)
        assert self._da_queue is not None
        mean_u = probs_u.mean(dim=0)
        self._da_queue[self._da_ptr] = mean_u
        self._da_ptr = (self._da_ptr + 1) % int(self.spec.da_len)
        self._da_count = min(int(self.spec.da_len), self._da_count + 1)
        p_model = self._da_queue[: self._da_count].mean(dim=0)
        if bool(self.spec.dist_uniform):
            target = self._da_target
        else:
            if probs_l is None:
                raise InductiveValidationError("dist_uniform=False requires labeled probabilities.")
            target = probs_l.mean(dim=0)
        ratio = target / p_model.clamp_min(1e-6)
        aligned = probs_u * ratio
        aligned = aligned / aligned.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return aligned

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> CoMatchMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "params lambda_u=%s lambda_c=%s p_cutoff=%s temperature=%s contrast_p_cutoff=%s "
            "smoothing_alpha=%s queue_size=%s da_len=%s dist_align=%s dist_uniform=%s "
            "hard_label=%s use_cat=%s batch_size=%s max_epochs=%s detach_target=%s "
            "has_model_bundle=%s device=%s seed=%s",
            self.spec.lambda_u,
            self.spec.lambda_c,
            self.spec.p_cutoff,
            self.spec.temperature,
            self.spec.contrast_p_cutoff,
            self.spec.smoothing_alpha,
            self.spec.queue_size,
            self.spec.da_len,
            self.spec.dist_align,
            self.spec.dist_uniform,
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
            raise InductiveValidationError("CoMatch requires torch tensors (torch backend).")

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        X_l = ds.X_l
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        X_u_w = ds.X_u_w
        X_u_s0 = ds.X_u_s

        if X_u_w is None:
            raise InductiveValidationError("CoMatch requires X_u_w.")

        if X_u_s0 is None and ds.views:
            for key in ("X_u_s0", "X_u_s_0", "X_u_s", "X_u_strong0"):
                cand = ds.views.get(key)
                if cand is not None:
                    X_u_s0 = cand
                    break

        if X_u_s0 is None:
            raise InductiveValidationError("CoMatch requires X_u_s (first strong view).")

        X_u_s1 = None
        if ds.views:
            for key in ("X_u_s1", "X_u_s_1", "X_u_strong1", "X_u_s2", "X_u_s_2"):
                cand = ds.views.get(key)
                if cand is not None:
                    X_u_s1 = cand
                    break
        if X_u_s1 is None:
            raise InductiveValidationError(
                "CoMatch requires a second strong unlabeled view in views (e.g. views['X_u_s_1'])."
            )

        def _len(x):
            if isinstance(x, dict) and "x" in x:
                return int(x["x"].shape[0])
            if hasattr(x, "shape"):
                return int(x.shape[0])
            return 0

        logger.info(
            "CoMatch sizes: n_labeled=%s n_unlabeled=%s",
            _len(X_l),
            _len(X_u_w),
        )

        if _len(X_l) == 0:
            raise InductiveValidationError("X_l must be non-empty.")
        if _len(X_u_w) == 0:
            raise InductiveValidationError("X_u_w must be non-empty.")
        if _len(X_u_s0) == 0 or _len(X_u_s1) == 0:
            raise InductiveValidationError("X_u_s0/X_u_s1 must be non-empty.")
        if _len(X_u_w) != _len(X_u_s0) or _len(X_u_w) != _len(X_u_s1):
            raise InductiveValidationError("Unlabeled views must have the same number of rows.")

        ensure_float_tensor(X_l, name="X_l")
        ensure_float_tensor(X_u_w, name="X_u_w")
        ensure_float_tensor(X_u_s0, name="X_u_s0")
        ensure_float_tensor(X_u_s1, name="X_u_s1")

        def _feats(x):
            if isinstance(x, dict) and "x" in x:
                return int(x["x"].shape[1])
            return int(x.shape[1])

        if _feats(X_u_s0) != _feats(X_u_w) or _feats(X_u_s1) != _feats(X_u_w):
            raise InductiveValidationError("Unlabeled views must share the same feature size.")

        if y_l.dtype != torch.int64:
            raise InductiveValidationError("y_l must be int64 for torch cross entropy.")

        if self.spec.model_bundle is None:
            raise InductiveValidationError("model_bundle must be provided for CoMatch.")
        bundle = ensure_model_bundle(self.spec.model_bundle)
        model = bundle.model
        optimizer = bundle.optimizer

        def _get_dev(x):
            return (
                x["x"].device if (isinstance(x, dict) and "x" in x) else getattr(x, "device", None)
            )

        ensure_model_device(model, device=_get_dev(X_l))

        if int(self.spec.batch_size) <= 0:
            raise InductiveValidationError("batch_size must be >= 1.")
        if int(self.spec.max_epochs) <= 0:
            raise InductiveValidationError("max_epochs must be >= 1.")
        if float(self.spec.lambda_u) < 0:
            raise InductiveValidationError("lambda_u must be >= 0.")
        if float(self.spec.lambda_c) < 0:
            raise InductiveValidationError("lambda_c must be >= 0.")
        if not (0.0 <= float(self.spec.p_cutoff) <= 1.0):
            raise InductiveValidationError("p_cutoff must be in [0, 1].")
        if float(self.spec.temperature) <= 0:
            raise InductiveValidationError("temperature must be > 0.")
        if not (0.0 <= float(self.spec.contrast_p_cutoff) <= 1.0):
            raise InductiveValidationError("contrast_p_cutoff must be in [0, 1].")
        if not (0.0 <= float(self.spec.smoothing_alpha) <= 1.0):
            raise InductiveValidationError("smoothing_alpha must be in [0, 1].")
        if int(self.spec.queue_size) < 0:
            raise InductiveValidationError("queue_size must be >= 0.")
        if int(self.spec.min_queue_fill) < 0:
            raise InductiveValidationError("min_queue_fill must be >= 0.")
        if int(self.spec.min_queue_fill) > int(self.spec.queue_size):
            raise InductiveValidationError("min_queue_fill must be <= queue_size.")
        if int(self.spec.da_len) < 0:
            raise InductiveValidationError("da_len must be >= 0.")

        self._queue_feats = None
        self._queue_probs = None
        self._queue_ptr = 0
        self._queue_count = 0
        self._da_queue = None
        self._da_ptr = 0
        self._da_count = 0
        self._da_target = None

        steps_l = num_batches(_len(X_l), int(self.spec.batch_size))
        steps_u = num_batches(_len(X_u_w), int(self.spec.batch_size))
        steps_per_epoch = max(int(steps_l), int(steps_u))

        gen_l = torch.Generator().manual_seed(int(seed))
        gen_u = torch.Generator().manual_seed(int(seed) + 1)

        def _slice(data, idx):
            if not isinstance(data, dict):
                return data[idx]
            out = {}
            if "x" in data:
                out["x"] = data["x"][idx]
            if "edge_index" in data:
                try:
                    from torch_geometric.utils import subgraph

                    out["edge_index"], _ = subgraph(
                        idx, data["edge_index"], relabel_nodes=True, num_nodes=_len(data)
                    )
                except ImportError:
                    pass
            for k, v in data.items():
                if k not in ("x", "edge_index"):
                    if hasattr(v, "shape") and v.shape[0] == _len(data):
                        out[k] = v[idx]
                    else:
                        out[k] = v
            return out

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
                _len(X_u_w),
                batch_size=int(self.spec.batch_size),
                generator=gen_u,
                device=_get_dev(X_u_w),
                steps=steps_per_epoch,
            )
            for step, ((x_lb, y_lb), idx_u) in enumerate(zip(iter_l, iter_u_idx, strict=False)):
                x_uw = _slice(X_u_w, idx_u)
                x_us0 = _slice(X_u_s0, idx_u)
                x_us1 = _slice(X_u_s1, idx_u)

                if bool(self.spec.use_cat):
                    inputs = concat_data([x_lb, x_uw, x_us0, x_us1])
                    logits_all, feats_all = _extract_logits_and_features(model(inputs))
                    if int(logits_all.ndim) != 2 or int(feats_all.ndim) != 2:
                        raise InductiveValidationError("Model logits/feat must be 2D tensors.")
                    num_lb = int(_len(x_lb))
                    num_u = int(_len(x_uw))
                    expected = num_lb + num_u + int(_len(x_us0)) + int(_len(x_us1))
                    if int(logits_all.shape[0]) != expected or int(feats_all.shape[0]) != expected:
                        raise InductiveValidationError(
                            "Concatenated logits/feat batch size does not match inputs."
                        )
                    logits_lb = logits_all[:num_lb]
                    feats_lb = feats_all[:num_lb]
                    logits_uw = logits_all[num_lb : num_lb + num_u]
                    feats_uw = feats_all[num_lb : num_lb + num_u]
                    start = num_lb + num_u
                    logits_us0 = logits_all[start : start + num_u]
                    feats_us0 = feats_all[start : start + num_u]
                    logits_us1 = logits_all[start + num_u :]
                    feats_us1 = feats_all[start + num_u :]
                else:
                    logits_lb, feats_lb = _extract_logits_and_features(model(x_lb))
                    logits_us0, feats_us0 = _extract_logits_and_features(model(x_us0))
                    logits_us1, feats_us1 = _extract_logits_and_features(model(x_us1))
                    with torch.no_grad():
                        logits_uw, feats_uw = _extract_logits_and_features(model(x_uw))

                if (
                    int(logits_lb.ndim) != 2
                    or int(logits_uw.ndim) != 2
                    or int(logits_us0.ndim) != 2
                    or int(logits_us1.ndim) != 2
                ):
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                if (
                    int(feats_lb.ndim) != 2
                    or int(feats_uw.ndim) != 2
                    or int(feats_us0.ndim) != 2
                    or int(feats_us1.ndim) != 2
                ):
                    raise InductiveValidationError("Model feats must be 2D (batch, dim).")
                if logits_uw.shape != logits_us0.shape or logits_uw.shape != logits_us1.shape:
                    raise InductiveValidationError("Unlabeled logits shape mismatch.")
                if logits_uw.shape[1] != logits_lb.shape[1]:
                    raise InductiveValidationError("Logits must agree on class dimension.")
                if (
                    feats_uw.shape[1] != feats_us0.shape[1]
                    or feats_uw.shape[1] != feats_us1.shape[1]
                ):
                    raise InductiveValidationError(
                        "Feature dims must match across unlabeled views."
                    )
                if feats_lb.shape[1] != feats_uw.shape[1]:
                    raise InductiveValidationError("Feature dims must match for labeled/unlabeled.")
                if y_lb.min().item() < 0 or y_lb.max().item() >= int(logits_lb.shape[1]):
                    raise InductiveValidationError("y_l labels must be within [0, n_classes).")

                if self._queue_feats is None and int(self.spec.queue_size) > 0:
                    self._init_memory(
                        feat_dim=int(feats_uw.shape[1]),
                        n_classes=int(logits_lb.shape[1]),
                        device=feats_uw.device,
                    )

                sup_loss = torch.nn.functional.cross_entropy(logits_lb, y_lb)

                probs_l = torch.softmax(logits_lb.detach(), dim=1)
                if bool(self.spec.detach_target):
                    with torch.no_grad():
                        probs_uw = _compute_prob(
                            logits_uw, temperature=float(self.spec.temperature)
                        )
                        probs_uw = self._dist_align(probs_uw, probs_l=probs_l)
                        feats_uw_norm = _l2_normalize(feats_uw)
                        probs_smoothed = self._memory_smooth(probs_uw, feats_uw_norm)
                        mask = (probs_smoothed.max(dim=1).values >= float(self.spec.p_cutoff)).to(
                            logits_us0.dtype
                        )
                        q_graph = _build_pseudo_graph(
                            probs_smoothed, threshold=float(self.spec.contrast_p_cutoff)
                        )
                else:
                    probs_uw = _compute_prob(logits_uw, temperature=float(self.spec.temperature))
                    probs_uw = self._dist_align(probs_uw, probs_l=probs_l)
                    feats_uw_norm = _l2_normalize(feats_uw)
                    probs_smoothed = self._memory_smooth(probs_uw, feats_uw_norm)
                    mask = (probs_smoothed.max(dim=1).values >= float(self.spec.p_cutoff)).to(
                        logits_us0.dtype
                    )
                    q_graph = _build_pseudo_graph(
                        probs_smoothed, threshold=float(self.spec.contrast_p_cutoff)
                    )

                feats_lb_norm = _l2_normalize(feats_lb.detach())
                probs_lb = _one_hot(y_lb, n_classes=int(logits_lb.shape[1]))
                feats_bank = torch.cat([feats_uw_norm.detach(), feats_lb_norm], dim=0)
                probs_bank = torch.cat([probs_uw.detach(), probs_lb], dim=0)
                self._update_memory(feats_bank, probs_bank)

                feats_us0_norm = _l2_normalize(feats_us0)
                feats_us1_norm = _l2_normalize(feats_us1)

                if bool(self.spec.hard_label):
                    pseudo = probs_smoothed.argmax(dim=1)
                    loss_u = torch.nn.functional.cross_entropy(logits_us0, pseudo, reduction="none")
                else:
                    log_probs = torch.nn.functional.log_softmax(logits_us0, dim=1)
                    loss_u = -(probs_smoothed * log_probs).sum(dim=1)

                if int(mask.numel()) == 0:
                    unsup_loss = torch.zeros((), device=logits_us0.device)
                else:
                    denom = mask.sum().clamp_min(1.0)
                    unsup_loss = (loss_u * mask).sum() / denom

                contrast_loss = _contrastive_loss(
                    feats_us0_norm,
                    feats_us1_norm,
                    q_graph,
                    temperature=float(self.spec.temperature),
                )

                if step == 0:
                    mask_mean = float(mask.mean().item()) if int(mask.numel()) else 0.0
                    logger.debug(
                        "CoMatch epoch=%s p_cutoff=%s mask_mean=%.3f sup_loss=%.4f "
                        "unsup_loss=%.4f contrast_loss=%.4f",
                        epoch,
                        self.spec.p_cutoff,
                        mask_mean,
                        float(sup_loss.item()),
                        float(unsup_loss.item()),
                        float(contrast_loss.item()),
                    )

                loss = (
                    sup_loss
                    + float(self.spec.lambda_u) * unsup_loss
                    + float(self.spec.lambda_c) * contrast_loss
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
