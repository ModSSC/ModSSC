from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.deep_utils import (
    concat_data,
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


def _forward_shared(bundle: TorchModelBundle, X: Any) -> Any:
    meta = bundle.meta or {}
    if isinstance(meta, Mapping):
        forward = meta.get("forward_features") or meta.get("feature_extractor")
        if callable(forward):
            return forward(X)
    out = bundle.model(X)
    torch = optional_import("torch", extra="inductive-torch")
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, Mapping) and "feat" in out:
        return extract_features(out)
    raise InductiveValidationError(
        "shared_bundle.model must return a torch.Tensor feature embedding, a mapping with key "
        "'feat', or provide meta['forward_features']."
    )


def _forward_head(bundle: TorchModelBundle, features: Any) -> Any:
    meta = bundle.meta or {}
    if isinstance(meta, Mapping):
        head = meta.get("forward_head") or meta.get("head")
        if callable(head):
            return extract_logits(head(features))
    return extract_logits(bundle.model(features))


def _soft_cross_entropy(logits: Any, targets: Any) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    log_probs = torch.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def _one_hot(labels: Any, *, n_classes: int) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    return torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=int(n_classes)).to(
        dtype=torch.float32
    )


def _output_smearing(labels: Any, *, n_classes: int, std: float, generator: Any) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    one_hot = _one_hot(labels, n_classes=int(n_classes))
    if float(std) <= 0:
        return one_hot
    noise = torch.randn(
        (int(one_hot.shape[0]), int(n_classes)),
        generator=generator,
        device=one_hot.device,
        dtype=one_hot.dtype,
    )
    noise = torch.relu(noise * float(std))
    smeared = one_hot + noise
    denom = smeared.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return smeared / denom


def _sample_pool(X_u: Any, *, n_pool: int, generator: Any) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    n_u = int(get_torch_len(X_u))
    if int(n_pool) >= n_u:
        return X_u
    idx = torch.randperm(n_u, generator=generator, device="cpu")[: int(n_pool)]
    if getattr(get_torch_device(X_u), "type", "cpu") != "cpu":
        idx = idx.to(device=get_torch_device(X_u))
    return slice_data(X_u, idx)


def _dropout_filter(
    X: Any,
    labels: Any,
    *,
    shared_bundle: TorchModelBundle,
    head_j: TorchModelBundle,
    head_h: TorchModelBundle,
    passes: int,
    drop_fraction: float,
    batch_size: int,
    freeze_bn: bool,
) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    n = int(get_torch_len(X))
    if n == 0 or int(passes) <= 1:
        return torch.ones((n,), dtype=torch.bool, device=get_torch_device(X))
    max_mismatches = int(math.floor(float(passes) * float(drop_fraction)))
    counts = torch.zeros((n,), dtype=torch.int64, device=get_torch_device(X))

    shared_model = shared_bundle.model
    head_j_model = head_j.model
    head_h_model = head_h.model
    was_shared = shared_model.training
    was_j = head_j_model.training
    was_h = head_h_model.training

    shared_model.train()
    head_j_model.train()
    head_h_model.train()
    with (
        freeze_batchnorm(shared_model, enabled=bool(freeze_bn)),
        freeze_batchnorm(head_j_model, enabled=bool(freeze_bn)),
        freeze_batchnorm(head_h_model, enabled=bool(freeze_bn)),
        torch.no_grad(),
    ):
        for _ in range(int(passes)):
            for start in range(0, n, int(batch_size)):
                end = min(start + int(batch_size), n)
                x_batch = slice_data(X, slice(start, end))
                feats = _forward_shared(shared_bundle, x_batch)
                logits_j = _forward_head(head_j, feats)
                logits_h = _forward_head(head_h, feats)
                if int(logits_j.ndim) != 2 or int(logits_h.ndim) != 2:
                    raise InductiveValidationError("Head logits must be 2D (batch, classes).")
                if logits_j.shape != logits_h.shape:
                    raise InductiveValidationError("Head logits must have the same shape.")
                probs = (torch.softmax(logits_j, dim=1) + torch.softmax(logits_h, dim=1)) / 2.0
                pred = probs.argmax(dim=1)
                counts[start:end] += (pred != labels[start:end]).to(torch.int64)

    shared_model.train(was_shared)
    head_j_model.train(was_j)
    head_h_model.train(was_h)
    return counts <= max_mismatches


def _label_unlabeled(
    X_pool: Any,
    *,
    shared_bundle: TorchModelBundle,
    head_j: TorchModelBundle,
    head_h: TorchModelBundle,
    sigma_t: float,
    stability_passes: int,
    drop_fraction: float,
    batch_size: int,
    freeze_bn: bool,
) -> tuple[Any, Any]:
    torch = optional_import("torch", extra="inductive-torch")
    n_pool = int(get_torch_len(X_pool))
    if n_pool == 0:
        return X_pool, torch.empty((0,), device=get_torch_device(X_pool), dtype=torch.int64)

    shared_model = shared_bundle.model
    head_j_model = head_j.model
    head_h_model = head_h.model
    was_shared = shared_model.training
    was_j = head_j_model.training
    was_h = head_h_model.training

    shared_model.eval()
    head_j_model.eval()
    head_h_model.eval()

    idx_keep: list[Any] = []
    labels_keep: list[Any] = []
    with torch.no_grad():
        for start in range(0, n_pool, int(batch_size)):
            end = min(start + int(batch_size), n_pool)
            x_batch = slice_data(X_pool, slice(start, end))
            feats = _forward_shared(shared_bundle, x_batch)
            logits_j = _forward_head(head_j, feats)
            logits_h = _forward_head(head_h, feats)
            if int(logits_j.ndim) != 2 or int(logits_h.ndim) != 2:
                raise InductiveValidationError("Head logits must be 2D (batch, classes).")
            if logits_j.shape != logits_h.shape:
                raise InductiveValidationError("Head logits must have the same shape.")
            probs_j = torch.softmax(logits_j, dim=1)
            probs_h = torch.softmax(logits_h, dim=1)
            pred_j = probs_j.argmax(dim=1)
            pred_h = probs_h.argmax(dim=1)
            conf_j = probs_j.max(dim=1).values
            conf_h = probs_h.max(dim=1).values
            avg_conf = (conf_j + conf_h) / 2.0
            agree = pred_j == pred_h
            mask = agree & (avg_conf >= float(sigma_t))
            if bool(mask.any()):
                local_idx = mask.nonzero(as_tuple=False).reshape(-1)
                idx_keep.append(local_idx + int(start))
                labels_keep.append(pred_j[mask])

    shared_model.train(was_shared)
    head_j_model.train(was_j)
    head_h_model.train(was_h)

    if not idx_keep:
        return slice_data(X_pool, slice(0, 0)), torch.empty(
            (0,), device=get_torch_device(X_pool), dtype=torch.int64
        )

    idx = torch.cat(idx_keep, dim=0)
    labels = torch.cat(labels_keep, dim=0)
    X_pl = slice_data(X_pool, idx)
    if int(stability_passes) > 1:
        keep_mask = _dropout_filter(
            X_pl,
            labels,
            shared_bundle=shared_bundle,
            head_j=head_j,
            head_h=head_h,
            passes=int(stability_passes),
            drop_fraction=float(drop_fraction),
            batch_size=int(batch_size),
            freeze_bn=bool(freeze_bn),
        )
        if bool(keep_mask.any()):
            X_pl = slice_data(X_pl, keep_mask)
            labels = labels[keep_mask]
        else:
            X_pl = slice_data(X_pl, slice(0, 0))
            labels = labels[:0]
    return X_pl, labels


def _train_head(
    X: Any,
    targets: Any,
    *,
    shared_bundle: TorchModelBundle,
    head_bundle: TorchModelBundle,
    update_shared: bool,
    batch_size: int,
    epochs: int,
    seed: int,
) -> None:
    torch = optional_import("torch", extra="inductive-torch")
    n = int(get_torch_len(X))
    if n == 0:
        return
    steps = num_batches(n, int(batch_size))
    gen = torch.Generator().manual_seed(int(seed))

    shared_model = shared_bundle.model
    head_model = head_bundle.model
    opt_shared = shared_bundle.optimizer
    opt_head = head_bundle.optimizer

    for _ in range(int(epochs)):
        iter_batches = cycle_batches(
            X,
            targets,
            batch_size=int(batch_size),
            generator=gen,
            steps=steps,
        )
        if update_shared:
            shared_model.train()
        else:
            shared_model.eval()
        head_model.train()
        for x_batch, y_batch in iter_batches:
            freeze_bn = int(get_torch_len(x_batch)) < 2
            with (
                freeze_batchnorm(shared_model, enabled=freeze_bn),
                freeze_batchnorm(head_model, enabled=freeze_bn),
            ):
                if update_shared:
                    feats = _forward_shared(shared_bundle, x_batch)
                else:
                    with torch.no_grad():
                        feats = _forward_shared(shared_bundle, x_batch)
                logits = _forward_head(head_bundle, feats)
                if int(logits.ndim) != 2:
                    raise InductiveValidationError("Head logits must be 2D (batch, classes).")
                if int(logits.shape[0]) != int(y_batch.shape[0]):
                    raise InductiveValidationError("Logits batch size does not match targets.")
                loss = _soft_cross_entropy(logits, y_batch)
                opt_head.zero_grad()
                if update_shared:
                    opt_shared.zero_grad()
                loss.backward()
                opt_head.step()
                if update_shared:
                    opt_shared.step()


def _validate_bundle_set(
    *,
    shared_bundle: TorchModelBundle | None,
    head_bundles: tuple[TorchModelBundle, TorchModelBundle, TorchModelBundle] | None,
    device: Any,
) -> tuple[TorchModelBundle, tuple[TorchModelBundle, TorchModelBundle, TorchModelBundle]]:
    if shared_bundle is None:
        raise InductiveValidationError("shared_bundle must be provided for TriNet.")
    if head_bundles is None:
        raise InductiveValidationError("head_bundles must be provided for TriNet.")
    if len(head_bundles) != 3:
        raise InductiveValidationError("head_bundles must contain exactly three bundles.")

    shared_bundle = ensure_model_bundle(shared_bundle)
    head_bundles = tuple(ensure_model_bundle(b) for b in head_bundles)

    ensure_model_device(shared_bundle.model, device=device)
    for bundle in head_bundles:
        ensure_model_device(bundle.model, device=device)

    shared_ids = {id(p) for p in shared_bundle.model.parameters()}
    head_ids = [{id(p) for p in bundle.model.parameters()} for bundle in head_bundles]
    for idx, ids in enumerate(head_ids):
        if shared_ids & ids:
            raise InductiveValidationError(
                f"head_bundles[{idx}] shares parameters with shared_bundle."
            )
    for i in range(len(head_ids)):
        for j in range(i + 1, len(head_ids)):
            if head_ids[i] & head_ids[j]:
                raise InductiveValidationError(
                    f"head_bundles[{i}] shares parameters with head_bundles[{j}]."
                )

    return shared_bundle, head_bundles


@dataclass(frozen=True)
class TriNetSpec:
    """Specification for Tri-Net (torch-only)."""

    shared_bundle: TorchModelBundle | None = None
    head_bundles: tuple[TorchModelBundle, TorchModelBundle, TorchModelBundle] | None = None
    sigma0: float = 0.95
    sigma_os: float = 0.25
    sigma_decay: float = 0.05
    output_smearing_std: float = 0.05
    pool_base: int = 1000
    max_rounds: int = 30
    fine_tune_interval: int = 4
    batch_size: int = 16
    init_epochs: int = 1
    train_epochs: int = 1
    des_passes: int = 6
    des_drop_fraction: float = 1.0 / 3.0
    labeling_stability_passes: int = 1
    freeze_bn: bool = True


class TriNetMethod(InductiveMethod):
    """Tri-Net semi-supervised deep learning (torch-only)."""

    info = MethodInfo(
        method_id="trinet",
        name="Tri-Net",
        year=2018,
        family="agreement",
        supports_gpu=True,
        paper_title="Tri-net for Semi-Supervised Deep Learning",
        paper_pdf="",
        official_code="",
    )

    def __init__(self, spec: TriNetSpec | None = None) -> None:
        self.spec = spec or TriNetSpec()
        self._shared_bundle: TorchModelBundle | None = None
        self._head_bundles: tuple[TorchModelBundle, TorchModelBundle, TorchModelBundle] | None = (
            None
        )
        self._backend: str | None = None

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> TriNetMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        if data is None:
            raise InductiveValidationError("data must not be None.")

        backend = detect_backend(data.X_l)
        if backend != "torch":
            raise InductiveValidationError("TriNet requires torch tensors (torch backend).")

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u is None:
            raise InductiveValidationError("TriNet requires X_u (unlabeled data).")

        X_l = ds.X_l
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        X_u = ds.X_u
        logger.info(
            "TriNet sizes: n_labeled=%s n_unlabeled=%s",
            int(get_torch_len(X_l)),
            int(get_torch_len(X_u)),
        )

        if int(get_torch_len(X_l)) == 0:
            raise InductiveValidationError("X_l must be non-empty.")
        if int(get_torch_len(X_u)) == 0:
            raise InductiveValidationError("X_u must be non-empty.")

        ensure_float_tensor(X_l, name="X_l")
        ensure_float_tensor(X_u, name="X_u")

        if y_l.dtype != torch.int64:
            raise InductiveValidationError("y_l must be int64 for torch cross entropy.")

        shared_bundle, head_bundles = _validate_bundle_set(
            shared_bundle=self.spec.shared_bundle,
            head_bundles=self.spec.head_bundles,
            device=get_torch_device(X_l),
        )

        if int(self.spec.pool_base) <= 0:
            raise InductiveValidationError("pool_base must be >= 1.")
        if int(self.spec.max_rounds) <= 0:
            raise InductiveValidationError("max_rounds must be >= 1.")
        if int(self.spec.batch_size) <= 0:
            raise InductiveValidationError("batch_size must be >= 1.")
        if int(self.spec.init_epochs) <= 0:
            raise InductiveValidationError("init_epochs must be >= 1.")
        if int(self.spec.train_epochs) <= 0:
            raise InductiveValidationError("train_epochs must be >= 1.")
        if int(self.spec.des_passes) <= 0:
            raise InductiveValidationError("des_passes must be >= 1.")
        if int(self.spec.labeling_stability_passes) <= 0:
            raise InductiveValidationError("labeling_stability_passes must be >= 1.")
        if float(self.spec.output_smearing_std) < 0:
            raise InductiveValidationError("output_smearing_std must be >= 0.")
        if float(self.spec.sigma0) < 0 or float(self.spec.sigma0) > 1:
            raise InductiveValidationError("sigma0 must be in [0, 1].")
        if float(self.spec.sigma_os) < 0:
            raise InductiveValidationError("sigma_os must be >= 0.")
        if float(self.spec.sigma_decay) < 0:
            raise InductiveValidationError("sigma_decay must be >= 0.")
        if not (0.0 <= float(self.spec.des_drop_fraction) <= 1.0):
            raise InductiveValidationError("des_drop_fraction must be in [0, 1].")

        n_classes = int(y_l.max().item()) + 1
        if y_l.min().item() < 0:
            raise InductiveValidationError("y_l labels must be non-negative.")
        y_l_onehot = _one_hot(y_l, n_classes=n_classes)

        # Initialization with output smearing to create diverse heads.
        y_os_sets = [
            _output_smearing(
                y_l,
                n_classes=n_classes,
                std=float(self.spec.output_smearing_std),
                generator=torch.Generator(device=y_l.device).manual_seed(int(seed) + i),
            )
            for i in range(3)
        ]
        for i, y_os in enumerate(y_os_sets):
            _train_head(
                X_l,
                y_os,
                shared_bundle=shared_bundle,
                head_bundle=head_bundles[i],
                update_shared=True,
                batch_size=int(self.spec.batch_size),
                epochs=int(self.spec.init_epochs),
                seed=int(seed) + 10 + i,
            )

        flag_os = True
        sigma = float(self.spec.sigma0)
        n_u = int(get_torch_len(X_u))

        for t in range(1, int(self.spec.max_rounds) + 1):
            n_pool = min(int(self.spec.pool_base) * (2 ** int(t)), n_u)
            if (
                n_pool == n_u
                and int(self.spec.fine_tune_interval) > 0
                and int(t) % int(self.spec.fine_tune_interval) == 0
            ):
                y_os_sets = [
                    _output_smearing(
                        y_l,
                        n_classes=n_classes,
                        std=float(self.spec.output_smearing_std),
                        generator=torch.Generator(device=y_l.device).manual_seed(
                            int(seed) + 1000 + t * 10 + i
                        ),
                    )
                    for i in range(3)
                ]
                for i, y_os in enumerate(y_os_sets):
                    _train_head(
                        X_l,
                        y_os,
                        shared_bundle=shared_bundle,
                        head_bundle=head_bundles[i],
                        update_shared=True,
                        batch_size=int(self.spec.batch_size),
                        epochs=int(self.spec.init_epochs),
                        seed=int(seed) + 2000 + t * 10 + i,
                    )
                flag_os = True
                sigma = max(0.0, sigma - float(self.spec.sigma_decay))
                continue

            sigma_t = sigma - float(self.spec.sigma_os) if flag_os else sigma
            flag_os = False
            sigma_t = min(1.0, max(0.0, sigma_t))

            pool_gen = torch.Generator().manual_seed(int(seed) + 3000 + t)
            X_pool = _sample_pool(X_u, n_pool=n_pool, generator=pool_gen)

            pseudo_counts = []
            for v in range(3):
                j, h = [idx for idx in range(3) if idx != v]
                X_pl, y_pl = _label_unlabeled(
                    X_pool,
                    shared_bundle=shared_bundle,
                    head_j=head_bundles[j],
                    head_h=head_bundles[h],
                    sigma_t=sigma_t,
                    stability_passes=int(self.spec.labeling_stability_passes),
                    drop_fraction=float(self.spec.des_drop_fraction),
                    batch_size=int(self.spec.batch_size),
                    freeze_bn=bool(self.spec.freeze_bn),
                )
                if int(self.spec.des_passes) > 1 and int(get_torch_len(X_pl)) > 0:
                    keep_mask = _dropout_filter(
                        X_pl,
                        y_pl,
                        shared_bundle=shared_bundle,
                        head_j=head_bundles[j],
                        head_h=head_bundles[h],
                        passes=int(self.spec.des_passes),
                        drop_fraction=float(self.spec.des_drop_fraction),
                        batch_size=int(self.spec.batch_size),
                        freeze_bn=bool(self.spec.freeze_bn),
                    )
                    if bool(keep_mask.any()):
                        X_pl = slice_data(X_pl, keep_mask)
                        y_pl = y_pl[keep_mask]
                    else:
                        X_pl = slice_data(X_pl, slice(0, 0))
                        y_pl = y_pl[:0]

                if int(get_torch_len(X_pl)) > 0:
                    y_pl_onehot = _one_hot(y_pl, n_classes=n_classes)
                    X_hat = concat_data([X_l, X_pl])
                    y_hat = torch.cat([y_l_onehot, y_pl_onehot], dim=0)
                else:
                    X_hat = X_l
                    y_hat = y_l_onehot

                _train_head(
                    X_hat,
                    y_hat,
                    shared_bundle=shared_bundle,
                    head_bundle=head_bundles[v],
                    update_shared=(v == 0),
                    batch_size=int(self.spec.batch_size),
                    epochs=int(self.spec.train_epochs),
                    seed=int(seed) + 4000 + t * 10 + v,
                )
                pseudo_counts.append(int(get_torch_len(X_pl)))

            logger.debug(
                "TriNet round=%s pool=%s sigma_t=%.3f pseudo=%s",
                t,
                n_pool,
                float(sigma_t),
                pseudo_counts,
            )

        self._shared_bundle = shared_bundle
        self._head_bundles = head_bundles
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, X: Any) -> Any:
        if self._shared_bundle is None or self._head_bundles is None:
            raise RuntimeError("TriNetMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if backend != "torch":
            raise InductiveValidationError("TriNet predict_proba requires torch tensors.")
        torch = optional_import("torch", extra="inductive-torch")
        if not isinstance(X, torch.Tensor) and not isinstance(X, dict):
            raise InductiveValidationError("predict_proba requires torch.Tensor or dict inputs.")

        shared_model = self._shared_bundle.model
        head_models = [bundle.model for bundle in self._head_bundles]
        was_shared = shared_model.training
        was_heads = [m.training for m in head_models]
        shared_model.eval()
        for model in head_models:
            model.eval()
        with torch.no_grad():
            feats = _forward_shared(self._shared_bundle, X)
            probs = []
            for bundle in self._head_bundles:
                logits = _forward_head(bundle, feats)
                if int(logits.ndim) != 2:
                    raise InductiveValidationError("Head logits must be 2D (batch, classes).")
                probs.append(torch.softmax(logits, dim=1))
            n_classes = {int(p.shape[1]) for p in probs}
            if len(n_classes) != 1:
                raise InductiveValidationError("TriNet heads disagree on class count.")
            avg = torch.mean(torch.stack(probs, dim=0), dim=0)
        if was_shared:
            shared_model.train()
        for model, was in zip(head_models, was_heads, strict=True):
            if was:
                model.train()
        return avg

    def predict(self, X: Any) -> Any:
        proba = self.predict_proba(X)
        return proba.argmax(dim=1)
