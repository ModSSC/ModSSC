from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.deep_utils import (
    concat_data,
    get_torch_device,
    get_torch_len,
    slice_data,
)
from modssc.inductive.methods.utils import (
    BaseClassifierSpec,
    build_classifier,
    detect_backend,
    ensure_1d_labels,
    ensure_1d_labels_torch,
    ensure_classifier_backend,
    ensure_cpu_device,
    ensure_numpy_data,
    ensure_torch_data,
    flatten_if_numpy,
    predict_scores,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TriTrainingSpec(BaseClassifierSpec):
    max_iter: int = 20
    confidence_threshold: float | None = None
    max_new_labels: int | None = None
    bootstrap_ratio: float = 1.0


class TriTrainingMethod(InductiveMethod):
    """Tri-training with three classifiers (CPU/GPU)."""

    info = MethodInfo(
        method_id="tri_training",
        name="Tri-Training",
        year=2005,
        family="classic",
        supports_gpu=True,
        paper_title="Tri-training: Exploiting unlabeled data using three classifiers",
        paper_pdf="https://people.csail.mit.edu/umangs/papers/tri-training.pdf",
        official_code="",
    )

    def __init__(self, spec: TriTrainingSpec | None = None) -> None:
        self.spec = spec or TriTrainingSpec()
        self._clfs: list[Any] = []
        self._backend: str | None = None

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> TriTrainingMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        backend = detect_backend(data.X_l)
        ensure_classifier_backend(self.spec, backend=backend)
        logger.debug("backend=%s", backend)

        if backend == "numpy":
            ensure_cpu_device(device)
            ds = ensure_numpy_data(data)
            y_l = ensure_1d_labels(ds.y_l, name="y_l")

            if ds.X_u is None:
                raise InductiveValidationError("TriTraining requires X_u (unlabeled data).")

            X_l = np.asarray(ds.X_l)
            X_u = np.asarray(ds.X_u)
            y_l = np.asarray(y_l)
            logger.info(
                "Tri-training sizes: n_labeled=%s n_unlabeled=%s",
                int(X_l.shape[0]),
                int(X_u.shape[0]),
            )

            if X_l.shape[0] == 0:
                raise InductiveValidationError("X_l must be non-empty.")

            # Flatten features if >2D for standard classifiers
            X_l = flatten_if_numpy(X_l)
            X_u = flatten_if_numpy(X_u)

            rng = np.random.default_rng(int(seed))
            n_l = int(X_l.shape[0])
            n_boot = max(1, int(round(float(self.spec.bootstrap_ratio) * n_l)))

            clfs = [build_classifier(self.spec, seed=seed + i) for i in range(3)]
            boot_idx = [rng.choice(n_l, size=n_boot, replace=True) for _ in range(3)]

            added_idx = [set() for _ in range(3)]
            added_labels: list[dict[int, Any]] = [dict() for _ in range(3)]

            def _train(i: int) -> None:
                X_train = X_l[boot_idx[i]]
                y_train = y_l[boot_idx[i]]
                if added_idx[i]:
                    idx = np.asarray(sorted(added_idx[i]), dtype=np.int64)
                    X_train = np.concatenate([X_train, X_u[idx]], axis=0)
                    y_extra = np.asarray([added_labels[i][int(ii)] for ii in idx])
                    y_train = np.concatenate([y_train, y_extra], axis=0)
                clfs[i].fit(X_train, y_train)

            iter_count = 0
            while iter_count < int(self.spec.max_iter):
                for i in range(3):
                    _train(i)

                new_added = 0
                for i in range(3):
                    j, k = [x for x in range(3) if x != i]
                    scores_j = predict_scores(clfs[j], X_u, backend=backend)
                    scores_k = predict_scores(clfs[k], X_u, backend=backend)
                    pred_j = scores_j.argmax(axis=1)
                    pred_k = scores_k.argmax(axis=1)
                    agree = pred_j == pred_k
                    if not np.any(agree):
                        continue

                    if self.spec.confidence_threshold is not None:
                        conf_j = scores_j.max(axis=1)
                        conf_k = scores_k.max(axis=1)
                        agree = agree & (conf_j >= float(self.spec.confidence_threshold))
                        agree = agree & (conf_k >= float(self.spec.confidence_threshold))

                    idx = np.where(agree)[0]
                    if idx.size == 0:
                        continue

                    idx = np.asarray([ii for ii in idx if ii not in added_idx[i]], dtype=np.int64)
                    if idx.size == 0:
                        continue

                    if self.spec.max_new_labels is not None:
                        scores_agree = (scores_j[idx].max(axis=1) + scores_k[idx].max(axis=1)) / 2.0
                        idx = idx[np.argsort(scores_agree)[::-1][: int(self.spec.max_new_labels)]]

                    for ii, label in zip(idx.tolist(), pred_j[idx].tolist(), strict=True):
                        added_idx[i].add(int(ii))
                        added_labels[i][int(ii)] = label
                    new_added += int(idx.size)

                if new_added == 0:
                    break
                iter_count += 1

            for i in range(3):
                _train(i)

            self._clfs = clfs
            self._backend = backend
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        ds = ensure_torch_data(data, device=device)
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u is None:
            raise InductiveValidationError("TriTraining requires X_u (unlabeled data).")

        X_l = ds.X_l
        X_u = ds.X_u
        if int(get_torch_len(X_l)) == 0:
            raise InductiveValidationError("X_l must be non-empty.")
        logger.info(
            "Tri-training sizes: n_labeled=%s n_unlabeled=%s",
            int(get_torch_len(X_l)),
            int(get_torch_len(X_u)),
        )

        n_l = int(get_torch_len(X_l))
        n_boot = max(1, int(round(float(self.spec.bootstrap_ratio) * n_l)))
        gen = torch.Generator(device=get_torch_device(X_l)).manual_seed(int(seed))

        clfs = [build_classifier(self.spec, seed=seed + i) for i in range(3)]
        boot_idx = [
            torch.randint(0, n_l, (n_boot,), generator=gen, device=get_torch_device(X_l))
            for _ in range(3)
        ]

        added_idx = [set() for _ in range(3)]
        added_labels: list[dict[int, Any]] = [dict() for _ in range(3)]

        def _train(i: int) -> None:
            X_train = slice_data(X_l, boot_idx[i])
            y_train = y_l[boot_idx[i]]
            if added_idx[i]:
                idx = torch.tensor(
                    sorted(added_idx[i]), dtype=torch.long, device=get_torch_device(X_l)
                )
                X_train = concat_data([X_train, slice_data(X_u, idx)])
                y_extra = torch.tensor(
                    [added_labels[i][int(ii)] for ii in idx.tolist()],
                    dtype=y_l.dtype,
                    device=get_torch_device(X_l),
                )
                y_train = torch.cat([y_train, y_extra], dim=0)
            clfs[i].fit(X_train, y_train)

        iter_count = 0
        while iter_count < int(self.spec.max_iter):
            for i in range(3):
                _train(i)

            new_added = 0
            for i in range(3):
                j, k = [x for x in range(3) if x != i]
                scores_j = predict_scores(clfs[j], X_u, backend=backend)
                scores_k = predict_scores(clfs[k], X_u, backend=backend)
                pred_j = scores_j.argmax(dim=1)
                pred_k = scores_k.argmax(dim=1)
                agree = pred_j == pred_k
                if not bool(agree.any()):
                    continue

                if self.spec.confidence_threshold is not None:
                    conf_j = scores_j.max(dim=1).values
                    conf_k = scores_k.max(dim=1).values
                    agree = agree & (conf_j >= float(self.spec.confidence_threshold))
                    agree = agree & (conf_k >= float(self.spec.confidence_threshold))

                idx = agree.nonzero(as_tuple=False).reshape(-1)
                if int(idx.numel()) == 0:
                    continue

                idx = torch.tensor(
                    [int(ii) for ii in idx.tolist() if int(ii) not in added_idx[i]],
                    dtype=torch.long,
                    device=get_torch_device(X_l),
                )
                if int(idx.numel()) == 0:
                    continue

                if self.spec.max_new_labels is not None:
                    scores_agree = (
                        scores_j[idx].max(dim=1).values + scores_k[idx].max(dim=1).values
                    ) / 2.0
                    topk = min(int(self.spec.max_new_labels), int(idx.numel()))
                    order = torch.topk(scores_agree, k=topk).indices
                    idx = idx[order]

                for ii, label in zip(idx.tolist(), pred_j[idx].tolist(), strict=True):
                    added_idx[i].add(int(ii))
                    added_labels[i][int(ii)] = label
                new_added += int(idx.numel())

            if new_added == 0:
                logger.debug("Tri-training iter=%s no new labels; stopping.", iter_count)
                break
            logger.debug(
                "Tri-training iter=%s new_added=%s threshold=%s",
                iter_count,
                new_added,
                self.spec.confidence_threshold,
            )
            iter_count += 1

        for i in range(3):
            _train(i)

        self._clfs = clfs
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self._clfs:
            raise RuntimeError("TriTrainingMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict_proba input backend mismatch.")

        if backend == "numpy":
            X = flatten_if_numpy(X)

        scores_list = [predict_scores(clf, X, backend=backend) for clf in self._clfs]

        # Robustly align scores if shapes differ
        shapes = [s.shape[1] for s in scores_list]
        distinct_shapes = set(shapes)

        if len(distinct_shapes) > 1:
            # Check if we can align using classes_ attribute
            has_classes = [hasattr(clf, "classes_") for clf in self._clfs]
            if not all(has_classes):
                raise InductiveValidationError(
                    f"TriTraining classifiers disagree on class counts {shapes}, "
                    "and not all classifiers expose the 'classes_' attribute to allow alignment. "
                    "Cannot safely merge predictions."
                )

            # Align based on classes_
            all_classes_set = set()
            for clf in self._clfs:
                all_classes_set.update(clf.classes_.tolist())

            sorted_classes = sorted(list(all_classes_set))
            global_map = {c: i for i, c in enumerate(sorted_classes)}
            final_n_classes = len(sorted_classes)

            aligned_scores = []
            if backend == "numpy":
                for clf, s in zip(self._clfs, scores_list, strict=False):
                    target = np.zeros((s.shape[0], final_n_classes), dtype=s.dtype)
                    # Map local columns to global columns
                    for local_idx, cls_label in enumerate(clf.classes_):
                        global_idx = global_map[cls_label]
                        target[:, global_idx] = s[:, local_idx]
                    aligned_scores.append(target)

                avg = np.mean(np.stack(aligned_scores, axis=0), axis=0)
            else:
                torch = optional_import("torch", extra="inductive-torch")
                for clf, s in zip(self._clfs, scores_list, strict=False):
                    target = torch.zeros(
                        (s.shape[0], final_n_classes), dtype=s.dtype, device=s.device
                    )
                    for local_idx, cls_label in enumerate(clf.classes_):
                        # Assuming classes_ is numpy or list even for torch backend wrappers
                        # Convert to python generic for safety
                        val = cls_label.item() if hasattr(cls_label, "item") else cls_label
                        global_idx = global_map[val]
                        target[:, global_idx] = s[:, local_idx]
                    aligned_scores.append(target)

                avg = torch.mean(torch.stack(aligned_scores, dim=0), dim=0)

            # Normalize row sums
            if backend == "numpy":
                row_sum = avg.sum(axis=1, keepdims=True)
                row_sum[row_sum == 0.0] = 1.0
                return (avg / row_sum).astype(np.float32, copy=False)
            else:
                row_sum = avg.sum(dim=1, keepdim=True)
                row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
                return avg / row_sum

        # Fast path if shapes match
        if backend == "numpy":
            avg = np.mean(np.stack(scores_list, axis=0), axis=0)
            row_sum = avg.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0.0] = 1.0
            return (avg / row_sum).astype(np.float32, copy=False)
        else:
            torch = optional_import("torch", extra="inductive-torch")
            avg = torch.mean(torch.stack(scores_list, dim=0), dim=0)
            row_sum = avg.sum(dim=1, keepdim=True)
            row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
            return avg / row_sum

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        backend = self._backend or detect_backend(X)
        if backend == "numpy":
            idx = proba.argmax(axis=1)
            classes = getattr(self._clfs[0], "classes_", None)
            if classes is None:
                return idx
            return np.asarray(classes)[idx]
        idx = proba.argmax(dim=1)
        classes_t = getattr(self._clfs[0], "classes_t_", None)
        if classes_t is None:
            return idx
        return classes_t[idx]
