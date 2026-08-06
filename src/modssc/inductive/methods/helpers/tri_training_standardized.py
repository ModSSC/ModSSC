"""Compatibility path for the historical ModSSC Tri-Training baseline.

The paper-faithful Zhou--Li implementation has different update equations.
Keeping this routine separate makes the public ``standardized`` profile retain
the exact behavior shipped before the article-replication work.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import numpy as np

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.deep_utils import (
    concat_data,
    get_torch_device,
    get_torch_len,
    slice_data,
)
from modssc.inductive.methods.utils import (
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


def fit_standardized_tri_training(
    method: Any,
    data: Any,
    *,
    device: DeviceSpec,
    seed: int,
) -> Any:
    """Fit using the implementation from commit ``f69f3734``."""

    start = perf_counter()
    logger.info("Starting %s.fit", method.info.method_id)
    logger.debug("spec=%s device=%s seed=%s", method.spec, device, seed)
    backend = detect_backend(data.X_l)
    ensure_classifier_backend(method.spec, backend=backend)
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

        X_l = flatten_if_numpy(X_l)
        X_u = flatten_if_numpy(X_u)
        rng = np.random.default_rng(int(seed))
        n_l = int(X_l.shape[0])
        n_boot = max(1, int(round(float(method.spec.bootstrap_ratio) * n_l)))
        clfs = [build_classifier(method.spec, seed=seed + i) for i in range(3)]
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
        while iter_count < int(method.spec.max_iter):
            for i in range(3):
                _train(i)

            new_added = 0
            for i in range(3):
                j, k = [learner for learner in range(3) if learner != i]
                scores_j = predict_scores(clfs[j], X_u, backend=backend)
                scores_k = predict_scores(clfs[k], X_u, backend=backend)
                pred_j = scores_j.argmax(axis=1)
                pred_k = scores_k.argmax(axis=1)
                agree = pred_j == pred_k
                if not np.any(agree):
                    continue

                if method.spec.confidence_threshold is not None:
                    conf_j = scores_j.max(axis=1)
                    conf_k = scores_k.max(axis=1)
                    agree &= conf_j >= float(method.spec.confidence_threshold)
                    agree &= conf_k >= float(method.spec.confidence_threshold)

                idx = np.where(agree)[0]
                if idx.size == 0:
                    continue
                idx = np.asarray(
                    [ii for ii in idx if ii not in added_idx[i]],
                    dtype=np.int64,
                )
                if idx.size == 0:
                    continue

                if method.spec.max_new_labels is not None:
                    scores_agree = (scores_j[idx].max(axis=1) + scores_k[idx].max(axis=1)) / 2.0
                    idx = idx[np.argsort(scores_agree)[::-1][: int(method.spec.max_new_labels)]]

                for ii, label in zip(idx.tolist(), pred_j[idx].tolist(), strict=True):
                    added_idx[i].add(int(ii))
                    added_labels[i][int(ii)] = label
                new_added += int(idx.size)

            if new_added == 0:
                break
            iter_count += 1

        for i in range(3):
            _train(i)
        method._clfs = clfs
        method._backend = backend
        logger.info("Finished %s.fit in %.3fs", method.info.method_id, perf_counter() - start)
        return method

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
    n_boot = max(1, int(round(float(method.spec.bootstrap_ratio) * n_l)))
    gen = torch.Generator(device=get_torch_device(X_l)).manual_seed(int(seed))
    clfs = [build_classifier(method.spec, seed=seed + i) for i in range(3)]
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
                sorted(added_idx[i]),
                dtype=torch.long,
                device=get_torch_device(X_l),
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
    while iter_count < int(method.spec.max_iter):
        for i in range(3):
            _train(i)

        new_added = 0
        for i in range(3):
            j, k = [learner for learner in range(3) if learner != i]
            scores_j = predict_scores(clfs[j], X_u, backend=backend)
            scores_k = predict_scores(clfs[k], X_u, backend=backend)
            pred_j = scores_j.argmax(dim=1)
            pred_k = scores_k.argmax(dim=1)
            agree = pred_j == pred_k
            if not bool(agree.any()):
                continue

            if method.spec.confidence_threshold is not None:
                conf_j = scores_j.max(dim=1).values
                conf_k = scores_k.max(dim=1).values
                agree &= conf_j >= float(method.spec.confidence_threshold)
                agree &= conf_k >= float(method.spec.confidence_threshold)

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

            if method.spec.max_new_labels is not None:
                scores_agree = (
                    scores_j[idx].max(dim=1).values + scores_k[idx].max(dim=1).values
                ) / 2.0
                topk = min(int(method.spec.max_new_labels), int(idx.numel()))
                idx = idx[torch.topk(scores_agree, k=topk).indices]

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
            method.spec.confidence_threshold,
        )
        iter_count += 1

    for i in range(3):
        _train(i)
    method._clfs = clfs
    method._backend = backend
    logger.info("Finished %s.fit in %.3fs", method.info.method_id, perf_counter() - start)
    return method


__all__ = ["fit_standardized_tri_training"]
