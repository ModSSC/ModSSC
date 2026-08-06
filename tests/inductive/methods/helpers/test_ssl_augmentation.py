from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.helpers.ssl_augmentation import ssl_batch_views


class _Augmenter:
    def weak_batch(self, X, *, indices, sample_ids, step):
        return X[indices] + np.asarray(sample_ids)[:, None] * 0 + step

    def pair_batch(self, X, *, indices, sample_ids, step):
        base = X[indices] + np.asarray(sample_ids)[:, None] * 0
        return base + step, base + step + 1


class _FailingAugmenter:
    def weak_batch(self, *_args, **_kwargs):
        raise ValueError("bad replay")


def _call(data):
    return ssl_batch_views(
        data,
        X_l=np.arange(8).reshape(4, 2),
        X_u_w=np.arange(12).reshape(6, 2),
        X_u_s=np.arange(12).reshape(6, 2) + 100,
        idx_l=np.array([1, 3]),
        idx_u=np.array([0, 2]),
        optimization_step=4,
    )


def test_ssl_batch_views_keeps_fixed_views_without_online_runtime() -> None:
    fixed = _call(SimpleNamespace(meta=[]))

    np.testing.assert_array_equal(fixed[0], np.array([[2, 3], [6, 7]]))
    np.testing.assert_array_equal(fixed[1], np.array([[0, 1], [4, 5]]))
    np.testing.assert_array_equal(fixed[2], np.array([[100, 101], [104, 105]]))


def test_ssl_batch_views_materializes_replayable_online_views() -> None:
    data = SimpleNamespace(
        X_u=np.arange(12).reshape(6, 2),
        meta={
            "online_augmentation": _Augmenter(),
            "idx_l": np.array([10, 11, 12, 13]),
            "idx_u": np.array([20, 21, 22, 23, 24, 25]),
        },
    )

    x_l, x_u_w, x_u_s = _call(data)

    np.testing.assert_array_equal(x_l, np.array([[6, 7], [10, 11]]))
    np.testing.assert_array_equal(x_u_w, np.array([[4, 5], [8, 9]]))
    np.testing.assert_array_equal(x_u_s, np.array([[5, 6], [9, 10]]))


def test_ssl_batch_views_requires_online_source_and_absolute_indices() -> None:
    with pytest.raises(InductiveValidationError, match="requires X_u"):
        _call(SimpleNamespace(meta={"online_augmentation": _Augmenter()}))

    with pytest.raises(InductiveValidationError, match="meta.idx_l"):
        _call(
            SimpleNamespace(
                X_u=np.arange(12).reshape(6, 2),
                meta={"online_augmentation": _Augmenter()},
            )
        )


def test_ssl_batch_views_wraps_runtime_index_errors() -> None:
    data = SimpleNamespace(
        X_u=np.arange(12).reshape(6, 2),
        meta={
            "online_augmentation": _FailingAugmenter(),
            "idx_l": np.arange(4),
            "idx_u": np.arange(6),
        },
    )

    with pytest.raises(InductiveValidationError, match="bad replay"):
        _call(data)
