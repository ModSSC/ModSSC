from __future__ import annotations

import numpy as np
import pytest

from modssc.data_loader.types import LoadedDataset, Split
from modssc.preprocess import resolve_fit_indices
from modssc.sampling.result import SamplingResult


def _dataset() -> LoadedDataset:
    return LoadedDataset(
        train=Split(
            X=np.zeros((5, 2), dtype=np.float32),
            y=np.arange(5, dtype=np.int64),
        )
    )


def _result(*, graph: bool) -> SamplingResult:
    masks = {
        "train": np.array([True, True, True, False, False]),
        "val": np.array([False, False, False, True, False]),
        "test": np.array([False, False, False, False, True]),
        "labeled": np.array([True, False, False, False, False]),
        "unlabeled": np.array([False, True, True, False, False]),
    }
    indices = {
        "train": np.array([0, 1, 2]),
        "val": np.array([3]),
        "test": np.array([4]),
        "train_labeled": np.array([0]),
        "train_unlabeled": np.array([1, 2]),
    }
    return SamplingResult(
        schema_version=1,
        created_at="",
        dataset_fingerprint="dataset",
        split_fingerprint="split",
        plan={},
        masks=masks if graph else {},
        indices={} if graph else indices,
        refs={} if graph else {key: "train" for key in indices},
    )


@pytest.mark.parametrize("graph", [False, True])
def test_resolve_fit_indices_uses_native_sampling_semantics(graph: bool) -> None:
    sampling = _result(graph=graph)

    np.testing.assert_array_equal(
        resolve_fit_indices(
            dataset=_dataset(),
            sampling=sampling,
            fit_on="train_labeled",
        ),
        [0],
    )
    np.testing.assert_array_equal(
        resolve_fit_indices(
            dataset=_dataset(),
            sampling=sampling,
            fit_on="train_unlabeled",
        ),
        [1, 2],
    )


def test_resolve_fit_indices_allows_no_fitting_scope() -> None:
    assert (
        resolve_fit_indices(
            dataset=_dataset(),
            sampling=_result(graph=False),
            fit_on=None,
        )
        is None
    )
