from __future__ import annotations

import numpy as np
import pytest

from bench.orchestrators.sampling import _concat_rows, prepare_dataset
from modssc.data_loader.types import LoadedDataset, Split


def _plan(*, merge: bool) -> dict[str, object]:
    return {"policy": {"merge_official_splits": merge}}


def _dataset() -> LoadedDataset:
    return LoadedDataset(
        train=Split(X=np.array([[1.0], [2.0]]), y=np.array([0, 1])),
        test=Split(X=np.array([[3.0]]), y=np.array([1])),
        meta={"dataset_fingerprint": "a" * 64, "modality": "vision"},
    )


def test_prepare_dataset_is_identity_without_merge() -> None:
    dataset = _dataset()
    assert prepare_dataset(dataset, plan_dict=_plan(merge=False)) is dataset


def test_prepare_dataset_merges_official_splits_and_derives_identity() -> None:
    first = prepare_dataset(_dataset(), plan_dict=_plan(merge=True))
    second = prepare_dataset(_dataset(), plan_dict=_plan(merge=True))

    assert first.test is None
    np.testing.assert_array_equal(first.train.X, np.array([[1.0], [2.0], [3.0]]))
    np.testing.assert_array_equal(first.train.y, np.array([0, 1, 1]))
    assert first.meta["official_splits_merged"] is True
    assert first.meta["dataset_fingerprint_source"] == "a" * 64
    assert first.meta["dataset_fingerprint"] == second.meta["dataset_fingerprint"]
    assert first.meta["dataset_fingerprint"] != "a" * 64


def test_concat_rows_supports_matching_mappings_and_torch() -> None:
    merged = _concat_rows(
        {"x": np.array([[1]]), "mask": np.array([[True]])},
        {"x": np.array([[2]]), "mask": np.array([[False]])},
        field="X",
    )
    np.testing.assert_array_equal(merged["x"], np.array([[1], [2]]))

    torch = pytest.importorskip("torch")
    tensor = _concat_rows(torch.tensor([[1]]), torch.tensor([[2]]), field="X")
    torch.testing.assert_close(tensor, torch.tensor([[1], [2]]))


def test_prepare_dataset_rejects_invalid_merge_contracts() -> None:
    base = _dataset()
    with pytest.raises(ValueError, match="official test"):
        prepare_dataset(
            LoadedDataset(train=base.train, meta=base.meta),
            plan_dict=_plan(merge=True),
        )
    with pytest.raises(ValueError, match="graph edges or masks"):
        prepare_dataset(
            LoadedDataset(
                train=Split(X=base.train.X, y=base.train.y, edges=np.array([[0], [1]])),
                test=base.test,
                meta=base.meta,
            ),
            plan_dict=_plan(merge=True),
        )
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_dataset(
            LoadedDataset(train=base.train, test=base.test, meta={}),
            plan_dict=_plan(merge=True),
        )
    with pytest.raises(ValueError, match="keys differ"):
        _concat_rows({"x": np.array([1])}, {"y": np.array([2])}, field="X")
    with pytest.raises(ValueError, match="unsupported"):
        _concat_rows([1], [2], field="X")
