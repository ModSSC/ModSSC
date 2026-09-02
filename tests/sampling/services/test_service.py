from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import modssc.sampling.api as api
import modssc.sampling.services.service as service
from modssc.sampling.errors import SamplingValidationError
from modssc.sampling.plan import HoldoutSplitSpec, LabelingSpec, PartitionSpec, SamplingPlan


def test_api_module_aliases_internal_service() -> None:
    assert api is service
    assert api.sample is service.sample
    assert api.default_split_cache_dir is service.default_split_cache_dir
    assert api._idx_to_mask is service._idx_to_mask


def test_partition_scope_fails_closed_when_labels_fall_outside_final_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace(
        train=SimpleNamespace(y=np.array([0, 1, 0, 1])),
        test=None,
        meta={"dataset_fingerprint": "dataset"},
    )
    plan = SamplingPlan(
        split=HoldoutSplitSpec(
            test_size=1,
            val_size=0,
            stratify=False,
            shuffle=False,
        ),
        labeling=LabelingSpec(
            mode="count",
            value=1,
            selection_scope="partition",
        ),
    )
    monkeypatch.setattr(
        service,
        "select_labeled",
        lambda **_kwargs: np.array([0], dtype=np.int64),
    )

    with pytest.raises(SamplingValidationError, match="outside the final train"):
        service.sample(dataset, plan=plan, seed=1, save=False)


def test_class_balanced_partition_truncation_preserves_stream_order() -> None:
    selected = service._select_partition_indices(
        n_samples=5,
        y=np.array([0, 0, 0, 1, 1]),
        spec=PartitionSpec(max_samples=3, shuffle=False, ordering="class_balanced_stream"),
        rng=np.random.default_rng(0),
    )

    assert selected.dtype == np.int64
    assert selected.shape == (3,)


def test_class_balanced_stream_handles_empty_population() -> None:
    result = service._class_balanced_stream_order(np.array([], dtype=np.int64))
    assert result.dtype == np.int64
    assert result.size == 0
