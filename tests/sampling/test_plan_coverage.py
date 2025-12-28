from __future__ import annotations

import pytest

from modssc.sampling.plan import (
    HoldoutSplitSpec,
    ImbalanceSpec,
    KFoldSplitSpec,
    LabelingSpec,
    SamplingPlan,
    SamplingPolicy,
    _ensure_mapping,
)


def test_holdout_split_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown split kind"):
        HoldoutSplitSpec.from_dict({"kind": "nope"})


def test_kfold_split_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown split kind"):
        KFoldSplitSpec.from_dict({"kind": "nope"})


def test_labeling_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown labeling mode"):
        LabelingSpec.from_dict({"mode": "nope"})


def test_labeling_rejects_invalid_fixed_indices() -> None:
    with pytest.raises(ValueError, match="fixed_indices"):
        LabelingSpec.from_dict({"fixed_indices": "bad"})


def test_labeling_accepts_fixed_indices_list() -> None:
    spec = LabelingSpec.from_dict({"fixed_indices": [1, 2, 3], "mode": "count", "value": 2})
    assert spec.fixed_indices == [1, 2, 3]


def test_labeling_rejects_invalid_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown labeling strategy"):
        LabelingSpec.from_dict({"strategy": "nope"})


def test_imbalance_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown imbalance kind"):
        ImbalanceSpec.from_dict({"kind": "nope"})


def test_imbalance_rejects_unknown_apply_to() -> None:
    with pytest.raises(ValueError, match="Unknown imbalance apply_to"):
        ImbalanceSpec.from_dict({"apply_to": "nope"})


def test_sampling_plan_rejects_unknown_split_kind() -> None:
    with pytest.raises(ValueError, match="Unknown split kind"):
        SamplingPlan.from_dict({"split": {"kind": "nope"}})


def test_ensure_mapping_none_returns_empty() -> None:
    assert _ensure_mapping(None, "split") == {}


def test_ensure_mapping_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="split must be a mapping"):
        _ensure_mapping(123, "split")


def test_sampling_policy_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unknown keys in policy"):
        SamplingPolicy.from_dict({"unknown": True})
