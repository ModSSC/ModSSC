from __future__ import annotations

import pytest

from modssc.sampling.fingerprint import stable_hash
from modssc.sampling.plan import (
    FixedIndicesArtifactSpec,
    HoldoutSplitSpec,
    ImbalanceSpec,
    KFoldSplitSpec,
    LabelingSpec,
    OrderedPartitionArtifactSpec,
    PartitionSpec,
    SamplingComponentSeeds,
    SamplingPlan,
    SamplingPolicy,
    _ensure_mapping,
)


def test_holdout_split_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown split kind"):
        HoldoutSplitSpec.from_dict({"kind": "nope"})


def test_holdout_split_serializes_exact_sizes_and_rejects_invalid_controls() -> None:
    spec = HoldoutSplitSpec(test_size=2, val_size=1, holdout_from="end")
    assert spec.as_dict()["test_size"] == 2
    assert spec.as_dict()["val_size"] == 1
    assert spec.as_dict()["holdout_from"] == "end"

    with pytest.raises(ValueError, match="test_size"):
        HoldoutSplitSpec.from_dict({"test_size": True})
    with pytest.raises(ValueError, match="holdout_from"):
        HoldoutSplitSpec.from_dict({"holdout_from": "middle"})


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


def test_labeling_class_counts_round_trip_is_normalized_and_sorted() -> None:
    spec = LabelingSpec.from_dict(
        {
            "mode": "count",
            "value": 12,
            "strategy": "random",
            "class_counts": {"1": 3, "0": 9},
            "selection_order": "permutation",
        }
    )

    assert spec.class_counts == {"1": 3, "0": 9}
    assert list(spec.as_dict()["class_counts"]) == ["0", "1"]
    assert LabelingSpec.from_dict(spec.as_dict()) == spec
    assert "class_counts" not in LabelingSpec().as_dict()


@pytest.mark.parametrize(
    ("class_counts", "message"),
    [
        ([], "must be a mapping"),
        ({}, "must not be empty"),
        ({"0": True, "1": 1}, "non-negative integers"),
        ({"0": 1.5, "1": 1}, "non-negative integers"),
        ({"0": -1, "1": 2}, "non-negative integers"),
        ({"0": 0, "1": 0}, "at least one labeled sample"),
        ({1: 1, "1": 1}, "duplicate labels"),
    ],
)
def test_labeling_class_counts_rejects_invalid_mappings(class_counts, message) -> None:
    with pytest.raises(ValueError, match=message):
        LabelingSpec.from_dict({"mode": "count", "value": 2, "class_counts": class_counts})


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "fraction", "value": 1.0},
        {"mode": "count", "value": 11},
    ],
)
def test_labeling_class_counts_requires_matching_count_mode_and_total(change) -> None:
    raw = {"mode": "count", "value": 12, "class_counts": {"0": 9, "1": 3}}
    raw.update(change)
    with pytest.raises(ValueError, match="mode='count'.*value equal"):
        LabelingSpec.from_dict(raw)


def test_fixed_indices_artifact_round_trip() -> None:
    raw = {
        "path": "/immutable/permutations.npz",
        "sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "key": "perm",
        "index_stride": 5,
        "index_offset": 2,
        "expected_size": 30,
        "expected_per_class": 3,
    }
    artifact = FixedIndicesArtifactSpec.from_dict(raw)
    assert artifact.as_dict() == raw
    labeling = LabelingSpec.from_dict({"fixed_indices_artifact": raw})
    assert labeling.fixed_indices_artifact == artifact
    assert LabelingSpec.from_dict(labeling.as_dict()) == labeling


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"path": ""}, "path must be non-empty"),
        ({"sha256": "A" * 64}, "lowercase SHA-256"),
        ({"source_sha256": "B" * 64}, "source_sha256"),
        ({"key": ""}, "key must be non-empty"),
        ({"index_stride": 0}, "index_stride must be positive"),
        ({"index_stride": 2, "index_offset": 2}, "0 <= offset"),
        ({"expected_size": 0}, "expected_size must be positive"),
        ({"expected_per_class": 0}, "expected_per_class must be positive"),
        ({"unknown": True}, "Unknown keys"),
    ],
)
def test_fixed_indices_artifact_rejects_invalid_fields(change, message) -> None:
    raw = {
        "path": "/immutable/permutations.npz",
        "sha256": "a" * 64,
        "source_sha256": "b" * 64,
    }
    raw.update(change)
    with pytest.raises(ValueError, match=message):
        FixedIndicesArtifactSpec.from_dict(raw)


def test_labeling_rejects_two_fixed_index_sources() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        LabelingSpec.from_dict(
            {
                "fixed_indices": [1],
                "fixed_indices_artifact": {
                    "path": "/immutable/permutations.npz",
                    "sha256": "a" * 64,
                    "source_sha256": "b" * 64,
                },
            }
        )

    with pytest.raises(ValueError, match="must be a mapping"):
        LabelingSpec.from_dict({"fixed_indices_artifact": []})

    with pytest.raises(ValueError, match="mutually exclusive"):
        LabelingSpec.from_dict(
            {
                "mode": "count",
                "value": 1,
                "fixed_indices": [1],
                "class_counts": {"0": 1},
            }
        )


def test_labeling_rejects_invalid_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown labeling strategy"):
        LabelingSpec.from_dict({"strategy": "nope"})


def test_labeling_random_strategy_round_trip() -> None:
    spec = LabelingSpec.from_dict(
        {
            "mode": "count",
            "value": 60,
            "strategy": "random",
            "min_per_class": 0,
        }
    )

    assert spec.strategy == "random"
    assert LabelingSpec.from_dict(spec.as_dict()) == spec


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


def test_labeling_rejects_unknown_selection_order() -> None:
    with pytest.raises(ValueError, match="selection_order"):
        LabelingSpec.from_dict({"selection_order": "unstable"})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rng_backend", "global"),
        ("selection_scope", "test"),
        ("unlabeled_pool", "unknown"),
    ],
)
def test_labeling_rejects_invalid_native_protocol_controls(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        LabelingSpec.from_dict({field: value})


def test_partition_round_trip_and_validation() -> None:
    spec = PartitionSpec.from_dict({"max_samples": 3442, "shuffle": True})

    assert spec == PartitionSpec(max_samples=3442, shuffle=True)
    assert PartitionSpec.from_dict(spec.as_dict()) == spec
    with pytest.raises(ValueError, match="positive integer"):
        PartitionSpec.from_dict({"max_samples": 0})
    with pytest.raises(ValueError, match="Unknown keys in partition"):
        PartitionSpec.from_dict({"size": 3442})
    with pytest.raises(ValueError, match="partition.ordering"):
        PartitionSpec.from_dict({"ordering": "unstable"})


def test_ordered_partition_artifact_round_trip() -> None:
    raw = {
        "path": "/immutable/cifar10-splits.npz",
        "sha256": "b" * 64,
        "unlabeled_pool": "includes_labeled",
        "test_ref": "test",
        "expected_train_size": 50000,
        "expected_val_size": 0,
        "expected_test_size": 10000,
        "expected_labeled_size": 250,
        "expected_unlabeled_size": 50000,
        "expected_per_class": 25,
    }

    artifact = OrderedPartitionArtifactSpec.from_dict(raw)
    assert artifact.as_dict() == raw
    partition = PartitionSpec.from_dict({"ordered_indices_artifact": raw})
    assert partition.ordered_indices_artifact == artifact
    assert PartitionSpec.from_dict(partition.as_dict()) == partition


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"path": ""}, "path must be non-empty"),
        ({"sha256": "B" * 64}, "lowercase SHA-256"),
        ({"unlabeled_pool": "unknown"}, "unlabeled_pool"),
        ({"test_ref": "unknown"}, "test_ref"),
        ({"expected_train_size": -1}, "must be non-negative"),
        ({"expected_per_class": 0}, "must be positive"),
        ({"unknown": True}, "Unknown keys"),
    ],
)
def test_ordered_partition_artifact_rejects_invalid_fields(change, message) -> None:
    raw = {
        "path": "/immutable/cifar10-splits.npz",
        "sha256": "b" * 64,
    }
    raw.update(change)
    with pytest.raises(ValueError, match=message):
        OrderedPartitionArtifactSpec.from_dict(raw)


def test_partition_rejects_two_pool_sources() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        PartitionSpec.from_dict(
            {
                "max_samples": 10,
                "ordered_indices_artifact": {
                    "path": "/immutable/cifar10-splits.npz",
                    "sha256": "b" * 64,
                },
            }
        )
    with pytest.raises(ValueError, match="must be a mapping"):
        PartitionSpec.from_dict({"ordered_indices_artifact": []})


def test_sampling_plan_serializes_only_active_partition() -> None:
    assert "partition" not in SamplingPlan().as_dict()

    plan = SamplingPlan.from_dict({"partition": {"max_samples": 3442}})

    assert plan.as_dict()["partition"] == {"max_samples": 3442, "shuffle": True}
    no_shuffle = SamplingPlan(partition=PartitionSpec(shuffle=False))
    assert SamplingPlan.from_dict(no_shuffle.as_dict()) == no_shuffle


def test_sampling_component_seeds_round_trip_and_resolve_overrides() -> None:
    seeds = SamplingComponentSeeds.from_dict({"split": 2005, "labeling": 17})

    assert seeds.as_dict() == {"split": 2005, "labeling": 17}
    resolved = seeds.resolve(9)
    assert resolved["split"] == 2005
    assert resolved["labeling"] == 17
    assert resolved["partition"] != resolved["imbalance"]

    plan = SamplingPlan(component_seeds=seeds)
    assert SamplingPlan.from_dict(plan.as_dict()) == plan
    assert plan.as_dict()["component_seeds"] == {"split": 2005, "labeling": 17}


@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_sampling_component_seeds_reject_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="component_seeds.split"):
        SamplingComponentSeeds.from_dict({"split": value})


def test_sampling_plan_omits_default_component_seeds_for_fingerprint_compatibility() -> None:
    plan = SamplingPlan()

    assert "component_seeds" not in plan.as_dict()
    with pytest.raises(ValueError, match="Unknown keys in component_seeds"):
        SamplingComponentSeeds.from_dict({"test": 1})


def test_historical_public_sampling_constructor_order_is_unchanged() -> None:
    split = KFoldSplitSpec(k=3)
    labeling = LabelingSpec("count", 6, True, 0, "balanced", [1, 2])
    imbalance = ImbalanceSpec("long_tail", "labeled", None, 0.5, 2)
    policy = SamplingPolicy(False, False, True)

    plan = SamplingPlan(split, labeling, imbalance, policy)

    assert (plan.split, plan.labeling, plan.imbalance, plan.policy) == (
        split,
        labeling,
        imbalance,
        policy,
    )
    assert plan.partition == PartitionSpec()
    assert plan.component_seeds == SamplingComponentSeeds()
    with pytest.raises(TypeError):
        SamplingPlan(split, labeling, imbalance, policy, PartitionSpec())  # type: ignore[misc]
    with pytest.raises(TypeError):
        SamplingPolicy(False, False, True, True)  # type: ignore[misc]


def test_schema_v1_default_plan_has_exact_historical_payload_and_hash() -> None:
    plan = SamplingPlan()

    assert plan.fingerprint_schema_version() == 1
    assert plan.fingerprint_payload() == plan.as_dict()
    assert "selection_order" not in plan.as_dict()["labeling"]
    assert "merge_official_splits" not in plan.as_dict()["policy"]
    assert (
        stable_hash(
            {
                "schema_version": 1,
                "dataset_fingerprint": "deadbeef",
                "plan": plan.fingerprint_payload(),
                "seed": 7,
            }
        )
        == "266add85b6b5fecc997e3700c53710eba69db9cd7e6849777c0a5b41070f7832"
    )


def test_schema_v2_artifact_identity_is_content_based_not_path_based() -> None:
    common = {
        "sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "expected_size": 10,
    }
    first = SamplingPlan(
        labeling=LabelingSpec(
            fixed_indices_artifact=FixedIndicesArtifactSpec(path="/machine-a/perm.npz", **common)
        )
    )
    second = SamplingPlan(
        labeling=LabelingSpec(
            fixed_indices_artifact=FixedIndicesArtifactSpec(path="/machine-b/perm.npz", **common)
        )
    )

    assert first.fingerprint_schema_version() == 2
    assert first.as_dict() != second.as_dict()
    assert first.fingerprint_payload() == second.fingerprint_payload()
    artifact_identity = first.fingerprint_payload()["labeling"]["fixed_indices_artifact"]
    assert "path" not in artifact_identity
    assert artifact_identity["sha256"] == "a" * 64
    assert artifact_identity["source_sha256"] == "b" * 64


@pytest.mark.parametrize(
    "plan",
    [
        SamplingPlan(partition=PartitionSpec(max_samples=10)),
        SamplingPlan(component_seeds=SamplingComponentSeeds(split=7)),
        SamplingPlan(policy=SamplingPolicy(merge_official_splits=True)),
        SamplingPlan(split=HoldoutSplitSpec(test_size=10)),
        SamplingPlan(labeling=LabelingSpec(selection_order="permutation")),
        SamplingPlan(
            labeling=LabelingSpec(
                mode="count",
                value=2,
                class_counts={"0": 1, "1": 1},
            )
        ),
        SamplingPlan(labeling=LabelingSpec(rng_backend="legacy_random_state")),
        SamplingPlan(labeling=LabelingSpec(selection_scope="partition")),
        SamplingPlan(labeling=LabelingSpec(unlabeled_pool="includes_labeled")),
        SamplingPlan(labeling=LabelingSpec(strategy="random")),
    ],
)
def test_each_post_v1_sampling_feature_selects_schema_v2(plan: SamplingPlan) -> None:
    assert plan.fingerprint_schema_version() == 2


def test_merge_official_splits_serializes_only_when_enabled() -> None:
    enabled = SamplingPlan(policy=SamplingPolicy(merge_official_splits=True))

    assert enabled.as_dict()["policy"]["merge_official_splits"] is True


def test_ordered_partition_fingerprint_identity_omits_only_location() -> None:
    artifact = OrderedPartitionArtifactSpec(path="/machine/partition.npz", sha256="a" * 64)
    plan = SamplingPlan(partition=PartitionSpec(ordered_indices_artifact=artifact))

    assert plan.as_dict()["partition"]["ordered_indices_artifact"]["path"] == artifact.path
    identity = plan.fingerprint_payload()["partition"]["ordered_indices_artifact"]
    assert identity == {key: value for key, value in artifact.as_dict().items() if key != "path"}


def test_sampling_plan_from_dict_accepts_valid_kfold() -> None:
    plan = SamplingPlan.from_dict({"split": {"kind": "kfold", "k": 3}})

    assert isinstance(plan.split, KFoldSplitSpec)
    assert plan.split.k == 3


def test_fixed_indices_artifact_is_exported_from_sampling_facade() -> None:
    from modssc.sampling import FixedIndicesArtifactSpec as PublicSpec

    assert PublicSpec is FixedIndicesArtifactSpec
