from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

LEGACY_FINGERPRINT_SCHEMA_VERSION = 1
CURRENT_FINGERPRINT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class FixedIndicesArtifactSpec:
    """Immutable external label-index table selected by run seed.

    The selected row is ``run_seed * index_stride + index_offset``. This
    represents paper artifacts that interleave several label budgets for every
    trial, such as Calder et al.'s 100 x 5 MNIST permutations.
    """

    path: str
    sha256: str
    source_sha256: str
    key: str = "perm"
    index_stride: int = 1
    index_offset: int = 0
    expected_size: int | None = None
    expected_per_class: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "source_sha256": self.source_sha256,
            "key": self.key,
            "index_stride": int(self.index_stride),
            "index_offset": int(self.index_offset),
            "expected_size": (None if self.expected_size is None else int(self.expected_size)),
            "expected_per_class": (
                None if self.expected_per_class is None else int(self.expected_per_class)
            ),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FixedIndicesArtifactSpec:
        _assert_known_keys(
            d,
            {
                "path",
                "sha256",
                "source_sha256",
                "key",
                "index_stride",
                "index_offset",
                "expected_size",
                "expected_per_class",
            },
            "labeling.fixed_indices_artifact",
        )
        path = d.get("path")
        sha256 = d.get("sha256")
        source_sha256 = d.get("source_sha256")
        key = d.get("key", "perm")
        if not isinstance(path, str) or not path:
            raise ValueError("labeling.fixed_indices_artifact.path must be non-empty")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(
                "labeling.fixed_indices_artifact.sha256 must be a lowercase SHA-256 digest"
            )
        if (
            not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_sha256)
        ):
            raise ValueError(
                "labeling.fixed_indices_artifact.source_sha256 must be a lowercase SHA-256 digest"
            )
        if not isinstance(key, str) or not key:
            raise ValueError("labeling.fixed_indices_artifact.key must be non-empty")
        stride = int(d.get("index_stride", 1))
        offset = int(d.get("index_offset", 0))
        expected_size = d.get("expected_size")
        expected_per_class = d.get("expected_per_class")
        if stride <= 0:
            raise ValueError("labeling.fixed_indices_artifact.index_stride must be positive")
        if offset < 0 or offset >= stride:
            raise ValueError(
                "labeling.fixed_indices_artifact.index_offset must satisfy "
                "0 <= offset < index_stride"
            )
        if expected_size is not None and int(expected_size) <= 0:
            raise ValueError("labeling.fixed_indices_artifact.expected_size must be positive")
        if expected_per_class is not None and int(expected_per_class) <= 0:
            raise ValueError("labeling.fixed_indices_artifact.expected_per_class must be positive")
        return cls(
            path=path,
            sha256=sha256,
            source_sha256=source_sha256,
            key=key,
            index_stride=stride,
            index_offset=offset,
            expected_size=None if expected_size is None else int(expected_size),
            expected_per_class=(None if expected_per_class is None else int(expected_per_class)),
        )


@dataclass(frozen=True)
class OrderedPartitionArtifactSpec:
    """Authenticated, ordered train/validation/test and SSL pools.

    The artifact is a NumPy archive with five arrays per run seed:
    ``seed_<N>__train``, ``seed_<N>__val``, ``seed_<N>__test``,
    ``seed_<N>__train_labeled`` and ``seed_<N>__train_unlabeled``.
    Array order is significant and is preserved by sampling and replay.

    Some paper implementations deliberately include labeled examples in the
    unlabeled loader. ``unlabeled_pool='includes_labeled'`` records that
    protocol explicitly instead of weakening the default disjoint-partition
    invariant used by standardized ModSSC experiments.
    """

    path: str
    sha256: str
    unlabeled_pool: Literal["complement", "includes_labeled"] = "complement"
    test_ref: Literal["train", "test"] = "test"
    expected_train_size: int | None = None
    expected_val_size: int | None = None
    expected_test_size: int | None = None
    expected_labeled_size: int | None = None
    expected_unlabeled_size: int | None = None
    expected_per_class: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "unlabeled_pool": self.unlabeled_pool,
            "test_ref": self.test_ref,
            "expected_train_size": self.expected_train_size,
            "expected_val_size": self.expected_val_size,
            "expected_test_size": self.expected_test_size,
            "expected_labeled_size": self.expected_labeled_size,
            "expected_unlabeled_size": self.expected_unlabeled_size,
            "expected_per_class": self.expected_per_class,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> OrderedPartitionArtifactSpec:
        size_fields = {
            "expected_train_size",
            "expected_val_size",
            "expected_test_size",
            "expected_labeled_size",
            "expected_unlabeled_size",
            "expected_per_class",
        }
        _assert_known_keys(
            d,
            {"path", "sha256", "unlabeled_pool", "test_ref", *size_fields},
            "partition.ordered_indices_artifact",
        )
        path = d.get("path")
        sha256 = d.get("sha256")
        if not isinstance(path, str) or not path:
            raise ValueError("partition.ordered_indices_artifact.path must be non-empty")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(
                "partition.ordered_indices_artifact.sha256 must be a lowercase SHA-256 digest"
            )
        unlabeled_pool = str(d.get("unlabeled_pool", "complement"))
        if unlabeled_pool not in ("complement", "includes_labeled"):
            raise ValueError(
                "partition.ordered_indices_artifact.unlabeled_pool must be "
                "'complement' or 'includes_labeled'"
            )
        test_ref = str(d.get("test_ref", "test"))
        if test_ref not in ("train", "test"):
            raise ValueError(
                "partition.ordered_indices_artifact.test_ref must be 'train' or 'test'"
            )
        sizes: dict[str, int | None] = {}
        for name in size_fields:
            value = d.get(name)
            if value is None:
                sizes[name] = None
                continue
            parsed = int(value)
            if name == "expected_per_class":
                if parsed <= 0:
                    raise ValueError(
                        "partition.ordered_indices_artifact.expected_per_class must be positive"
                    )
            elif parsed < 0:
                raise ValueError(f"partition.ordered_indices_artifact.{name} must be non-negative")
            sizes[name] = parsed
        return cls(
            path=path,
            sha256=sha256,
            unlabeled_pool=unlabeled_pool,  # type: ignore[arg-type]
            test_ref=test_ref,  # type: ignore[arg-type]
            **sizes,
        )


@dataclass(frozen=True)
class SamplingPolicy:
    """Policy for handling official provider splits.

    - respect_official_test: if dataset.test exists, keep it as the test set
    - use_official_graph_masks: if graph dataset provides masks, use them as train/val/test masks
    - allow_override_official: if True, user-defined split parameters take precedence and
      inductive sampling ignores provider test partitions instead of erroring
    """

    respect_official_test: bool = True
    use_official_graph_masks: bool = True
    allow_override_official: bool = False
    merge_official_splits: bool = field(default=False, kw_only=True)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SamplingPolicy:
        _assert_known_keys(
            d,
            {
                "respect_official_test",
                "use_official_graph_masks",
                "allow_override_official",
                "merge_official_splits",
            },
            "policy",
        )
        return cls(
            respect_official_test=bool(d.get("respect_official_test", True)),
            use_official_graph_masks=bool(d.get("use_official_graph_masks", True)),
            allow_override_official=bool(d.get("allow_override_official", False)),
            merge_official_splits=bool(d.get("merge_official_splits", False)),
        )


@dataclass(frozen=True)
class HoldoutSplitSpec:
    kind: Literal["holdout"] = "holdout"
    test_fraction: float = 0.2
    val_fraction: float = 0.1
    stratify: bool = True
    shuffle: bool = True
    test_size: int | None = field(default=None, kw_only=True)
    val_size: int | None = field(default=None, kw_only=True)
    holdout_from: Literal["start", "end"] = field(default="start", kw_only=True)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "test_fraction": float(self.test_fraction),
            "val_fraction": float(self.val_fraction),
            "stratify": bool(self.stratify),
            "shuffle": bool(self.shuffle),
        }
        if self.test_size is not None:
            result["test_size"] = int(self.test_size)
        if self.val_size is not None:
            result["val_size"] = int(self.val_size)
        if self.holdout_from != "start":
            result["holdout_from"] = self.holdout_from
        return result

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> HoldoutSplitSpec:
        _assert_known_keys(
            d,
            {
                "kind",
                "test_fraction",
                "val_fraction",
                "stratify",
                "shuffle",
                "test_size",
                "val_size",
                "holdout_from",
            },
            "split",
        )
        kind = str(d.get("kind", "holdout"))
        if kind != "holdout":
            raise ValueError(f"Unknown split kind: {kind!r}")
        test_size = d.get("test_size")
        val_size = d.get("val_size")
        for name, value in (("test_size", test_size), ("val_size", val_size)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"split.{name} must be a non-negative integer")
        holdout_from = str(d.get("holdout_from", "start"))
        if holdout_from not in ("start", "end"):
            raise ValueError("split.holdout_from must be 'start' or 'end'")
        return cls(
            test_fraction=float(d.get("test_fraction", 0.2)),
            val_fraction=float(d.get("val_fraction", 0.1)),
            stratify=bool(d.get("stratify", True)),
            shuffle=bool(d.get("shuffle", True)),
            test_size=test_size,
            val_size=val_size,
            holdout_from=holdout_from,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class KFoldSplitSpec:
    kind: Literal["kfold"] = "kfold"
    k: int = 5
    fold: int = 0
    stratify: bool = True
    shuffle: bool = True
    val_fraction: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "k": int(self.k),
            "fold": int(self.fold),
            "stratify": bool(self.stratify),
            "shuffle": bool(self.shuffle),
            "val_fraction": float(self.val_fraction),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> KFoldSplitSpec:
        _assert_known_keys(
            d,
            {"kind", "k", "fold", "stratify", "shuffle", "val_fraction"},
            "split",
        )
        kind = str(d.get("kind", "kfold"))
        if kind != "kfold":
            raise ValueError(f"Unknown split kind: {kind!r}")
        return cls(
            k=int(d.get("k", 5)),
            fold=int(d.get("fold", 0)),
            stratify=bool(d.get("stratify", True)),
            shuffle=bool(d.get("shuffle", True)),
            val_fraction=float(d.get("val_fraction", 0.0)),
        )


SplitSpec = HoldoutSplitSpec | KFoldSplitSpec


@dataclass(frozen=True)
class SamplingComponentSeeds:
    """Optional exact RNG seeds for individual sampling stages.

    Components left unset retain the historical behavior and derive their seed
    from the master sampling seed.  An explicit split seed is useful for paper
    protocols that keep one test set fixed while redrawing labeled and
    unlabeled pools across repetitions.
    """

    partition: int | Literal["run"] | None = None
    split: int | Literal["run"] | None = None
    labeling: int | Literal["run"] | None = None
    imbalance: int | Literal["run"] | None = None

    def as_dict(self) -> dict[str, int | str]:
        values = {
            "partition": self.partition,
            "split": self.split,
            "labeling": self.labeling,
            "imbalance": self.imbalance,
        }
        return {
            key: value if value == "run" else int(value)
            for key, value in values.items()
            if value is not None
        }

    def resolve(self, master_seed: int) -> dict[str, int]:
        from modssc.sampling.fingerprint import derive_seed

        overrides = self.as_dict()
        resolved: dict[str, int] = {}
        for component in ("partition", "split", "labeling", "imbalance"):
            override = overrides.get(component)
            if override == "run":
                resolved[component] = int(master_seed)
            elif override is None:
                resolved[component] = derive_seed(master_seed, component)
            else:
                resolved[component] = int(override)
        return resolved

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SamplingComponentSeeds:
        allowed = {"partition", "split", "labeling", "imbalance"}
        _assert_known_keys(d, allowed, "component_seeds")
        values: dict[str, int | Literal["run"] | None] = {}
        for component in allowed:
            value = d.get(component)
            if value is None:
                values[component] = None
                continue
            if value == "run":
                values[component] = "run"
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"component_seeds.{component} must be a non-negative integer or 'run'"
                )
            values[component] = int(value)
        return cls(**values)


@dataclass(frozen=True)
class PartitionSpec:
    """Optional row-level pool selection before train/validation/test splitting.

    ``max_samples`` limits the number of rows considered by the experimental
    partition while keeping the canonical dataset and its fingerprint intact.
    Selected indices always remain relative to the canonical dataset.
    """

    max_samples: int | None = None
    shuffle: bool = True
    ordering: Literal["canonical", "class_balanced_stream"] = "canonical"
    ordered_indices_artifact: OrderedPartitionArtifactSpec | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "max_samples": None if self.max_samples is None else int(self.max_samples),
            "shuffle": bool(self.shuffle),
        }
        if self.ordering != "canonical":
            result["ordering"] = self.ordering
        if self.ordered_indices_artifact is not None:
            result["ordered_indices_artifact"] = self.ordered_indices_artifact.as_dict()
        return result

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PartitionSpec:
        _assert_known_keys(
            d,
            {"max_samples", "shuffle", "ordering", "ordered_indices_artifact"},
            "partition",
        )
        raw_max_samples = d.get("max_samples")
        max_samples = None if raw_max_samples is None else int(raw_max_samples)
        if max_samples is not None and max_samples <= 0:
            raise ValueError("partition.max_samples must be a positive integer")
        ordering = str(d.get("ordering", "canonical"))
        if ordering not in ("canonical", "class_balanced_stream"):
            raise ValueError("partition.ordering must be 'canonical' or 'class_balanced_stream'")
        artifact_raw = d.get("ordered_indices_artifact")
        if artifact_raw is None:
            artifact = None
        else:
            artifact = OrderedPartitionArtifactSpec.from_dict(
                _ensure_mapping(
                    artifact_raw,
                    "partition.ordered_indices_artifact",
                )
            )
        if max_samples is not None and artifact is not None:
            raise ValueError(
                "partition.max_samples and partition.ordered_indices_artifact "
                "are mutually exclusive"
            )
        return cls(
            max_samples=max_samples,
            shuffle=bool(d.get("shuffle", True)),
            ordering=ordering,  # type: ignore[arg-type]
            ordered_indices_artifact=artifact,
        )


@dataclass(frozen=True)
class LabelingSpec:
    """How to select labeled samples within the train partition.

    Modes:
    - fraction: value in (0, 1], selects that fraction of train samples
    - count: value is an integer count of labeled samples
    - per_class: value is an integer count per class

    Strategies:
    - proportional: allocate the target proportionally across classes
    - balanced: allocate the target as evenly as possible across classes
    - random: sample uniformly from the full train partition, without class allocation

    If fixed_indices is provided, it is used directly (validated) and the mode is ignored.
    If class_counts is provided, it freezes an exact labeled quota for every observed class.
    """

    mode: Literal["fraction", "count", "per_class"] = "fraction"
    value: float | int = 0.1
    per_class: bool = False
    min_per_class: int = 1
    strategy: Literal["proportional", "balanced", "random"] = "proportional"
    fixed_indices: Sequence[int] | None = None
    fixed_indices_artifact: FixedIndicesArtifactSpec | None = field(default=None, kw_only=True)
    selection_order: Literal["choice", "permutation"] = field(default="choice", kw_only=True)
    class_counts: Mapping[str, int] | None = field(default=None, kw_only=True)
    rng_backend: Literal["generator", "legacy_random_state"] = field(
        default="generator", kw_only=True
    )
    selection_scope: Literal["train", "partition"] = field(default="train", kw_only=True)
    unlabeled_pool: Literal["complement", "includes_labeled"] = field(
        default="complement", kw_only=True
    )

    def as_dict(self) -> dict[str, Any]:
        result = {
            "mode": self.mode,
            "value": float(self.value) if self.mode == "fraction" else int(self.value),
            "per_class": bool(self.per_class),
            "min_per_class": int(self.min_per_class),
            "strategy": self.strategy,
            "fixed_indices": None
            if self.fixed_indices is None
            else [int(i) for i in self.fixed_indices],
        }
        # Default values introduced after schema v1 are omitted so historical
        # plans retain their exact serialized representation.
        if self.fixed_indices_artifact is not None:
            result["fixed_indices_artifact"] = self.fixed_indices_artifact.as_dict()
        if self.selection_order != "choice":
            result["selection_order"] = self.selection_order
        if self.rng_backend != "generator":
            result["rng_backend"] = self.rng_backend
        if self.selection_scope != "train":
            result["selection_scope"] = self.selection_scope
        if self.unlabeled_pool != "complement":
            result["unlabeled_pool"] = self.unlabeled_pool
        if self.class_counts is not None:
            result["class_counts"] = {
                str(label): int(count)
                for label, count in sorted(self.class_counts.items(), key=lambda item: str(item[0]))
            }
        return result

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LabelingSpec:
        _assert_known_keys(
            d,
            {
                "mode",
                "value",
                "per_class",
                "min_per_class",
                "strategy",
                "fixed_indices",
                "fixed_indices_artifact",
                "class_counts",
                "selection_order",
                "rng_backend",
                "selection_scope",
                "unlabeled_pool",
            },
            "labeling",
        )
        mode = str(d.get("mode", "fraction"))
        if mode not in ("fraction", "count", "per_class"):
            raise ValueError(f"Unknown labeling mode: {mode!r}")
        value = d.get("value", 0.1)
        value = float(value) if mode == "fraction" else int(value)
        fixed_indices = d.get("fixed_indices", None)
        if fixed_indices is not None:
            if isinstance(fixed_indices, (str, bytes)) or not isinstance(fixed_indices, Sequence):
                raise ValueError("labeling.fixed_indices must be a sequence of integers")
            fixed_indices = [int(i) for i in fixed_indices]
        artifact_raw = d.get("fixed_indices_artifact")
        class_counts_raw = d.get("class_counts")
        configured_sources = sum(
            source is not None for source in (fixed_indices, artifact_raw, class_counts_raw)
        )
        if configured_sources > 1:
            raise ValueError(
                "labeling.fixed_indices, labeling.fixed_indices_artifact, and "
                "labeling.class_counts are mutually exclusive"
            )
        if artifact_raw is None:
            artifact = None
        else:
            artifact_obj = _ensure_mapping(
                artifact_raw,
                "labeling.fixed_indices_artifact",
            )
            artifact = FixedIndicesArtifactSpec.from_dict(artifact_obj)
        if class_counts_raw is None:
            class_counts = None
        else:
            class_counts_obj = _ensure_mapping(class_counts_raw, "labeling.class_counts")
            if not class_counts_obj:
                raise ValueError("labeling.class_counts must not be empty")
            class_counts = {}
            for raw_label, raw_count in class_counts_obj.items():
                label = str(raw_label)
                if label in class_counts:
                    raise ValueError(
                        "labeling.class_counts contains duplicate labels after normalization"
                    )
                if isinstance(raw_count, bool) or not isinstance(raw_count, int):
                    raise ValueError("labeling.class_counts values must be non-negative integers")
                count = int(raw_count)
                if count < 0:
                    raise ValueError("labeling.class_counts values must be non-negative integers")
                class_counts[label] = count
            if not any(count > 0 for count in class_counts.values()):
                raise ValueError("labeling.class_counts must request at least one labeled sample")
            if mode != "count" or int(value) != sum(class_counts.values()):
                raise ValueError(
                    "labeling.class_counts requires mode='count' and value equal to its total"
                )
        strategy = str(d.get("strategy", "proportional"))
        if strategy not in ("proportional", "balanced", "random"):
            raise ValueError(f"Unknown labeling strategy: {strategy!r}")
        selection_order = str(d.get("selection_order", "choice"))
        if selection_order not in ("choice", "permutation"):
            raise ValueError(f"Unknown labeling selection_order: {selection_order!r}")
        rng_backend = str(d.get("rng_backend", "generator"))
        if rng_backend not in ("generator", "legacy_random_state"):
            raise ValueError("labeling.rng_backend must be 'generator' or 'legacy_random_state'")
        selection_scope = str(d.get("selection_scope", "train"))
        if selection_scope not in ("train", "partition"):
            raise ValueError("labeling.selection_scope must be 'train' or 'partition'")
        unlabeled_pool = str(d.get("unlabeled_pool", "complement"))
        if unlabeled_pool not in ("complement", "includes_labeled"):
            raise ValueError("labeling.unlabeled_pool must be 'complement' or 'includes_labeled'")
        return cls(
            mode=mode,  # type: ignore[arg-type]
            value=value,
            per_class=bool(d.get("per_class", False)),
            min_per_class=int(d.get("min_per_class", 1)),
            strategy=strategy,  # type: ignore[arg-type]
            fixed_indices=fixed_indices,
            fixed_indices_artifact=artifact,
            class_counts=class_counts,
            selection_order=selection_order,  # type: ignore[arg-type]
            rng_backend=rng_backend,  # type: ignore[arg-type]
            selection_scope=selection_scope,  # type: ignore[arg-type]
            unlabeled_pool=unlabeled_pool,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ImbalanceSpec:
    """Optional class imbalance scenario.

    Kinds:
    - none
    - subsample_max_per_class: cap each class to max_per_class (applies to train or labeled)
    - long_tail: exponential decay per class rank (applies to train or labeled)

    apply_to:
    - train: modify train_idx before labeling
    - labeled: modify labeled subset after labeling (removed labeled become unlabeled)
    """

    kind: Literal["none", "subsample_max_per_class", "long_tail"] = "none"
    apply_to: Literal["train", "labeled"] = "train"
    max_per_class: int | None = None
    alpha: float | None = None
    min_per_class: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "apply_to": self.apply_to,
            "max_per_class": None if self.max_per_class is None else int(self.max_per_class),
            "alpha": None if self.alpha is None else float(self.alpha),
            "min_per_class": int(self.min_per_class),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ImbalanceSpec:
        _assert_known_keys(
            d,
            {"kind", "apply_to", "max_per_class", "alpha", "min_per_class"},
            "imbalance",
        )
        kind = str(d.get("kind", "none"))
        if kind not in ("none", "subsample_max_per_class", "long_tail"):
            raise ValueError(f"Unknown imbalance kind: {kind!r}")
        apply_to = str(d.get("apply_to", "train"))
        if apply_to not in ("train", "labeled"):
            raise ValueError(f"Unknown imbalance apply_to: {apply_to!r}")
        return cls(
            kind=kind,  # type: ignore[arg-type]
            apply_to=apply_to,  # type: ignore[arg-type]
            max_per_class=d.get("max_per_class", None),
            alpha=d.get("alpha", None),
            min_per_class=int(d.get("min_per_class", 1)),
        )


@dataclass(frozen=True)
class SamplingPlan:
    """Full sampling plan."""

    split: SplitSpec = field(default_factory=HoldoutSplitSpec)
    labeling: LabelingSpec = field(default_factory=LabelingSpec)
    imbalance: ImbalanceSpec = field(default_factory=ImbalanceSpec)
    policy: SamplingPolicy = field(default_factory=SamplingPolicy)
    partition: PartitionSpec = field(default_factory=PartitionSpec, kw_only=True)
    component_seeds: SamplingComponentSeeds = field(
        default_factory=SamplingComponentSeeds,
        kw_only=True,
    )

    def as_dict(self) -> dict[str, Any]:
        result = {
            "split": self.split.as_dict(),
            "labeling": self.labeling.as_dict(),
            "imbalance": self.imbalance.as_dict(),
            "policy": {
                "respect_official_test": bool(self.policy.respect_official_test),
                "use_official_graph_masks": bool(self.policy.use_official_graph_masks),
                "allow_override_official": bool(self.policy.allow_override_official),
            },
        }
        if self.policy.merge_official_splits:
            result["policy"]["merge_official_splits"] = True
        if self.partition != PartitionSpec():
            result["partition"] = self.partition.as_dict()
        if self.component_seeds != SamplingComponentSeeds():
            result["component_seeds"] = self.component_seeds.as_dict()
        return result

    def fingerprint_schema_version(self) -> int:
        """Return the sampling identity schema required by this plan.

        Schema 1 is the exact public behavior shipped before artifact-backed
        partitions, independently seeded components, and ordered selection.
        Activating any of those features opts the plan into schema 2.
        """

        if self.partition != PartitionSpec():
            return CURRENT_FINGERPRINT_SCHEMA_VERSION
        if self.component_seeds != SamplingComponentSeeds():
            return CURRENT_FINGERPRINT_SCHEMA_VERSION
        if self.policy.merge_official_splits:
            return CURRENT_FINGERPRINT_SCHEMA_VERSION
        if (
            self.split != HoldoutSplitSpec()
            and isinstance(self.split, HoldoutSplitSpec)
            and (
                self.split.test_size is not None
                or self.split.val_size is not None
                or self.split.holdout_from != "start"
            )
        ):
            return CURRENT_FINGERPRINT_SCHEMA_VERSION
        if self.labeling.fixed_indices_artifact is not None:
            return CURRENT_FINGERPRINT_SCHEMA_VERSION
        if self.labeling.selection_order != "choice":
            return CURRENT_FINGERPRINT_SCHEMA_VERSION
        if self.labeling.class_counts is not None:
            return CURRENT_FINGERPRINT_SCHEMA_VERSION
        if self.labeling.rng_backend != "generator":
            return CURRENT_FINGERPRINT_SCHEMA_VERSION
        if self.labeling.selection_scope != "train":
            return CURRENT_FINGERPRINT_SCHEMA_VERSION
        if self.labeling.unlabeled_pool != "complement":
            return CURRENT_FINGERPRINT_SCHEMA_VERSION
        if self.labeling.strategy == "random":
            return CURRENT_FINGERPRINT_SCHEMA_VERSION
        return LEGACY_FINGERPRINT_SCHEMA_VERSION

    def fingerprint_payload(self) -> dict[str, Any]:
        """Return a location-independent payload for split identity.

        Runtime serialization keeps artifact paths so a plan can be replayed.
        Identity is instead bound to authenticated file contents, allowing the
        same immutable artifact to be moved between machines without changing
        the scientific split fingerprint.
        """

        payload = self.as_dict()
        labeling = payload["labeling"]
        _replace_artifact_path_with_content_identity(
            labeling.get("fixed_indices_artifact"),
        )
        partition = payload.get("partition")
        if isinstance(partition, dict):
            _replace_artifact_path_with_content_identity(
                partition.get("ordered_indices_artifact"),
            )
        return payload

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SamplingPlan:
        _assert_known_keys(
            d,
            {"component_seeds", "partition", "split", "labeling", "imbalance", "policy"},
            "plan",
        )
        component_seeds_obj = _ensure_mapping(d.get("component_seeds", {}), "component_seeds")
        component_seeds = SamplingComponentSeeds.from_dict(component_seeds_obj)

        partition_obj = _ensure_mapping(d.get("partition", {}), "partition")
        partition = PartitionSpec.from_dict(partition_obj)

        split_obj = _ensure_mapping(d.get("split", {}), "split")
        split_kind = str(split_obj.get("kind", "holdout"))
        if split_kind == "kfold":
            split = KFoldSplitSpec.from_dict(split_obj)
        elif split_kind == "holdout":
            split = HoldoutSplitSpec.from_dict(split_obj)
        else:
            raise ValueError(f"Unknown split kind: {split_kind!r}")

        labeling_obj = _ensure_mapping(d.get("labeling", {}), "labeling")
        labeling = LabelingSpec.from_dict(labeling_obj)

        imbalance_obj = _ensure_mapping(d.get("imbalance", {}), "imbalance")
        imbalance = ImbalanceSpec.from_dict(imbalance_obj)

        policy_obj = _ensure_mapping(d.get("policy", {}), "policy")
        policy = SamplingPolicy.from_dict(policy_obj)

        return cls(
            component_seeds=component_seeds,
            partition=partition,
            split=split,
            labeling=labeling,
            imbalance=imbalance,
            policy=policy,
        )


def _ensure_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _assert_known_keys(d: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(d.keys()) - allowed
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown keys in {name}: {keys}")


def _replace_artifact_path_with_content_identity(value: Any) -> None:
    if isinstance(value, dict) and "path" in value:
        value.pop("path")
