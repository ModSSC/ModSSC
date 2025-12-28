from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class SamplingPolicy:
    """Policy for handling official provider splits.

    - respect_official_test: if dataset.test exists, keep it as the test set
    - use_official_graph_masks: if graph dataset provides masks, use them as train/val/test masks
    """

    respect_official_test: bool = True
    use_official_graph_masks: bool = True
    allow_override_official: bool = False

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SamplingPolicy:
        _assert_known_keys(
            d,
            {"respect_official_test", "use_official_graph_masks", "allow_override_official"},
            "policy",
        )
        return cls(
            respect_official_test=bool(d.get("respect_official_test", True)),
            use_official_graph_masks=bool(d.get("use_official_graph_masks", True)),
            allow_override_official=bool(d.get("allow_override_official", False)),
        )


@dataclass(frozen=True)
class HoldoutSplitSpec:
    kind: Literal["holdout"] = "holdout"
    test_fraction: float = 0.2
    val_fraction: float = 0.1
    stratify: bool = True
    shuffle: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "test_fraction": float(self.test_fraction),
            "val_fraction": float(self.val_fraction),
            "stratify": bool(self.stratify),
            "shuffle": bool(self.shuffle),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> HoldoutSplitSpec:
        _assert_known_keys(
            d, {"kind", "test_fraction", "val_fraction", "stratify", "shuffle"}, "split"
        )
        kind = str(d.get("kind", "holdout"))
        if kind != "holdout":
            raise ValueError(f"Unknown split kind: {kind!r}")
        return cls(
            test_fraction=float(d.get("test_fraction", 0.2)),
            val_fraction=float(d.get("val_fraction", 0.1)),
            stratify=bool(d.get("stratify", True)),
            shuffle=bool(d.get("shuffle", True)),
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
class LabelingSpec:
    """How to select labeled samples within the train partition.

    Modes:
    - fraction: value in (0, 1], selects that fraction of train samples
    - count: value is an integer count of labeled samples
    - per_class: value is an integer count per class

    If fixed_indices is provided, it is used directly (validated) and the mode is ignored.
    """

    mode: Literal["fraction", "count", "per_class"] = "fraction"
    value: float | int = 0.1
    per_class: bool = False
    min_per_class: int = 1
    strategy: Literal["proportional", "balanced"] = "proportional"
    fixed_indices: Sequence[int] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "value": float(self.value) if self.mode == "fraction" else int(self.value),
            "per_class": bool(self.per_class),
            "min_per_class": int(self.min_per_class),
            "strategy": self.strategy,
            "fixed_indices": None
            if self.fixed_indices is None
            else [int(i) for i in self.fixed_indices],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LabelingSpec:
        _assert_known_keys(
            d,
            {"mode", "value", "per_class", "min_per_class", "strategy", "fixed_indices"},
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
        strategy = str(d.get("strategy", "proportional"))
        if strategy not in ("proportional", "balanced"):
            raise ValueError(f"Unknown labeling strategy: {strategy!r}")
        return cls(
            mode=mode,  # type: ignore[arg-type]
            value=value,
            per_class=bool(d.get("per_class", False)),
            min_per_class=int(d.get("min_per_class", 1)),
            strategy=strategy,  # type: ignore[arg-type]
            fixed_indices=fixed_indices,
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split.as_dict(),
            "labeling": self.labeling.as_dict(),
            "imbalance": self.imbalance.as_dict(),
            "policy": {
                "respect_official_test": bool(self.policy.respect_official_test),
                "use_official_graph_masks": bool(self.policy.use_official_graph_masks),
                "allow_override_official": bool(self.policy.allow_override_official),
            },
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SamplingPlan:
        _assert_known_keys(d, {"split", "labeling", "imbalance", "policy"}, "plan")
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

        return cls(split=split, labeling=labeling, imbalance=imbalance, policy=policy)


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
