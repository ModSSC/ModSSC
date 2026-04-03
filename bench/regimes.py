from __future__ import annotations

REGIME_LABELS_PER_CLASS: dict[str, int] = {
    "R1": 1,
    "R2": 3,
    "R3": 5,
    "R4": 10,
    "R5": 20,
    "R6": 50,
}

REGIME_ORDER: tuple[str, ...] = tuple(REGIME_LABELS_PER_CLASS)

# Article-facing requested regimes before dataset-specific eligibility filtering.
DATASET_REQUESTED_REGIMES: dict[str, tuple[str, ...]] = {
    "ag_news": ("R1", "R2", "R3", "R4", "R5", "R6"),
    "adult": ("R1", "R2", "R3", "R4", "R5", "R6"),
    "amazon_polarity": ("R1", "R2", "R3", "R4", "R5", "R6"),
    "amazon_reviews_multi_en": ("R1", "R2", "R3", "R4", "R5", "R6"),
    "breast_cancer": ("R1", "R2", "R3", "R4", "R5"),
    "cifar10": ("R1", "R2", "R3", "R4", "R5", "R6"),
    "cifar100": ("R1", "R2", "R3", "R4", "R5"),
    "citeseer": ("R1", "R2", "R3", "R4", "R5"),
    "cora": ("R1", "R2", "R3", "R4", "R5"),
    "dbpedia_14": ("R1", "R2", "R3", "R4", "R5", "R6"),
    "imdb": ("R1", "R2", "R3", "R4", "R5", "R6"),
    "iris": ("R1", "R2", "R3", "R4"),
    "mnist": ("R1", "R2", "R3", "R4", "R5", "R6"),
    "pubmed": ("R1", "R2", "R3", "R4", "R5"),
    "speechcommands": ("R1", "R2", "R3", "R4", "R5", "R6"),
    "stl10": ("R1", "R2", "R3", "R4", "R5"),
    "svhn": ("R1", "R2", "R3", "R4", "R5", "R6"),
    "toy": ("R1", "R2", "R3", "R4", "R5"),
    "yesno": ("R1", "R2", "R3", "R4", "R5"),
    "yelp_polarity": ("R1", "R2", "R3", "R4", "R5", "R6"),
    "yelp_review_full": ("R1", "R2", "R3", "R4", "R5"),
}

SLURM_SPLIT_BY_REGIME_DATASETS: frozenset[str] = frozenset({"imdb", "mnist"})


def labels_per_class(regime: str) -> int:
    try:
        return int(REGIME_LABELS_PER_CLASS[regime])
    except KeyError as exc:
        raise KeyError(f"Unknown article regime: {regime!r}") from exc


def is_regime(value: str) -> bool:
    return value in REGIME_LABELS_PER_CLASS


def requested_regimes_for_dataset(dataset: str) -> tuple[str, ...]:
    try:
        return DATASET_REQUESTED_REGIMES[dataset]
    except KeyError as exc:
        raise KeyError(f"No requested article regimes configured for dataset {dataset!r}") from exc
