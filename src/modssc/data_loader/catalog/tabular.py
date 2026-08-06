from __future__ import annotations

from modssc.data_loader.types import DatasetSpec

# Notes on reproducibility:
# Prefer OpenML data_id over name to avoid ambiguity when multiple "active" versions exist.

TABULAR_CATALOG: dict[str, DatasetSpec] = {
    "iris": DatasetSpec(
        key="iris",
        provider="openml",
        uri="openml:61",
        modality="tabular",
        task="classification",
        description="Iris (OpenML data_id=61). No official split.",
        required_extra="openml",
        source_kwargs={"data_id": 61},
    ),
    "adult": DatasetSpec(
        key="adult",
        provider="openml",
        uri="openml:1590",
        modality="tabular",
        task="classification",
        description="Adult (OpenML data_id=1590). No official split.",
        required_extra="openml",
        source_kwargs={"data_id": 1590},
    ),
    "breast_cancer": DatasetSpec(
        key="breast_cancer",
        provider="openml",
        uri="openml:15",
        modality="tabular",
        task="classification",
        description="Breast Cancer Wisconsin (OpenML data_id=15). Binary, numeric features.",
        required_extra="openml",
        source_kwargs={"data_id": 15},
    ),
    "wdbc": DatasetSpec(
        key="wdbc",
        provider="openml",
        uri="openml:1510",
        modality="tabular",
        task="classification",
        description=(
            "Wisconsin Diagnostic Breast Cancer (WDBC; OpenML data_id=1510). "
            "569 rows, 30 real-valued features, binary diagnosis target, no official split."
        ),
        required_extra="openml",
        source_kwargs={"data_id": 1510},
        homepage="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        license="CC BY 4.0",
        citation=(
            "Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). "
            "Breast Cancer Wisconsin (Diagnostic) [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B."
        ),
    ),
    "vote": DatasetSpec(
        key="vote",
        provider="openml",
        uri="openml:56",
        modality="tabular",
        task="classification",
        description=(
            "Congressional Voting Records (OpenML data_id=56; UCI id=105). "
            "435 rows, 16 nominal yes/no attributes with missing values, two classes, "
            "and no official split."
        ),
        required_extra="openml",
        source_kwargs={"data_id": 56},
        homepage=("https://archive.ics.uci.edu/dataset/105/congressional+voting+records"),
        license="CC BY 4.0",
        citation=(
            "Congressional Voting Records [Dataset]. (1987). "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5C01P."
        ),
    ),
    "wine": DatasetSpec(
        key="wine",
        provider="openml",
        uri="openml:187",
        modality="tabular",
        task="classification",
        description=(
            "UCI Wine (OpenML data_id=187). 178 rows, 13 continuous features, "
            "three classes, no missing values, and no official split."
        ),
        required_extra="openml",
        source_kwargs={"data_id": 187},
        homepage="https://archive.ics.uci.edu/dataset/109/wine",
        license="CC BY 4.0",
        citation=(
            "Aeberhard, S., & Forina, M. (1992). Wine [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J."
        ),
    ),
}
