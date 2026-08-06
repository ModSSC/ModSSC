from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

OFFICIAL_REPOSITORY = "https://github.com/jwcalder/GraphLearningOld"
OFFICIAL_COMMIT = "04bece45cd512cf1a3bcddb163b767ca44a746e1"
OFFICIAL_KNN_SHA256 = "5b42bb234888c83eed763958a17fdfb8a55c09a2f0071b55a61635d86dc90db5"
# These two archive digests are provenance only. The archives themselves are
# deliberately not distributed or opened by ModSSC.
OFFICIAL_PERMUTATIONS_SHA256 = "4d2f9949f4ce20d2644cb4c070766421751070dc625c05a0219b1c9d60045770"
OFFICIAL_LABELS_SHA256 = "ec01dca8550a4bf9a4c8559c5c9c1c3ed5b8dd4fb9ab2e771883b03c8635ab2e"
OFFICIAL_SOURCE_SHA256 = "e2d16b74ac7d9ba3daab1c2d020e97b268e26bc378fba1f1077bbfd8707a3372"

PERMUTATIONS_ARTIFACT_SHA256 = "8740039403c6e287e24f0cb0a9013011c9ffc552dedc06ae6bd2ab00b3af1fb3"
MNIST_LABELS_CONTENT_SHA256 = "818800b46032126b329f9306cb69a6842cc53ea30318374769bb6f46cc861467"
OFFICIAL_RESULTS_SHA256 = {
    "laplace_learning": "894e3b33ae18bf0e43c5413dfe72b0e150f9a40d95027bba145fd309bf429b6b",
    "poisson_learning": "a20e0bc231fa0a05a8b1dc341d42b387e8b7129da63df28ecbf7e5f733be4374",
}
PROTOCOL_FILES = {
    "README.md": {
        "size_bytes": 831,
        "sha256": "2e9480e4ff7918796cc0bb6015d309236f91e1f63358bf59cb833095ce372195",
    },
    "LICENSE.graphlearning-mit.txt": {
        "size_bytes": 1_065,
        "sha256": "8e74f73130eb4067387bdf8b62f76cd7de137eaae1cc12117a33a332073580ea",
    },
    "graph/mnist-vae-knn30.npz": {
        "size_bytes": 14_177_339,
        "sha256": OFFICIAL_KNN_SHA256,
    },
    "references/mnist-vae-k10-laplace-accuracy.csv": {
        "size_bytes": 4_498,
        "sha256": OFFICIAL_RESULTS_SHA256["laplace_learning"],
    },
    "references/mnist-vae-k10-poisson-accuracy.csv": {
        "size_bytes": 4_528,
        "sha256": OFFICIAL_RESULTS_SHA256["poisson_learning"],
    },
    "splits/mnist-table1-permutations.ragged-int64-v1.npz": {
        "size_bytes": 44_827,
        "sha256": PERMUTATIONS_ARTIFACT_SHA256,
    },
}
TABLE1_TARGETS = {
    "laplace_learning": {
        1: (16.1, 6.2),
        2: (28.2, 10.3),
        3: (42.0, 12.4),
        4: (57.8, 12.3),
        5: (69.5, 12.2),
    },
    "poisson_learning": {
        1: (90.2, 4.0),
        2: (93.6, 1.6),
        3: (94.5, 1.1),
        4: (94.9, 0.8),
        5: (95.3, 0.7),
    },
}


class CalderOfficialArtifactError(RuntimeError):
    """Raised when the frozen ModSSC Calder protocol inputs are not exact."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalderOfficialArtifactError(f"cannot read protocol manifest: {path}") from exc
    if not isinstance(raw, dict):
        raise CalderOfficialArtifactError("protocol manifest root must be a mapping")
    return raw


def _result_rows(path: Path) -> dict[int, np.ndarray]:
    rows: dict[int, list[float]] = {budget: [] for budget in range(1, 6)}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CalderOfficialArtifactError(f"cannot read reference result file: {path}") from exc
    for line in lines:
        columns = line.split(",")
        if len(columns) != 2:
            continue
        try:
            total_labels = int(columns[0])
            accuracy = float(columns[1])
        except ValueError:
            continue
        if total_labels % 10 == 0 and total_labels // 10 in rows:
            rows[total_labels // 10].append(accuracy)
    return {budget: np.asarray(values, dtype=np.float64) for budget, values in rows.items()}


def _verify_manifest(root: Path) -> dict[str, Any]:
    manifest = _read_mapping(root / "MANIFEST.json")
    provenance = manifest.get("provenance")
    files = manifest.get("files")
    if not isinstance(provenance, Mapping) or not isinstance(files, Mapping):
        raise CalderOfficialArtifactError("protocol manifest is missing provenance/files mappings")
    expected_provenance = {
        "repository": OFFICIAL_REPOSITORY,
        "commit": OFFICIAL_COMMIT,
        "version": "0.0.3",
        "source_license": "MIT",
        "source_sha256": OFFICIAL_SOURCE_SHA256,
        "original_labels_archive_sha256": OFFICIAL_LABELS_SHA256,
        "original_permutations_archive_sha256": OFFICIAL_PERMUTATIONS_SHA256,
        "mnist_labels_content_sha256": MNIST_LABELS_CONTENT_SHA256,
    }
    if (
        manifest.get("schema_version") != 2
        or manifest.get("kind") != "modssc.calder2020-protocol-inputs"
        or dict(provenance) != expected_provenance
    ):
        raise CalderOfficialArtifactError("protocol manifest provenance differs")
    if dict(files) != PROTOCOL_FILES:
        raise CalderOfficialArtifactError("protocol manifest file pins differ")
    expected_table = {
        "budgets_per_class": [1, 2, 3, 4, 5],
        "trials": 100,
        "permutation_layout": "trial-major, then budget-minor",
        "permutation_format": "ragged_int64_v1",
        "metric": "unlabeled_accuracy_percent",
        "ddof": 0,
    }
    if manifest.get("table1") != expected_table:
        raise CalderOfficialArtifactError("protocol manifest Table 1 contract differs")
    actual_paths = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_paths != {"MANIFEST.json", *PROTOCOL_FILES}:
        raise CalderOfficialArtifactError("protocol input bundle contains unexpected files")
    for relative, metadata in files.items():
        if not isinstance(relative, str) or not isinstance(metadata, Mapping):
            raise CalderOfficialArtifactError("protocol manifest file entry is malformed")
        path = root / relative
        if not path.is_file():
            raise CalderOfficialArtifactError(f"protocol input is missing: {relative}")
        if path.stat().st_size != metadata.get("size_bytes") or sha256_file(path) != metadata.get(
            "sha256"
        ):
            raise CalderOfficialArtifactError(f"protocol input differs: {relative}")
    return manifest


def _load_safe_permutations(root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = root / "splits/mnist-table1-permutations.ragged-int64-v1.npz"
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"metadata_json", "offsets", "values"}:
                raise CalderOfficialArtifactError("safe permutation archive fields differ")
            metadata_bytes = np.asarray(archive["metadata_json"], dtype=np.uint8).tobytes()
            offsets = np.asarray(archive["offsets"])
            values = np.asarray(archive["values"])
    except (OSError, ValueError) as exc:
        raise CalderOfficialArtifactError("cannot read safe permutation archive") from exc
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalderOfficialArtifactError("safe permutation metadata is invalid") from exc
    expected_metadata = {
        "format": "ragged_int64_v1",
        "row_count": 500,
        "schema_version": 1,
        "source_key": "perm",
        "source_sha256": OFFICIAL_PERMUTATIONS_SHA256,
    }
    if metadata != expected_metadata:
        raise CalderOfficialArtifactError("safe permutation metadata differs")
    if offsets.dtype.str != "<i8" or values.dtype.str != "<i8":
        raise CalderOfficialArtifactError("safe permutation arrays must use little-endian int64")
    if offsets.shape != (501,) or values.shape != (15_000,):
        raise CalderOfficialArtifactError("safe permutation arrays have unexpected shapes")
    expected_lengths = np.tile(np.arange(1, 6, dtype=np.int64) * 10, 100)
    if (
        offsets[0] != 0
        or offsets[-1] != values.size
        or not np.array_equal(np.diff(offsets), expected_lengths)
    ):
        raise CalderOfficialArtifactError("safe permutation offsets differ")
    for index in range(500):
        row = values[offsets[index] : offsets[index + 1]]
        if row.min() < 0 or row.max() >= 70_000 or np.unique(row).size != row.size:
            raise CalderOfficialArtifactError("safe permutation row has invalid indices")
    return offsets, values, metadata


def _labels_content_sha256(labels: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(labels, dtype="<i8").reshape(-1))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def verify_calder_official_assets(
    root: Path,
    *,
    dataset_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Authenticate all safe Table 1 inputs without executing upstream code."""

    candidate = root.expanduser()
    if candidate.is_symlink():
        raise CalderOfficialArtifactError("protocol input root must not be a symlink")
    source = candidate.resolve()
    if not source.is_dir():
        raise CalderOfficialArtifactError(f"protocol input root is missing: {source}")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise CalderOfficialArtifactError("protocol input bundle must not contain symlinks")
    manifest = _verify_manifest(source)

    knn_path = source / "graph/mnist-vae-knn30.npz"
    with np.load(knn_path, allow_pickle=False) as archive:
        row_ids = np.asarray(archive["I"], dtype=np.int64)
        neighbors = np.asarray(archive["J"], dtype=np.int64)
        distances = np.asarray(archive["D"], dtype=np.float64)
    if row_ids.shape != (70_000, 30) or neighbors.shape != row_ids.shape:
        raise CalderOfficialArtifactError("MNIST kNN indices have an unexpected shape")
    if distances.shape != row_ids.shape:
        raise CalderOfficialArtifactError("MNIST kNN distances have an unexpected shape")
    expected_rows = np.arange(70_000, dtype=np.int64)[:, None]
    if not np.array_equal(row_ids, np.broadcast_to(expected_rows, row_ids.shape)):
        raise CalderOfficialArtifactError("MNIST kNN query ids differ")
    if not np.array_equal(neighbors[:, 0], np.arange(70_000, dtype=np.int64)):
        raise CalderOfficialArtifactError("paper k=10 semantics do not include self first")
    if not np.array_equal(distances[:, 0], np.zeros(70_000, dtype=np.float64)):
        raise CalderOfficialArtifactError("self-neighbour distances are not zero")

    offsets, permutations, _metadata = _load_safe_permutations(source)
    if dataset_labels is not None:
        labels = np.asarray(dataset_labels, dtype=np.int64).reshape(-1)
        if labels.shape != (70_000,) or set(np.unique(labels).tolist()) != set(range(10)):
            raise CalderOfficialArtifactError("MNIST labels have an unexpected layout")
        if _labels_content_sha256(labels) != MNIST_LABELS_CONTENT_SHA256:
            raise CalderOfficialArtifactError("ModSSC merged MNIST label ordering differs")
        for trial in range(100):
            for budget in range(1, 6):
                row_index = trial * 5 + budget - 1
                indices = permutations[offsets[row_index] : offsets[row_index + 1]]
                counts = np.bincount(labels[indices], minlength=10)
                if not np.array_equal(counts, np.full(10, budget, dtype=np.int64)):
                    raise CalderOfficialArtifactError(
                        "safe permutation row is not balanced per class"
                    )

    result_evidence: dict[str, Any] = {}
    for method_id, filename in {
        "laplace_learning": "mnist-vae-k10-laplace-accuracy.csv",
        "poisson_learning": "mnist-vae-k10-poisson-accuracy.csv",
    }.items():
        rows = _result_rows(source / "references" / filename)
        stats: dict[str, Any] = {}
        for budget, expected in TABLE1_TARGETS[method_id].items():
            values = rows[budget]
            if values.shape != (100,):
                raise CalderOfficialArtifactError(
                    f"reference {method_id} budget {budget} does not contain 100 trials"
                )
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            if round(mean, 1) != expected[0] or round(std, 1) != expected[1]:
                raise CalderOfficialArtifactError(
                    f"reference {method_id} budget {budget} differs from Table 1"
                )
            stats[str(budget)] = {"mean": mean, "std": std}
        result_evidence[method_id] = stats

    return {
        "repository": OFFICIAL_REPOSITORY,
        "commit": OFFICIAL_COMMIT,
        "manifest": manifest,
        "knn_sha256": OFFICIAL_KNN_SHA256,
        "permutations_sha256": OFFICIAL_PERMUTATIONS_SHA256,
        "permutations_artifact_sha256": PERMUTATIONS_ARTIFACT_SHA256,
        "labels_sha256": OFFICIAL_LABELS_SHA256,
        "labels_content_sha256": MNIST_LABELS_CONTENT_SHA256,
        "results": result_evidence,
        "n_nodes": 70_000,
        "stored_neighbors": 30,
        "paper_k_includes_self": True,
        "trials": 100,
    }


__all__ = [
    "CalderOfficialArtifactError",
    "MNIST_LABELS_CONTENT_SHA256",
    "OFFICIAL_COMMIT",
    "OFFICIAL_KNN_SHA256",
    "OFFICIAL_LABELS_SHA256",
    "OFFICIAL_PERMUTATIONS_SHA256",
    "OFFICIAL_REPOSITORY",
    "OFFICIAL_RESULTS_SHA256",
    "OFFICIAL_SOURCE_SHA256",
    "PERMUTATIONS_ARTIFACT_SHA256",
    "PROTOCOL_FILES",
    "TABLE1_TARGETS",
    "sha256_file",
    "verify_calder_official_assets",
]
