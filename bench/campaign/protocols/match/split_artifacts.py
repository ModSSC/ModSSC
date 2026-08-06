"""Materialize authenticated CIFAR-10 paper partitions for Match protocols.

This script has no project dependency beyond NumPy. It mirrors the pinned
Google FixMatch and Microsoft USB split generators and writes deterministic
NumPy archives whose array order is part of the scientific protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import tarfile
import zipfile
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

GOOGLE_FIXMATCH_COMMIT = "d4985a158065947dba803e626ee9a6721709c570"
TORCHSSL_COMMIT = "03193a1b7883727db1ce9c092e083091e18aedbb"
MICROSOFT_USB_COMMIT = "1ef4cbebcc0b368158315aeb425053858cf6c845"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "assets" / "cifar10_paper_splits"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cifar10_python(archive_path: Path) -> tuple[np.ndarray, int]:
    labels: list[int] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for batch in range(1, 6):
            member = archive.extractfile(f"cifar-10-batches-py/data_batch_{batch}")
            if member is None:
                raise ValueError(f"missing CIFAR-10 training batch {batch}")
            payload = pickle.load(member, encoding="bytes")
            labels.extend(int(label) for label in payload[b"labels"])
        test_member = archive.extractfile("cifar-10-batches-py/test_batch")
        if test_member is None:
            raise ValueError("missing CIFAR-10 test batch")
        test_payload = pickle.load(test_member, encoding="bytes")
    return np.asarray(labels, dtype=np.int64), len(test_payload[b"labels"])


def _load_cifar10_labels(source_path: Path) -> tuple[np.ndarray, int, str]:
    if source_path.suffix == ".npy":
        labels = np.load(source_path, allow_pickle=False)
        return (
            np.asarray(labels, dtype=np.int64),
            10_000,
            "ModSSC canonical CIFAR-10 train_y.npy",
        )
    if source_path.suffix == ".pt":
        with zipfile.ZipFile(source_path) as archive:
            payload = pickle.loads(archive.read("archive/data.pkl"))
        labels = np.asarray(payload["clean_label"], dtype=np.int64)
        return labels, 10_000, "UCSC-REAL/CIFAR-10_human.pt:clean_label"
    labels, test_size = _load_cifar10_python(source_path)
    return labels, test_size, "Toronto CIFAR-10 Python archive"


def _google_unlabeled_order(labels: np.ndarray) -> np.ndarray:
    classes = int(np.max(labels)) + 1
    class_data = [deque() for _ in range(classes)]
    positions = np.zeros(classes, dtype=np.int64)
    train_stats = np.asarray(
        [np.count_nonzero(labels == label) for label in range(classes)],
        dtype=np.float64,
    )
    train_stats /= train_stats.max()
    ordered: list[int] = []
    for position, label in enumerate(labels):
        class_data[int(label)].append(position)
        while True:
            selected_class = int(np.argmax(train_stats - positions / max(int(positions.max()), 1)))
            if not class_data[selected_class]:
                break
            ordered.append(class_data[selected_class].popleft())
            positions[selected_class] += 1
    for remaining in class_data:
        ordered.extend(remaining)
    return np.asarray(ordered, dtype=np.int64)


def _google_labeled_indices(
    labels: np.ndarray,
    *,
    seed: int,
    size: int,
) -> np.ndarray:
    classes = int(np.max(labels)) + 1
    by_class = [np.flatnonzero(labels == label).astype(np.int64) for label in range(classes)]
    rng = np.random.RandomState(seed)
    for indices in by_class:
        rng.shuffle(indices)
    train_stats = np.asarray([indices.size for indices in by_class], dtype=np.float64)
    train_stats /= train_stats.max()
    positions = np.zeros(classes, dtype=np.int64)
    selected: list[int] = []
    for _ in range(size):
        selected_class = int(np.argmax(train_stats - positions / max(int(positions.max()), 1)))
        selected.append(int(by_class[selected_class][positions[selected_class]]))
        positions[selected_class] += 1
    return np.sort(np.asarray(selected, dtype=np.int64))


def _torchssl_pools(
    labels: np.ndarray,
    *,
    seed: int,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    classes = int(np.max(labels)) + 1
    if size % classes:
        raise ValueError("balanced TorchSSL labels must divide evenly by class")
    per_class = size // classes
    rng = np.random.RandomState(seed)
    labeled: list[int] = []
    for label in range(classes):
        indices = np.flatnonzero(labels == label).astype(np.int64)
        rng.shuffle(indices)
        labeled.extend(int(index) for index in indices[:per_class])
    labeled_array = np.asarray(labeled, dtype=np.int64)
    # TorchSSL's ``split_ssl_data(..., include_lb_to_ulb=True)`` returns the
    # original complete ``data`` array. Its order is therefore the canonical
    # CIFAR-10 order, not a labeled-first reconstruction.
    unlabeled_array = np.arange(labels.size, dtype=np.int64)
    return labeled_array, unlabeled_array


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def _write_deterministic_npz(
    path: Path,
    *,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    metadata_bytes = json.dumps(
        metadata,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    payloads = {
        "metadata_json": np.frombuffer(metadata_bytes, dtype=np.uint8),
        **arrays,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(payloads):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                _npy_bytes(payloads[name]),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _partition_arrays(
    *,
    seeds: list[int],
    train_by_seed: dict[int, np.ndarray],
    val_by_seed: dict[int, np.ndarray],
    labeled_by_seed: dict[int, np.ndarray],
    unlabeled_by_seed: dict[int, np.ndarray],
    test_size: int,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    test = np.arange(test_size, dtype=np.int64)
    for seed in seeds:
        prefix = f"seed_{seed}__"
        arrays[f"{prefix}train"] = train_by_seed[seed]
        arrays[f"{prefix}val"] = val_by_seed[seed]
        arrays[f"{prefix}test"] = test
        arrays[f"{prefix}train_labeled"] = labeled_by_seed[seed]
        arrays[f"{prefix}train_unlabeled"] = unlabeled_by_seed[seed]
    return arrays


def generate(
    source_path: Path,
    output_dir: Path,
    *,
    dataset_fingerprint: str | None = None,
) -> list[Path]:
    labels, test_size, label_source = _load_cifar10_labels(source_path)
    if labels.shape != (50_000,) or test_size != 10_000:
        raise ValueError("unexpected CIFAR-10 source sizes")
    source_sha256 = _sha256(source_path)
    labels_sha256 = hashlib.sha256(np.asarray(labels, dtype="<i8").tobytes()).hexdigest()
    outputs: list[Path] = []

    google_seeds = list(range(1, 6))
    google_unlabeled = _google_unlabeled_order(labels)
    google_val = google_unlabeled[:1]
    google_train = google_unlabeled[1:]
    google_labels = {
        seed: _google_labeled_indices(labels, seed=seed, size=250) for seed in google_seeds
    }
    for seed, labeled in google_labels.items():
        if np.intersect1d(labeled, google_val).size:
            raise ValueError(f"FixMatch seed {seed} labels the one-example validation record")
        if np.setdiff1d(labeled, google_train).size:
            raise ValueError(f"FixMatch seed {seed} labels are outside the train pool")
    google_path = output_dir / "fixmatch-google-cifar10-250-seeds1-5.npz"
    _write_deterministic_npz(
        google_path,
        metadata={
            "artifact_id": "fixmatch-google-cifar10-250-seeds1-5",
            "canonical_labels_sha256": labels_sha256,
            "canonical_label_source": label_source,
            "canonical_label_source_sha256": source_sha256,
            "dataset_fingerprint": dataset_fingerprint,
            "generator": (
                "google-research/fixmatch/scripts/create_split.py and scripts/create_unlabeled.py"
            ),
            "source_commit": GOOGLE_FIXMATCH_COMMIT,
            "schema_version": 1,
            "seeds": google_seeds,
            "test_ref": "test",
            "test_source_size": test_size,
            "train_source_size": int(labels.size),
            "unlabeled_pool": "includes_labeled",
            "validation_size": 1,
        },
        arrays=_partition_arrays(
            seeds=google_seeds,
            train_by_seed={seed: google_train for seed in google_seeds},
            val_by_seed={seed: google_val for seed in google_seeds},
            labeled_by_seed=google_labels,
            unlabeled_by_seed={seed: google_train for seed in google_seeds},
            test_size=test_size,
        ),
    )
    outputs.append(google_path)

    for label_count in (40, 250):
        usb_seeds = [0, 1, 2]
        usb_labels: dict[int, np.ndarray] = {}
        usb_unlabeled: dict[int, np.ndarray] = {}
        for seed in usb_seeds:
            labeled, unlabeled = _torchssl_pools(
                labels,
                seed=seed,
                size=label_count,
            )
            usb_labels[seed] = labeled
            usb_unlabeled[seed] = unlabeled
        usb_path = output_dir / f"torchssl-cifar10-{label_count}-seeds0-2.npz"
        _write_deterministic_npz(
            usb_path,
            metadata={
                "artifact_id": f"torchssl-cifar10-{label_count}-seeds0-2",
                "canonical_labels_sha256": labels_sha256,
                "canonical_label_source": label_source,
                "canonical_label_source_sha256": source_sha256,
                "dataset_fingerprint": dataset_fingerprint,
                "generator": "datasets/data_utils.py:split_ssl_data",
                "include_lb_to_ulb": True,
                "source_commit": TORCHSSL_COMMIT,
                "secondary_control_commit": MICROSOFT_USB_COMMIT,
                "schema_version": 1,
                "seeds": usb_seeds,
                "test_ref": "test",
                "test_source_size": test_size,
                "train_source_size": int(labels.size),
                "unlabeled_pool": "includes_labeled",
                "validation_size": 0,
            },
            arrays=_partition_arrays(
                seeds=usb_seeds,
                train_by_seed=usb_unlabeled,
                val_by_seed={seed: np.asarray([], dtype=np.int64) for seed in usb_seeds},
                labeled_by_seed=usb_labels,
                unlabeled_by_seed=usb_unlabeled,
                test_size=test_size,
            ),
        )
        outputs.append(usb_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cifar10_label_source",
        type=Path,
        help="Toronto cifar-10-python.tar.gz or CIFAR-10_human.pt",
    )
    parser.add_argument("--dataset-fingerprint")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()
    for output in generate(
        args.cifar10_label_source.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        dataset_fingerprint=args.dataset_fingerprint,
    ):
        print(f"{output.name}  sha256={_sha256(output)}")


if __name__ == "__main__":
    main()
