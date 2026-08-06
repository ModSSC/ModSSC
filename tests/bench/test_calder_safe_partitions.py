from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from bench.campaign.protocols.calder.official import (
    OFFICIAL_PERMUTATIONS_SHA256,
    PERMUTATIONS_ARTIFACT_SHA256,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    REPO_ROOT / "bench/assets/calder2020/protocol_inputs/splits/"
    "mnist-table1-permutations.ragged-int64-v1.npz"
)


def test_committed_calder_partitions_are_safe_complete_and_authenticated() -> None:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == PERMUTATIONS_ARTIFACT_SHA256
    assert ARTIFACT.stat().st_mode & 0o777 == 0o644

    with np.load(ARTIFACT, allow_pickle=False) as archive:
        assert set(archive.files) == {"metadata_json", "offsets", "values"}
        metadata = json.loads(np.asarray(archive["metadata_json"], dtype=np.uint8).tobytes())
        offsets = np.asarray(archive["offsets"])
        values = np.asarray(archive["values"])

    assert metadata == {
        "format": "ragged_int64_v1",
        "row_count": 500,
        "schema_version": 1,
        "source_key": "perm",
        "source_sha256": OFFICIAL_PERMUTATIONS_SHA256,
    }
    assert offsets.dtype.str == values.dtype.str == "<i8"
    assert offsets.shape == (501,)
    assert values.shape == (15_000,)
    assert offsets[0] == 0
    assert offsets[-1] == values.size
    np.testing.assert_array_equal(
        np.diff(offsets),
        np.tile(np.arange(1, 6, dtype=np.int64) * 10, 100),
    )
    for row_index in range(500):
        row = values[offsets[row_index] : offsets[row_index + 1]]
        assert row.min() >= 0
        assert row.max() < 70_000
        assert np.unique(row).size == row.size


def test_calder_runtime_never_enables_pickle_loading() -> None:
    runtime_root = REPO_ROOT / "bench/campaign/protocols/calder"
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in runtime_root.rglob("*.py")
        if "allow_pickle=True" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
