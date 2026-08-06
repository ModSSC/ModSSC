from __future__ import annotations

import pytest

from bench.orchestrators.method_inductive import _partition_artifact_sha256


@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        (
            {
                "policy": {"partition_artifact_sha256": "current"},
                "partition_artifact_sha256": "legacy",
            },
            "current",
        ),
        ({"partition_artifact_sha256": "legacy"}, "legacy"),
        ({"policy": {}}, None),
        ({"policy": "not-a-mapping", "partition_artifact_sha256": "legacy"}, "legacy"),
    ],
)
def test_partition_artifact_sha256_reads_current_and_legacy_stats(
    stats: dict[str, object],
    expected: str | None,
) -> None:
    assert _partition_artifact_sha256(stats) == expected
