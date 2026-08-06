from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = REPO_ROOT / "provenance/article10/evidence/execution-history-bundle.json"


def test_execution_history_uses_a_portable_content_identity() -> None:
    record = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert record["schema_version"] == 1
    assert record["complete_committed_history"] is True
    assert record["artifact_uri"] == (
        f"modssc-artifact://replication/provenance/execution-history/{record['sha256']}"
    )
    assert len(record["sha256"]) == 64
    assert record["size_bytes"] > 0
    assert len(record["execution_commits"]) == 7
    assert all(len(commit) == 40 for commit in record["execution_commits"])
    assert "/Users/" not in EVIDENCE.read_text(encoding="utf-8")
    assert "/" + "lustre/" not in EVIDENCE.read_text(encoding="utf-8")


def test_dirty_execution_cannot_be_claimed_without_its_exact_snapshot() -> None:
    record = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    dirty = record["dirty_execution_provenance"]

    assert len(dirty) == 1
    assert dirty[0]["snapshot_status"] == "incomplete"
    assert dirty[0]["claim_eligible"] is False
    assert len(dirty[0]["expected_worktree_sha256"]) == 64
