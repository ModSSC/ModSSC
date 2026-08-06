from __future__ import annotations

DCL_PARTITION_SELECTION_KIND = "modssc.dcl-vote-conditioned-partition-selection"

PARTITION_SELECTION_TASK_FIELDS = frozenset(
    {
        "kind",
        "selection_path",
        "selection_sha256",
        "selection_rank",
        "source_task_id",
        "source_task_row_sha256",
        "replay_path",
        "split_fingerprint",
        "split_manifest_sha256",
        "split_json_sha256",
        "split_arrays_sha256",
    }
)

PARTITION_SELECTION_DIGEST_FIELDS = frozenset(
    {
        "selection_sha256",
        "source_task_id",
        "source_task_row_sha256",
        "split_fingerprint",
        "split_manifest_sha256",
        "split_json_sha256",
        "split_arrays_sha256",
    }
)
