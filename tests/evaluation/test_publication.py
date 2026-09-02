from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from modssc.evaluation import (
    PaperPublicationCard,
    PublicationError,
    PublicationRawArchive,
    PublicationSource,
    build_paper_publication,
    evaluate_acceptance,
    reconcile_seed_reports,
    verify_paper_publication,
)
from modssc.runtime.execution import RunIdentity
from modssc.runtime.protocol import effective_config_sha256, protocol_sha256
from modssc.runtime.software import software_sha256

_CARD_HASH = "a" * 64
_DATASET_HASH = "b" * 64
_DISTRIBUTION_HASH = "c" * 64
_ENVIRONMENT_HASH = "d" * 64
_ARCHIVE_MANIFEST_HASH = "e" * 64
_ARCHIVE_HASH = "f" * 64
_SOURCE_COMMIT = "1" * 40
_SOURCE_TREE = "2" * 40
_VERSIONS = {"python": "3.11", "modssc": "1", "numpy": "2", "git_sha": "source"}


def _report(
    seed: int,
    score: float | None = None,
    *,
    method_id: str = "paper_method",
    status: str = "success",
    portable_identity: bool = True,
    error_code: str | None = None,
    raw_error: str | None = None,
    run_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = {"method": {"id": method_id, "params": {}}, "run": {"seed": seed}}
    protocol = protocol_sha256(config)
    software = software_sha256(_VERSIONS)
    identity = RunIdentity(config_sha256=protocol, seed=seed, code_sha256=software)
    report: dict[str, Any] = {
        "run": {
            "seed": seed,
            "name": f"seed-{seed}",
            "run_id": identity.short_id if portable_identity else f"legacy-{seed}",
            "status": status,
            "error_code": error_code,
        },
        "hashes": {
            "config_hash": _CARD_HASH,
            "effective_config_hash": effective_config_sha256(config),
            "protocol_sha256": protocol,
            "software_sha256": software,
        },
        "config": config,
        "versions": dict(_VERSIONS),
        "metrics": None if status != "success" else {"test": {"accuracy": score}},
        "run_info": run_info or {"run_time_seconds": 1.0 + seed},
        "task_info": {"kind": "classification"},
        "graph_info": None,
        "error": raw_error,
    }
    if portable_identity:
        report["execution_identity"] = identity.to_dict()
        report["hashes"]["execution_identity_sha256"] = identity.sha256
    return report


def _source_digest(report: dict[str, Any]) -> str:
    raw = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _card(
    *,
    card_id: str = "paper-card",
    reports: list[dict[str, Any]] | None = None,
    requested_seeds: list[int] | None = None,
    acceptance: bool = True,
) -> PaperPublicationCard:
    reports = reports or [_report(0, 0.7), _report(1, 0.8)]
    requested = requested_seeds or [0, 1]
    expected_effective = {
        int(report["run"]["seed"]): str(report["hashes"]["effective_config_hash"])
        for report in reports
    }
    expected_protocol = {
        int(report["run"]["seed"]): str(report["hashes"]["protocol_sha256"]) for report in reports
    }
    # Missing seeds still have identities derivable from the frozen card.  Test
    # helpers derive them from the same generic configuration shape.
    for seed in requested:
        if seed not in expected_effective:
            config = {"method": {"id": "paper_method", "params": {}}, "run": {"seed": seed}}
            expected_effective[seed] = effective_config_sha256(config)
            expected_protocol[seed] = protocol_sha256(config)
    reconciliation = reconcile_seed_reports(
        requested_seeds=requested,
        reports=reports,
        expected_config_hashes={seed: _CARD_HASH for seed in requested},
        expected_protocol_hashes=expected_protocol,
    )
    acceptance_report = None
    if acceptance:
        acceptance_report = evaluate_acceptance(
            {
                "protocol_id": "paper-protocol-v1",
                "method_id": "paper_method",
                "repetitions": len(requested),
                "fidelity_ceiling": "paper_matched",
                "conformity": {
                    "status": "passed",
                    "basis": "independent equation and protocol review",
                    "evidence": ["docs/protocol-review"],
                    "review": {
                        "reviewed_by": "reviewer",
                        "reviewed_at": "2026-08-30T10:00:00+00:00",
                    },
                },
                "target": {
                    "split": "test",
                    "metric": "accuracy",
                    "published_mean": 0.75,
                    "margin_absolute": 0.2,
                },
            },
            reports,
        )
    return PaperPublicationCard(
        card_id=card_id,
        card_path=f"bench/configs/reproductions/paper_method/{card_id}.yaml",
        card_sha256=_CARD_HASH,
        method_id="paper_method",
        dataset_id="toy:paper",
        dataset_fingerprint=_DATASET_HASH,
        reconciliation=reconciliation,
        effective_config_sha256_by_seed=expected_effective,
        protocol_sha256_by_seed=expected_protocol,
        source_run_sha256_by_seed={
            int(report["run"]["seed"]): _source_digest(report) for report in reports
        },
        acceptance=acceptance_report,
    )


def _source() -> PublicationSource:
    return PublicationSource(
        git_commit=_SOURCE_COMMIT,
        git_tree=_SOURCE_TREE,
        clean=True,
        distribution_sha256=_DISTRIBUTION_HASH,
        environment_manifest_sha256=_ENVIRONMENT_HASH,
    )


def _archive() -> PublicationRawArchive:
    return PublicationRawArchive(
        archive_id="paper-archive-v1",
        archive_ref="archive:modssc/paper-archive-v1",
        format_version=1,
        manifest_sha256=_ARCHIVE_MANIFEST_HASH,
        archive_sha256=_ARCHIVE_HASH,
        bytes=1024,
        verified_after_transfer=True,
    )


def _build(*cards: PaperPublicationCard) -> dict[str, bytes]:
    return build_paper_publication(
        release_id="2026-08-30-paper-source123456",
        created_at="2026-08-30T12:30:00+02:00",
        source=_source(),
        raw_archive=_archive(),
        cards=cards or (_card(acceptance=True),),
        index_markdown="# Paper replication\n\nNo limitation was hidden.\n",
    )


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _rehash(files: dict[str, bytes]) -> None:
    files["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(files[name]).hexdigest()}  {name}\n"
        for name in sorted(set(files) - {"SHA256SUMS"})
    ).encode()


def test_builder_produces_canonical_article_only_bundle_and_verifies_it() -> None:
    files = _build()

    assert set(files) == {
        "SHA256SUMS",
        "index.md",
        "manifest.json",
        "observations.jsonl",
        "results.json",
    }
    verification = verify_paper_publication(files)
    assert verification.release_id == "2026-08-30-paper-source123456"
    assert verification.track == "paper"
    assert verification.card_count == 1
    assert verification.observation_count == 2
    assert verification.certifiable_card_count == 1

    manifest = json.loads(files["manifest.json"])
    assert manifest["created_at"] == "2026-08-30T10:30:00Z"
    assert manifest["track"] == "paper"
    assert manifest["integrity"] == {
        "digest_algorithm": "sha256",
        "line_endings": "LF",
        "serialization": "canonical-json-v1",
        "text_encoding": "utf-8",
    }
    results = json.loads(files["results.json"])
    assert results["cards"][0]["acceptance"]["assessment_status"] == "passed"
    assert results["cards"][0]["reconciliation"]["certifiable"] is True


def test_builder_is_independent_of_card_and_requested_seed_input_order() -> None:
    first = _card(
        card_id="a-card", requested_seeds=[1, 0], reports=[_report(1, 0.8), _report(0, 0.7)]
    )
    second = _card(card_id="z-card", requested_seeds=[0, 1])

    forward = _build(first, second)
    reverse = _build(second, first)

    assert forward == reverse
    manifest = json.loads(forward["manifest.json"])
    assert [card["card_id"] for card in manifest["cards"]] == ["a-card", "z-card"]
    assert manifest["cards"][0]["requested_seeds"] == [0, 1]


def test_projection_never_serializes_operational_paths_raw_errors_or_job_ids() -> None:
    failed = _report(
        1,
        status="failed",
        error_code="E_TRAINING_FAILED",
        raw_error="secret raw traceback at /Users/researcher/private.py",
        run_info={
            "run_time_seconds": 4.0,
            "hostname": "gpu-node-private",
            "SLURM_JOB_ID": "123456",
        },
    )
    success = _report(0, 0.7)
    protocols = {
        int(report["run"]["seed"]): str(report["hashes"]["protocol_sha256"])
        for report in (success, failed)
    }
    reconciliation = reconcile_seed_reports(
        requested_seeds=[0, 1],
        reports=[success, failed],
        source_paths=[Path("/Users/researcher/run-0.json"), Path("/gpfs/work/run-1.json")],
        expected_protocol_hashes=protocols,
    )
    base = _card(reports=[success, failed])
    card = replace(base, reconciliation=reconciliation)

    files = _build(card)
    rendered = b"".join(files.values())

    assert b"/Users/" not in rendered
    assert b"/gpfs/" not in rendered
    assert b"secret raw traceback" not in rendered
    assert b"gpu-node-private" not in rendered
    assert b"SLURM_JOB_ID" not in rendered
    observations = [json.loads(line) for line in files["observations.jsonl"].splitlines()]
    assert observations[1]["status"] == "failed"
    assert observations[1]["error_code"] == "E_TRAINING_FAILED"
    assert observations[1]["metrics"] is None
    assert json.loads(files["results.json"])["cards"][0]["reconciliation"]["certifiable"] is False


def test_missing_run_is_explicit_but_does_not_masquerade_as_legacy_identity() -> None:
    report = _report(0, 0.7)
    files = _build(_card(reports=[report], requested_seeds=[0, 1]))

    observations = [json.loads(line) for line in files["observations.jsonl"].splitlines()]
    assert observations[1] == {
        "card_id": "paper-card",
        "error_code": None,
        "execution_identity_sha256": None,
        "metrics": None,
        "protocol_sha256": protocol_sha256(
            {"method": {"id": "paper_method", "params": {}}, "run": {"seed": 1}}
        ),
        "run_id": None,
        "run_time_seconds": None,
        "seed": 1,
        "software_sha256": None,
        "source_run_sha256": None,
        "status": "missing",
    }
    reconciliation = json.loads(files["results.json"])["cards"][0]["reconciliation"]
    assert reconciliation["execution_identity_complete"] is False
    assert reconciliation["certifiable"] is False


def test_builder_rejects_every_observed_legacy_or_incomplete_execution_identity() -> None:
    legacy = _report(0, 0.7, portable_identity=False)
    reconciliation = reconcile_seed_reports(
        requested_seeds=[0],
        reports=[legacy],
        require_execution_identity=False,
    )
    config = legacy["config"]
    card = PaperPublicationCard(
        card_id="legacy-card",
        card_path="bench/configs/reproductions/paper_method/legacy-card.yaml",
        card_sha256=_CARD_HASH,
        method_id="paper_method",
        dataset_id="toy:paper",
        dataset_fingerprint=_DATASET_HASH,
        reconciliation=reconciliation,
        effective_config_sha256_by_seed={0: effective_config_sha256(config)},
        protocol_sha256_by_seed={0: protocol_sha256(config)},
        source_run_sha256_by_seed={0: _source_digest(legacy)},
    )

    with pytest.raises(PublicationError, match="execution_identity"):
        _build(card)


@pytest.mark.parametrize(
    ("source", "archive", "message"),
    [
        (replace(_source(), clean=False), _archive(), "clean tree"),
        (_source(), replace(_archive(), verified_after_transfer=False), "verified after transfer"),
        (_source(), replace(_archive(), archive_ref="/gpfs/work/archive.tar"), "private path"),
    ],
)
def test_builder_rejects_unsealed_or_nonportable_evidence(
    source: PublicationSource,
    archive: PublicationRawArchive,
    message: str,
) -> None:
    with pytest.raises(PublicationError, match=message):
        build_paper_publication(
            release_id="release",
            created_at="2026-08-30T10:00:00Z",
            source=source,
            raw_archive=archive,
            cards=[_card()],
            index_markdown="# Result\n",
        )


def test_verifier_rejects_added_missing_or_checksum_mismatched_files() -> None:
    files = _build()
    files["raw.log"] = b"secret\n"
    with pytest.raises(PublicationError, match="files differ from schema"):
        verify_paper_publication(files)

    files = _build()
    files.pop("index.md")
    with pytest.raises(PublicationError, match="files differ from schema"):
        verify_paper_publication(files)

    files = _build()
    files["index.md"] += b"mutation\n"
    with pytest.raises(PublicationError, match="digest mismatch"):
        verify_paper_publication(files)


def test_verifier_rejects_unknown_schema_fields_even_with_fresh_checksums() -> None:
    files = _build()
    manifest = json.loads(files["manifest.json"])
    manifest["raw_scheduler_payload"] = {"job_id": 123}
    files["manifest.json"] = _canonical(manifest)
    _rehash(files)

    with pytest.raises(PublicationError, match="fields differ from schema"):
        verify_paper_publication(files)


@pytest.mark.parametrize(
    "private_text",
    [
        "/Users/researcher/private/output",
        "/linkhome/rech/private/output",
        r"C:\\Users\\researcher\\private\\output",
        "SLURM_JOB_ACCOUNT=secret",
    ],
)
def test_verifier_rejects_private_text_even_with_fresh_checksums(private_text: str) -> None:
    files = _build()
    files["index.md"] += f"{private_text}\n".encode()
    _rehash(files)

    with pytest.raises(PublicationError, match="private path"):
        verify_paper_publication(files)


def test_verifier_rejects_cross_file_identity_or_acceptance_tampering() -> None:
    files = _build()
    observations = [json.loads(line) for line in files["observations.jsonl"].splitlines()]
    observations[0]["execution_identity_sha256"] = "9" * 64
    files["observations.jsonl"] = b"".join(_canonical(item) for item in observations)
    _rehash(files)
    with pytest.raises(PublicationError, match="differs from manifest"):
        verify_paper_publication(files)

    files = _build()
    results = json.loads(files["results.json"])
    results["cards"][0]["acceptance"]["assessment_status"] = "failed"
    files["results.json"] = _canonical(results)
    _rehash(files)
    with pytest.raises(PublicationError, match="acceptance_sha256"):
        verify_paper_publication(files)


def test_verifier_rejects_noncanonical_json_even_with_matching_checksum() -> None:
    files = _build()
    manifest = json.loads(files["manifest.json"])
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    _rehash(files)

    with pytest.raises(PublicationError, match="not canonical JSON"):
        verify_paper_publication(files)


def test_builder_rejects_mismatched_seed_identity_maps_and_source_digests() -> None:
    card = _card()
    with pytest.raises(PublicationError, match="keys must exactly match requested seeds"):
        _build(replace(card, source_run_sha256_by_seed={0: "1" * 64}))

    effective = dict(card.effective_config_sha256_by_seed)
    effective[0] = "8" * 64
    with pytest.raises(PublicationError, match="identities differ from declared maps"):
        _build(replace(card, effective_config_sha256_by_seed=effective))

    duplicate_normalized_seed = dict(card.effective_config_sha256_by_seed)
    duplicate_normalized_seed["0"] = duplicate_normalized_seed[0]  # type: ignore[index]
    with pytest.raises(PublicationError, match="duplicate normalized seeds"):
        _build(replace(card, effective_config_sha256_by_seed=duplicate_normalized_seed))


def test_paper_publication_requires_native_acceptance_when_built_and_verified() -> None:
    with pytest.raises(PublicationError, match="requires native acceptance"):
        _build(_card(acceptance=False))

    files = _build()
    results = json.loads(files["results.json"])
    results["cards"][0]["acceptance"] = None
    files["results.json"] = _canonical(results)
    _rehash(files)
    with pytest.raises(PublicationError, match="requires native acceptance"):
        verify_paper_publication(files)


def test_builder_has_no_filesystem_or_orchestrator_inputs() -> None:
    import inspect

    parameters = set(inspect.signature(build_paper_publication).parameters)
    assert parameters == {
        "release_id",
        "created_at",
        "source",
        "raw_archive",
        "cards",
        "index_markdown",
        "supersedes",
    }
    assert not parameters & {"destination", "repo", "git", "slurm", "scheduler", "yaml"}


def test_all_terminal_negative_statuses_remain_publishable_but_uncertifiable() -> None:
    not_evaluable = _card(
        card_id="not-evaluable",
        reports=[
            _report(0, status="not_evaluable", error_code="E_NOT_EVALUABLE"),
            _report(1, status="not_evaluable", error_code="E_NOT_EVALUABLE"),
        ],
        requested_seeds=[0, 1],
    )
    failed = _card(
        card_id="failed",
        reports=[
            _report(0, status="failed", error_code="E_FAILED"),
            _report(1, status="failed", error_code="E_FAILED"),
        ],
        requested_seeds=[0, 1],
    )

    results = json.loads(_build(not_evaluable, failed)["results.json"])
    status_by_card = {
        card["card_id"]: (
            card["reconciliation"]["status"],
            card["reconciliation"]["certifiable"],
        )
        for card in results["cards"]
    }
    assert status_by_card == {
        "failed": ("failed", False),
        "not-evaluable": ("not_evaluable", False),
    }


def test_builder_rejects_invalid_timestamps_and_non_json_metrics() -> None:
    with pytest.raises(PublicationError, match="ISO-8601"):
        build_paper_publication(
            release_id="release",
            created_at="not-a-timestamp",
            source=_source(),
            raw_archive=_archive(),
            cards=[_card()],
            index_markdown="# Result\n",
        )
    with pytest.raises(PublicationError, match="timezone"):
        build_paper_publication(
            release_id="release",
            created_at="2026-08-30T10:00:00",
            source=_source(),
            raw_archive=_archive(),
            cards=[_card()],
            index_markdown="# Result\n",
        )

    card = _card()
    first_run = dict(card.reconciliation.runs[0])
    first_run["metrics"] = {"test": {"opaque": object()}}
    reconciliation = replace(
        card.reconciliation,
        runs=(first_run, *card.reconciliation.runs[1:]),
    )
    with pytest.raises(PublicationError, match="strict JSON"):
        _build(replace(card, reconciliation=reconciliation))


def test_verifier_rejects_invalid_json_and_non_utf8_text() -> None:
    files = _build()
    files["manifest.json"] = b"{\n"
    _rehash(files)
    with pytest.raises(PublicationError, match="UTF-8 JSON"):
        verify_paper_publication(files)

    files = _build()
    files["index.md"] = b"# \xff\n"
    _rehash(files)
    with pytest.raises(PublicationError, match="not UTF-8"):
        verify_paper_publication(files)
