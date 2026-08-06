from __future__ import annotations

import inspect
from pathlib import Path

from bench.campaign.protocols.calder import artifacts
from bench.campaign.protocols.calder.oracle import verify_calder_numerical_oracle

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_public_calder_artifacts_are_package_and_cache_driven() -> None:
    signature = inspect.signature(artifacts.prepare_calder_artifact_lock)
    assert tuple(signature.parameters) == (
        "package_root",
        "cache_root",
        "dataset_cache",
        "output",
    )
    source = Path(artifacts.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "MODSSC_WORK",
        "MODSSC_SCRATCH",
        "execution_site",
        "is_scheduled_execution",
        "src/modssc",
        "provenance/article10",
        "tools.",
    ):
        assert forbidden not in source


def test_packaged_calder_family_and_graph_resolve_without_source_checkout() -> None:
    family = artifacts.load_calder_config_family(REPO_ROOT)
    assert len(family.files) == 10
    assert all(set(record) == {"resource", "sha256"} for record in family.files)
    raw_graph = family.canonical_raw["graph"]["spec"]
    graph = artifacts.materialized_calder_graph_spec(raw_graph, package_root=REPO_ROOT)
    graph_path = Path(graph["precomputed_path"])
    assert graph_path.is_file()
    assert graph_path.is_relative_to(REPO_ROOT / "bench/assets/calder2020/protocol_inputs")


def test_scientific_payload_identity_uses_logical_modules_not_source_paths() -> None:
    family = artifacts.load_calder_config_family(REPO_ROOT)
    identity = artifacts.scientific_payload_identity(family)
    assert identity["kind"] == "modssc_scientific_payload"
    assert len(identity["sha256"]) == 64
    assert all(set(record) == {"module", "sha256"} for record in identity["modules"])
    assert not any("src/modssc" in str(record) for record in identity["modules"])


def test_packaged_calder_numerical_oracle_is_self_authenticated() -> None:
    evidence = verify_calder_numerical_oracle(REPO_ROOT)
    assert evidence["module"] == "modssc.transductive.methods.classic.laplace_learning"
    assert evidence["scope"] == "sealed_historical_replay"
    assert len(evidence["audited_modssc_source_sha256"]) == 64
    assert evidence["prediction_sha256"] != evidence["score_sha256"]
    assert len(evidence["seal_sha256"]) == 64
