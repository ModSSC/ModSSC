from __future__ import annotations

import hashlib
import json
import os
import sys
import tomllib
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bench import reproduce
from modssc.data_loader.errors import DataLoaderError


def _raw_card(
    *,
    method: str = "demo",
    profile: str = "paper:demo-2026",
    dataset: str = "toy",
    run_name: str = "reproduction_demo",
    seeds: list[int] | None = None,
    download: bool = False,
) -> dict[str, object]:
    return {
        "run": {"name": run_name, "seed": 1, "seeds": seeds or [1, 2]},
        "dataset": {"id": dataset, "download": download, "options": {}},
        "method": {"id": method, "profile": profile},
    }


def _write_card(root: Path, relative: str, raw: dict[str, object]) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _card(path: Path, *, card_id: str = "demo/card") -> reproduce.ReproductionCard:
    return reproduce.ReproductionCard(
        card_id=card_id,
        config_path=path,
        method_id="demo",
        profile="paper:demo-2026",
        dataset_id="toy",
        run_name="reproduction_demo",
        repetitions=2,
    )


def _demo_pin() -> reproduce.DatasetIntegrityPin:
    return reproduce.DatasetIntegrityPin(
        profile="paper:demo-2026",
        dataset_id="toy",
        options={},
        fingerprint="a" * 64,
        content_sha256="b" * 64,
        evidence="test fixture",
    )


def test_discover_cards_deduplicates_deployment_variants_and_skips_diagnostics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cards"
    _write_card(root, "demo/base.yaml", _raw_card())
    _write_card(root, "demo/base-a100.yaml", _raw_card(run_name="reproduction_demo_a100"))
    _write_card(
        root,
        "demo/screen.yaml",
        _raw_card(profile="paper:screen", run_name="screening_demo"),
    )
    _write_card(root, "demo/non-paper.yaml", _raw_card(profile="standardized"))

    cards = reproduce.discover_cards(root)

    assert [card.card_id for card in cards] == ["demo/base"]
    assert cards[0].repetitions == 2
    assert cards[0].as_dict()["method_id"] == "demo"


def test_checkout_has_23_self_contained_canonical_cards() -> None:
    cards = reproduce.discover_cards()
    reports = [reproduce.verify_card(card) for card in cards]

    assert len(cards) == 23
    assert [card.card_id for card in cards if card.method_id == "co_training"] == [
        "co_training/webkb_course_nigam_ghani_2000"
    ]
    assert all(report.execution_ready for report in reports), {
        report.card.card_id: [issue.code for issue in report.issues]
        for report in reports
        if not report.execution_ready
    }


def test_discover_cards_rejects_missing_directory_and_malformed_cards(tmp_path: Path) -> None:
    with pytest.raises(reproduce.ReproductionRegistryError, match="does not exist"):
        reproduce.discover_cards(tmp_path / "missing")

    root = tmp_path / "cards"
    _write_card(root, "demo/bad.yaml", {"run": {}})
    with pytest.raises(reproduce.ReproductionRegistryError, match="dataset must be a mapping"):
        reproduce.discover_cards(root)

    (root / "demo" / "bad.yaml").write_text("[not, a, mapping]", encoding="utf-8")
    with pytest.raises(reproduce.ReproductionRegistryError, match="must contain a mapping"):
        reproduce.discover_cards(root)


def test_resolve_card_accepts_id_profile_short_profile_and_unambiguous_method(
    tmp_path: Path,
) -> None:
    first = _card(tmp_path / "first.yaml")
    second = reproduce.ReproductionCard(
        card_id="other/card",
        config_path=tmp_path / "second.yaml",
        method_id="other",
        profile="paper:other-2026",
        dataset_id="toy",
        run_name="reproduction_other",
        repetitions=1,
    )
    cards = (first, second)

    assert reproduce.resolve_card("demo/card", cards) is first
    assert reproduce.resolve_card("paper:demo-2026", cards) is first
    assert reproduce.resolve_card("demo-2026", cards) is first
    assert reproduce.resolve_card("other", cards) is second

    duplicate = reproduce.ReproductionCard(
        card_id="demo/second",
        config_path=tmp_path / "third.yaml",
        method_id="demo",
        profile="paper:demo-second",
        dataset_id="toy",
        run_name="reproduction_demo_second",
        repetitions=1,
    )
    with pytest.raises(reproduce.ReproductionRegistryError, match="Ambiguous"):
        reproduce.resolve_card("demo", (*cards, duplicate))
    with pytest.raises(reproduce.ReproductionRegistryError, match="Unknown"):
        reproduce.resolve_card("missing", cards)


def test_verify_card_accepts_local_data_artifact_and_checks_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "artifact.npz"
    artifact.write_bytes(b"paper data")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    raw = _raw_card()
    raw["sampling"] = {
        "partition": {
            "ordered_indices_artifact": {
                "path": "${MODSSC_ROOT}/artifact.npz",
                "sha256": digest,
            }
        }
    }
    path = _write_card(tmp_path, "card.yaml", raw)
    monkeypatch.setattr(reproduce, "dataset_info", lambda _dataset: SimpleNamespace())
    monkeypatch.setattr(reproduce, "_dataset_integrity_pin", lambda _profile: _demo_pin())
    monkeypatch.setattr(reproduce.ExperimentConfig, "from_dict", lambda _raw: SimpleNamespace())

    report = reproduce.verify_card(_card(path), repo_root=tmp_path)

    assert report.execution_ready is True
    assert report.as_dict()["execution_ready"] is True
    assert report.as_dict()["scientific_status"] == "not_evaluated"
    assert "ready" not in report.as_dict()
    assert report.as_dict()["issues"] == []


def test_verify_card_rejects_missing_dataset_integrity_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _write_card(tmp_path, "card.yaml", _raw_card())
    monkeypatch.setattr(reproduce, "dataset_info", lambda _dataset: SimpleNamespace())
    monkeypatch.setattr(reproduce, "_dataset_integrity_pin", lambda _profile: None)
    monkeypatch.setattr(reproduce.ExperimentConfig, "from_dict", lambda _raw: SimpleNamespace())

    report = reproduce.verify_card(_card(path), repo_root=tmp_path)

    assert report.execution_ready is False
    assert "E_REPRO_DATASET_UNPINNED" in {issue.code for issue in report.issues}


def test_verify_card_reports_unsafe_or_incomplete_runtime_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "helper.py"
    source.write_text("pass\n", encoding="utf-8")
    mismatch = tmp_path / "wrong.npz"
    mismatch.write_bytes(b"wrong")
    raw = _raw_card(download=True)
    raw["method"] = {
        "id": "demo",
        "profile": "paper:demo-2026",
        "params": {
            "classifier_backend": "weka",
            "expected_jar_sha256": "abc",
            "helper_path": "${MODSSC_ROOT}/helper.py",
        },
    }
    raw["sampling"] = {
        "missing": {"path": "missing.npz"},
        "wrong": {"path": "wrong.npz", "sha256": "0" * 64},
        "future": "REPLACE_WITH_REAL_VALUE",
        "unknown": "${UNKNOWN_REPRO_ROOT}/value",
    }
    path = _write_card(tmp_path, "card.yaml", raw)

    def unknown_dataset(_dataset: str) -> None:
        raise DataLoaderError("not in the catalog")

    monkeypatch.setattr(reproduce, "dataset_info", unknown_dataset)

    def invalid_config(_raw: object) -> None:
        raise reproduce.BenchConfigError("invalid")

    monkeypatch.setattr(reproduce.ExperimentConfig, "from_dict", invalid_config)
    report = reproduce.verify_card(_card(path), repo_root=tmp_path)
    codes = {issue.code for issue in report.issues}

    assert report.execution_ready is False
    assert {
        "E_REPRO_CONFIG",
        "E_REPRO_DATASET",
        "E_REPRO_DOWNLOAD_POLICY",
        "E_REPRO_ENV",
        "E_REPRO_EXTERNAL_CODE",
        "E_REPRO_PLACEHOLDER",
        "E_REPRO_RESOURCE_HASH",
        "E_REPRO_RESOURCE_MISSING",
    } <= codes


@pytest.mark.parametrize("pin", [None, "not-a-sha256", "g" * 64])
def test_verify_card_rejects_unpinned_packaged_resource(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pin: str | None,
) -> None:
    artifact = tmp_path / "artifact.npz"
    artifact.write_bytes(b"paper data")
    resource: dict[str, str] = {"path": "artifact.npz"}
    if pin is not None:
        resource["sha256"] = pin
    raw = _raw_card()
    raw["sampling"] = {"resource": resource}
    path = _write_card(tmp_path, "card.yaml", raw)
    monkeypatch.setattr(reproduce, "dataset_info", lambda _dataset: SimpleNamespace())
    monkeypatch.setattr(reproduce.ExperimentConfig, "from_dict", lambda _raw: SimpleNamespace())

    report = reproduce.verify_card(_card(path), repo_root=tmp_path)

    assert "E_REPRO_RESOURCE_UNPINNED" in {issue.code for issue in report.issues}


def test_verify_card_rejects_absolute_outside_and_symlink_escape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    inside = root / "inside.npz"
    inside.write_bytes(b"inside")
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"outside")
    symlink = root / "escape.npz"
    symlink.symlink_to(outside)
    raw = _raw_card()
    raw["sampling"] = {
        "absolute": {
            "path": str(inside),
            "sha256": hashlib.sha256(inside.read_bytes()).hexdigest(),
        },
        "outside": {
            "path": "../outside.npz",
            "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        },
        "symlink": {
            "path": "escape.npz",
            "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        },
    }
    path = _write_card(root, "card.yaml", raw)
    monkeypatch.setattr(reproduce, "dataset_info", lambda _dataset: SimpleNamespace())
    monkeypatch.setattr(reproduce.ExperimentConfig, "from_dict", lambda _raw: SimpleNamespace())

    report = reproduce.verify_card(_card(path), repo_root=root)

    boundary = [issue for issue in report.issues if issue.code == "E_REPRO_RESOURCE_BOUNDARY"]
    assert {issue.location for issue in boundary} == {
        "sampling.absolute.path",
        "sampling.outside.path",
        "sampling.symlink.path",
    }


def test_verify_card_reports_unreadable_config(tmp_path: Path) -> None:
    path = tmp_path / "card.yaml"
    path.write_text("[bad]", encoding="utf-8")
    report = reproduce.verify_card(_card(path), repo_root=tmp_path)
    assert report.execution_ready is False
    assert report.issues[0].code == "E_REPRO_CONFIG"


def test_prepare_card_uses_provider_without_changing_download_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = _raw_card(download=False)
    path = _write_card(tmp_path, "card.yaml", raw)
    calls: list[dict[str, object]] = []

    def fake_download(dataset_id: str, **kwargs: object) -> SimpleNamespace:
        calls.append({"dataset_id": dataset_id, **kwargs})
        return SimpleNamespace(meta={"dataset_fingerprint": "abc123"})

    monkeypatch.setattr(reproduce, "download_dataset", fake_download)
    monkeypatch.setattr(reproduce, "_require_static_verification", lambda _card: None)
    monkeypatch.setattr(
        reproduce,
        "_verify_prepared_dataset_integrity",
        lambda *_args, **_kwargs: (
            "abc123",
            "d" * 64,
            "paper_identity_authenticated",
            "dataset-integrity:paper-identity-authenticated",
        ),
    )
    cache = tmp_path / "cache"
    result = reproduce.prepare_card(_card(path), cache_dir=cache, force=True)

    assert result.dataset_fingerprint == "abc123"
    assert result.dataset_content_sha256 == "d" * 64
    assert result.dataset_integrity == "paper_identity_authenticated"
    assert calls == [
        {"dataset_id": "toy", "cache_dir": cache.resolve(), "force": True, "options": {}}
    ]
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["dataset"]["download"] is False


def test_prepare_card_rejects_invalid_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = _raw_card()
    raw["dataset"]["options"] = []  # type: ignore[index]
    path = _write_card(tmp_path, "card.yaml", raw)
    monkeypatch.setattr(reproduce, "_require_static_verification", lambda _card: None)
    with pytest.raises(reproduce.ReproductionRegistryError, match="options must be a mapping"):
        reproduce.prepare_card(_card(path), cache_dir=tmp_path / "cache")


def test_dataset_integrity_registry_contains_only_valid_evidence_backed_pins() -> None:
    pins = reproduce._load_dataset_integrity_registry()

    assert len(pins) == 23
    assert {card.profile for card in reproduce.discover_cards()} == set(pins)
    assert pins["paper:sohn2020-cifar10-table2-250"].dataset_id == "cifar10"
    assert pins["paper:zhou-li-2005-vote-table3-j48"].evidence
    assert all(len(pin.fingerprint) == 64 for pin in pins.values())
    assert all(len(pin.content_sha256) == 64 for pin in pins.values())


def test_dataset_integrity_registry_rejects_unproven_values(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocols": {
                    "paper:test": {
                        "dataset_id": "toy",
                        "options": {},
                        "fingerprint": "not-proven",
                        "content_sha256": "d" * 64,
                        "evidence": "test",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(reproduce.ReproductionRegistryError, match="fingerprint pin"):
        reproduce._load_dataset_integrity_registry(path)


def test_verify_prepared_dataset_integrity_authenticates_registered_pin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fingerprint = "a" * 64
    content = "b" * 64
    loaded = SimpleNamespace(
        meta={"dataset_fingerprint": fingerprint, "dataset_content_sha256": content}
    )
    pin = reproduce.DatasetIntegrityPin(
        profile="paper:demo-2026",
        dataset_id="toy",
        options={},
        fingerprint=fingerprint,
        content_sha256=content,
        evidence="existing test evidence",
    )
    raw = _raw_card()
    raw["sampling"] = {"plan": {}}
    monkeypatch.setattr(
        reproduce,
        "verify_dataset_content",
        lambda *_args, **_kwargs: {"content_sha256": content},
    )
    monkeypatch.setattr(reproduce, "_dataset_integrity_pin", lambda _profile: pin)
    from bench.orchestrators import sampling

    monkeypatch.setattr(sampling, "prepare_dataset", lambda dataset, **_kwargs: dataset)

    observed = reproduce._verify_prepared_dataset_integrity(
        _card(tmp_path / "card.yaml"),
        raw,
        loaded,
        cache_dir=tmp_path,
        options={},
    )

    assert observed == (
        fingerprint,
        content,
        "paper_identity_authenticated",
        "dataset-integrity:paper-identity-authenticated",
    )


def test_verify_prepared_dataset_integrity_rejects_missing_protocol_pin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fingerprint = "a" * 64
    content = "b" * 64
    loaded = SimpleNamespace(
        meta={"dataset_fingerprint": fingerprint, "dataset_content_sha256": content}
    )
    raw = _raw_card()
    raw["sampling"] = {"plan": {}}
    monkeypatch.setattr(
        reproduce,
        "verify_dataset_content",
        lambda *_args, **_kwargs: {"content_sha256": content},
    )
    monkeypatch.setattr(reproduce, "_dataset_integrity_pin", lambda _profile: None)
    from bench.orchestrators import sampling

    monkeypatch.setattr(sampling, "prepare_dataset", lambda dataset, **_kwargs: dataset)

    with pytest.raises(reproduce.ReproductionRegistryError, match="No packaged.*pin"):
        reproduce._verify_prepared_dataset_integrity(
            _card(tmp_path / "card.yaml"),
            raw,
            loaded,
            cache_dir=tmp_path,
            options={},
        )


@pytest.mark.parametrize("field", ["fingerprint", "content"])
def test_verify_prepared_dataset_integrity_rejects_pin_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
) -> None:
    fingerprint = "a" * 64
    content = "b" * 64
    loaded = SimpleNamespace(
        meta={"dataset_fingerprint": fingerprint, "dataset_content_sha256": content}
    )
    pin = reproduce.DatasetIntegrityPin(
        profile="paper:demo-2026",
        dataset_id="toy",
        options={},
        fingerprint=("c" * 64 if field == "fingerprint" else fingerprint),
        content_sha256=("d" * 64 if field == "content" else content),
        evidence="existing test evidence",
    )
    raw = _raw_card()
    raw["sampling"] = {"plan": {}}
    monkeypatch.setattr(
        reproduce,
        "verify_dataset_content",
        lambda *_args, **_kwargs: {"content_sha256": content},
    )
    monkeypatch.setattr(reproduce, "_dataset_integrity_pin", lambda _profile: pin)
    from bench.orchestrators import sampling

    monkeypatch.setattr(sampling, "prepare_dataset", lambda dataset, **_kwargs: dataset)

    with pytest.raises(reproduce.ReproductionRegistryError, match=f"Dataset {field} mismatch"):
        reproduce._verify_prepared_dataset_integrity(
            _card(tmp_path / "card.yaml"),
            raw,
            loaded,
            cache_dir=tmp_path,
            options={},
        )


def test_dataset_cache_environment_restores_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MODSSC_CACHE_DIR", "old-cache")
    monkeypatch.delenv("MODSSC_DATASET_CACHE_DIR", raising=False)
    target = tmp_path / "new-cache"
    with reproduce._dataset_cache_environment(target) as resolved:
        assert resolved == target.resolve()
        assert os.environ["MODSSC_CACHE_DIR"] == str(target.resolve())
        assert os.environ["MODSSC_DATASET_CACHE_DIR"] == str(target.resolve())
    assert os.environ["MODSSC_CACHE_DIR"] == "old-cache"
    assert "MODSSC_DATASET_CACHE_DIR" not in os.environ


def test_dataset_cache_environment_exposes_all_defaults_without_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in tuple(os.environ):
        if name.startswith("MODSSC_"):
            monkeypatch.delenv(name, raising=False)
    root = tmp_path / "cache-root"
    monkeypatch.setattr(reproduce, "default_dataset_cache_dir", lambda: root / "datasets")

    with reproduce._dataset_cache_environment(None) as resolved:
        assert resolved == (root / "datasets").resolve()
        assert os.environ["MODSSC_ROOT"] == str(reproduce.REPO_ROOT)
        assert os.environ["MODSSC_CACHE_ROOT"] == str(root.resolve())
        assert os.environ["MODSSC_DATASET_CACHE_DIR"] == str((root / "datasets").resolve())
        assert os.environ["MODSSC_OUTPUT_DIR"] == str((root / "output").resolve())
        assert os.environ["MODSSC_PREPROCESS_CACHE_DIR"] == str((root / "preprocess").resolve())
        assert os.environ["MODSSC_GRAPH_CACHE_DIR"] == str((root / "graph").resolve())
        assert os.environ["MODSSC_GRAPH_VIEWS_CACHE_DIR"] == str((root / "graph_views").resolve())
        assert os.environ["MODSSC_SPLIT_CACHE_DIR"] == str((root / "splits").resolve())

    assert not any(name.startswith("MODSSC_") for name in os.environ)


def test_wheel_packages_autonomous_bench_runner_and_entrypoint() -> None:
    project = tomllib.loads((reproduce.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"]["modssc-reproduce"] == "bench.reproduce:main"
    build = project["tool"]["hatch"]["build"]
    assert "/bench" not in build["exclude"]
    assert {"/tools", "/provenance", "/tests/tools"} <= set(build["exclude"])
    assert build["targets"]["wheel"]["packages"] == ["src/modssc", "bench"]


def test_working_directory_restores_cwd(tmp_path: Path) -> None:
    previous = Path.cwd()
    with reproduce._working_directory(tmp_path):
        assert Path.cwd() == tmp_path.resolve()
    assert Path.cwd() == previous


def test_prepare_match_splits_authenticates_each_seed(tmp_path: Path) -> None:
    import numpy as np

    arrays = {
        "metadata_json": np.frombuffer(
            json.dumps(
                {
                    "schema_version": 1,
                    "seeds": [1],
                    "unlabeled_pool": "includes_labeled",
                    "test_ref": "test",
                    "train_source_size": 4,
                    "test_source_size": 2,
                    "dataset_fingerprint": "dataset-sha",
                }
            ).encode("utf-8"),
            dtype=np.uint8,
        ),
        "seed_1__train": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "seed_1__val": np.asarray([], dtype=np.int64),
        "seed_1__test": np.asarray([0, 1], dtype=np.int64),
        "seed_1__train_labeled": np.asarray([0, 1], dtype=np.int64),
        "seed_1__train_unlabeled": np.asarray([0, 1, 2, 3], dtype=np.int64),
    }
    artifact = tmp_path / "splits.npz"
    np.savez_compressed(artifact, **arrays)
    raw = _raw_card(method="fixmatch", seeds=[1])
    raw["sampling"] = {
        "plan": {
            "partition": {
                "ordered_indices_artifact": {
                    "path": "${MODSSC_ROOT}/splits.npz",
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "unlabeled_pool": "includes_labeled",
                    "test_ref": "test",
                    "expected_train_size": 4,
                    "expected_val_size": 0,
                    "expected_test_size": 2,
                    "expected_labeled_size": 2,
                    "expected_unlabeled_size": 4,
                    "expected_per_class": 1,
                }
            }
        }
    }
    dataset = SimpleNamespace(
        train=SimpleNamespace(y=np.asarray([0, 1, 0, 1], dtype=np.int64)),
        test=SimpleNamespace(y=np.asarray([0, 1], dtype=np.int64)),
        meta={"dataset_fingerprint": "dataset-sha"},
    )

    assert (
        reproduce._prepare_match_splits(raw, dataset=dataset, repo_root=tmp_path)
        == "match-splits:1-seed(s)"
    )


def test_prepare_calder_prerequisites_dry_and_materialized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    verified: list[Path] = []
    campaign = types.ModuleType("bench.campaign")
    campaign.__path__ = []  # type: ignore[attr-defined]
    protocols = types.ModuleType("bench.campaign.protocols")
    protocols.__path__ = []  # type: ignore[attr-defined]
    calder = types.ModuleType("bench.campaign.protocols.calder")
    calder.__path__ = []  # type: ignore[attr-defined]
    calder_official = types.ModuleType("bench.campaign.protocols.calder.official")
    calder_official.verify_calder_official_assets = lambda path: verified.append(path) or {}
    calder_oracle = types.ModuleType("bench.campaign.protocols.calder.oracle")
    calder_oracle.verify_calder_numerical_oracle = lambda path: verified.append(path) or {}
    calder_artifacts = types.ModuleType("bench.campaign.protocols.calder.artifacts")
    calder_artifacts.prepare_calder_artifact_lock = lambda **_kwargs: {
        "pins": {"graph_fingerprint": "graph-sha"}
    }
    for name, module in {
        "bench.campaign": campaign,
        "bench.campaign.protocols": protocols,
        "bench.campaign.protocols.calder": calder,
        "bench.campaign.protocols.calder.official": calder_official,
        "bench.campaign.protocols.calder.oracle": calder_oracle,
        "bench.campaign.protocols.calder.artifacts": calder_artifacts,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    dry_checks = reproduce._prepare_calder_prerequisites(
        dataset_cache=tmp_path / "datasets",
        dry_run=True,
    )
    assert dry_checks[-1] == "calder-graph-cache:planned"
    assert verified

    monkeypatch.delenv("MODSSC_EXECUTION_JOB_ID", raising=False)
    monkeypatch.setenv("MODSSC_CACHE_ROOT", str(tmp_path))
    checks = reproduce._prepare_calder_prerequisites(
        dataset_cache=tmp_path / "datasets",
        dry_run=False,
    )
    assert checks[-1] == "calder-graph-cache:graph-sha"


def test_prepare_card_dry_run_has_no_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _write_card(tmp_path, "card.yaml", _raw_card(method="fixmatch"))
    card = _card(path)
    card = reproduce.ReproductionCard(
        **{**card.__dict__, "method_id": "fixmatch"},
    )
    monkeypatch.setattr(reproduce, "_require_static_verification", lambda _card: None)
    monkeypatch.setattr(
        reproduce,
        "download_dataset",
        lambda *_args, **_kwargs: pytest.fail("dry-run downloaded a dataset"),
    )
    monkeypatch.setattr(reproduce, "_dataset_integrity_pin", lambda _profile: _demo_pin())

    result = reproduce.prepare_card(card, cache_dir=tmp_path / "cache", dry_run=True)

    assert result.dry_run is True
    assert "dataset:planned" in result.protocol_checks
    assert "match-splits:sha256-authenticated" in result.protocol_checks


def test_run_card_is_fail_closed_then_delegates_to_bench_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    card = _card(tmp_path / "card.yaml")
    blocked = reproduce.VerificationReport(
        card=card,
        issues=(reproduce.VerificationIssue("E_TEST", "x", "blocked"),),
    )
    monkeypatch.setattr(reproduce, "verify_card", lambda _card: blocked)
    with pytest.raises(reproduce.ReproductionRegistryError, match="not self-contained"):
        reproduce.run_card(card)

    ready = reproduce.VerificationReport(card=card, issues=())
    monkeypatch.setattr(reproduce, "verify_card", lambda _card: ready)
    prepared: list[str] = []
    monkeypatch.setattr(
        reproduce,
        "prepare_card",
        lambda *_args, **_kwargs: prepared.append("ready"),
    )
    bench_main = types.ModuleType("bench.main")
    seen: dict[str, str] = {}

    def fake_run_experiment(*_args: object, **_kwargs: object) -> int:
        for name in (
            "MODSSC_CACHE_ROOT",
            "MODSSC_DATASET_CACHE_DIR",
            "MODSSC_GRAPH_CACHE_DIR",
            "MODSSC_GRAPH_VIEWS_CACHE_DIR",
            "MODSSC_OUTPUT_DIR",
            "MODSSC_PREPROCESS_CACHE_DIR",
            "MODSSC_SPLIT_CACHE_DIR",
        ):
            seen[name] = os.environ[name]
        assert Path.cwd() == reproduce.REPO_ROOT
        return 7

    bench_main.run_experiment = fake_run_experiment
    monkeypatch.setitem(sys.modules, "bench.main", bench_main)
    assert reproduce.run_card(card, cache_dir=tmp_path, seed=3) == 7
    assert prepared == ["ready"]
    assert set(seen) == {
        "MODSSC_CACHE_ROOT",
        "MODSSC_DATASET_CACHE_DIR",
        "MODSSC_GRAPH_CACHE_DIR",
        "MODSSC_GRAPH_VIEWS_CACHE_DIR",
        "MODSSC_OUTPUT_DIR",
        "MODSSC_PREPROCESS_CACHE_DIR",
        "MODSSC_SPLIT_CACHE_DIR",
    }


def test_real_dry_run_from_unrelated_cwd_without_modssc_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in tuple(os.environ):
        if name.startswith("MODSSC_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        reproduce,
        "default_dataset_cache_dir",
        lambda: tmp_path / "cache" / "datasets",
    )

    assert reproduce.main(["run", "fixmatch", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "execution plan ready" in output
    assert "scientific-status=not-evaluated" in output
    assert Path.cwd() == tmp_path
    assert not any(name.startswith("MODSSC_") for name in os.environ)


def test_run_card_executes_real_toy_experiment_from_clean_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in tuple(os.environ):
        if name.startswith("MODSSC_"):
            monkeypatch.delenv(name, raising=False)
    source = reproduce.REPO_ROOT / "bench/configs/experiments/toy_inductive.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["run"]["output_dir"] = str(tmp_path / "runs")
    config = tmp_path / "toy.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    card = reproduce.ReproductionCard(
        card_id="test/toy",
        config_path=config,
        method_id="pseudo_label",
        profile="paper:test-toy",
        dataset_id="toy",
        run_name="toy_pseudo_label_numpy",
        repetitions=1,
    )
    ready = reproduce.VerificationReport(card=card, issues=())
    monkeypatch.setattr(reproduce, "verify_card", lambda _card: ready)
    monkeypatch.setattr(reproduce, "prepare_card", lambda *_args, **_kwargs: None)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    code = reproduce.run_card(card, cache_dir=tmp_path / "cache" / "datasets", seed=7)

    assert code == 0
    assert list((tmp_path / "runs").rglob("run.json"))
    assert Path.cwd() == unrelated
    assert not any(name.startswith("MODSSC_") for name in os.environ)


def test_cli_list_show_verify_prepare_and_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "card.yaml"
    path.write_text("paper: card\n", encoding="utf-8")
    card = _card(path)
    monkeypatch.setattr(reproduce, "discover_cards", lambda: (card,))

    assert reproduce.main(["list"]) == 0
    assert "demo/card" in capsys.readouterr().out
    assert reproduce.main(["list", "--method", "other", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []

    assert reproduce.main(["show", "demo/card"]) == 0
    assert "method_id: demo" in capsys.readouterr().out
    assert reproduce.main(["show", "demo/card", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["card_id"] == "demo/card"
    assert reproduce.main(["show", "demo/card", "--raw"]) == 0
    assert capsys.readouterr().out == "paper: card\n"

    ready = reproduce.VerificationReport(card=card, issues=())
    blocked = reproduce.VerificationReport(
        card=card,
        issues=(reproduce.VerificationIssue("E_TEST", "method", "no"),),
    )
    monkeypatch.setattr(reproduce, "verify_card", lambda _card: ready)
    assert reproduce.main(["verify", "demo/card", "--json"]) == 0
    verified = json.loads(capsys.readouterr().out)[0]
    assert verified["execution_ready"] is True
    assert verified["scientific_status"] == "not_evaluated"
    monkeypatch.setattr(reproduce, "verify_card", lambda _card: blocked)
    assert reproduce.main(["verify"]) == 2
    assert "blocked" in capsys.readouterr().out

    prepared = reproduce.PreparationResult(
        "demo/card",
        "toy",
        "/cache",
        "sha",
        "content-sha",
        "paper_identity_authenticated",
        ("dataset:cached",),
        False,
    )
    monkeypatch.setattr(reproduce, "prepare_card", lambda *_args, **_kwargs: prepared)
    assert reproduce.main(["prepare", "demo/card", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["dataset_fingerprint"] == "sha"
    assert reproduce.main(["prepare", "demo/card"]) == 0
    prepared_output = capsys.readouterr().out
    assert "execution inputs prepared" in prepared_output
    assert "scientific-status=not-evaluated" in prepared_output

    assert reproduce.main(["run", "demo/card", "--dry-run"]) == 0
    dry_output = capsys.readouterr().out
    assert "execution plan ready" in dry_output
    assert "scientific-status=not-evaluated" in dry_output

    monkeypatch.setattr(reproduce, "run_card", lambda *_args, **_kwargs: 9)
    assert reproduce.main(["run", "demo/card", "--seed", "4"]) == 9


def test_cli_turns_registry_errors_into_exit_code_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reproduce,
        "discover_cards",
        lambda: (_ for _ in ()).throw(reproduce.ReproductionRegistryError("broken")),
    )
    with pytest.raises(SystemExit) as exc:
        reproduce.main(["list"])
    assert exc.value.code == 2
