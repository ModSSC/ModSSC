from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from tools.replication_audit.calder import artifacts as calder
from tools.replication_audit.calder.artifacts import (
    LOCK_KIND,
    LOCK_SCHEMA_VERSION,
    CalderArtifactError,
    artifact_tree_inventory,
    load_calder_config_family,
    require_execution_site,
    require_scheduled_compute_node,
    seal_calder_artifact_lock,
    verify_artifact_tree,
    verify_calder_artifact_lock,
    write_immutable_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_tree(root: Path, files: dict[str, bytes]) -> None:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _copy_calder_family(root: Path) -> Path:
    destination = root / "bench/configs/reproductions"
    for relative in calder.CALDER_CONFIGS:
        source = REPO_ROOT / "bench/configs/reproductions" / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return destination


def _mutate_yaml(path: Path, mutate) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(raw)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _mutate_all_calder(root: Path, mutate) -> None:
    reproduction_root = root / "bench/configs/reproductions"
    for relative in calder.CALDER_CONFIGS:
        _mutate_yaml(reproduction_root / relative, mutate)


def _valid_lock(tmp_path: Path, monkeypatch) -> dict:
    from modssc.graph.fingerprint import fingerprint_dict
    from modssc.graph.specs import GraphBuilderSpec

    protocol_inputs = tmp_path / "protocol-inputs"
    preprocess = tmp_path / "preprocess"
    graph = tmp_path / "graph"
    preprocess_fingerprint = "preprocess:" + "2" * 64
    prepared_fingerprint = "dataset:" + "1" * 64
    graph_spec = GraphBuilderSpec.from_dict(
        {
            **copy.deepcopy(calder._EXPECTED_GRAPH_SPEC),
            "precomputed_path": str(protocol_inputs / "graph/mnist-vae-knn30.npz"),
        }
    ).to_dict()
    spec_fingerprint = fingerprint_dict(graph_spec)
    graph_fingerprint = fingerprint_dict(
        {
            "dataset_fingerprint": prepared_fingerprint,
            "preprocess_fingerprint": preprocess_fingerprint,
            "spec": graph_spec,
            "seed": 1,
        }
    )
    _write_tree(
        protocol_inputs,
        {
            "MANIFEST.json": b"protocol-inputs",
            "graph/mnist-vae-knn30.npz": b"graph",
        },
    )
    _write_tree(preprocess, {"steps/value.npy": b"features"})
    _write_tree(
        graph,
        {
            "manifest.json": json.dumps(
                {
                    "fingerprint": graph_fingerprint,
                    "dataset_fingerprint": prepared_fingerprint,
                    "preprocess_fingerprint": preprocess_fingerprint,
                    "spec": graph_spec,
                    "spec_fingerprint": spec_fingerprint,
                    "seed": 1,
                }
            ).encode(),
            "edge_index.npy": b"edges",
        },
    )
    evidence = {
        "commit": "official-commit",
        "knn_sha256": "a" * 64,
        "permutations_sha256": "b" * 64,
        "permutations_artifact_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        "bench.campaign.protocols.calder.official.verify_calder_official_assets",
        lambda *_args, **_kwargs: evidence,
    )
    pinned_graph = {
        "spec": copy.deepcopy(calder._EXPECTED_GRAPH_SPEC),
        "expected_fingerprint": graph_fingerprint,
        "expected_preprocess_fingerprint": preprocess_fingerprint,
    }
    effective_payload = {
        "dataset": {"id": "mnist"},
        "sampling": {"seed": 0},
        "preprocess": {"seed": 1},
        "graph": pinned_graph,
    }
    return seal_calder_artifact_lock(
        {
            "schema_version": LOCK_SCHEMA_VERSION,
            "kind": LOCK_KIND,
            "builder": {
                "git_sha": "calder-test-commit",
                "source_identity": {
                    "kind": "git",
                    "sha": "calder-test-commit",
                },
                "environment": {
                    "git_sha": "calder-test-commit",
                    "git_dirty": False,
                },
                "config_files": [],
            },
            "pins": {
                "preprocess_fingerprint": preprocess_fingerprint,
                "graph_fingerprint": graph_fingerprint,
                "official_commit": evidence["commit"],
                "official_knn_sha256": evidence["knn_sha256"],
                "official_permutations_sha256": evidence["permutations_sha256"],
                "permutations_artifact_sha256": evidence["permutations_artifact_sha256"],
            },
            "dataset": {
                "id": "mnist",
                "n_nodes": 70_000,
                "prepared_fingerprint": prepared_fingerprint,
                "official_splits_merged": True,
            },
            "protocol": {
                "graph_seed": 1,
                **effective_payload,
                "effective_sha256": calder._canonical_sha256(effective_payload),
            },
            "artifacts": {
                "protocol_inputs": artifact_tree_inventory(protocol_inputs),
                "preprocess": artifact_tree_inventory(preprocess),
                "graph": artifact_tree_inventory(graph),
            },
        }
    )


def test_calder_family_pins_official_graph_and_flattened_mnist() -> None:
    family = load_calder_config_family(REPO_ROOT)
    assert len(family.files) == 10
    assert family.canonical_raw["preprocess"]["plan"]["output_key"] == "features.X"
    graph = family.canonical_raw["graph"]["spec"]
    assert graph["backend"] == "precomputed"
    assert graph["include_self_in_knn"] is True
    assert graph["precomputed_sha256"] == (
        "5b42bb234888c83eed763958a17fdfb8a55c09a2f0071b55a61635d86dc90db5"
    )


def test_plan_helpers_reject_malformed_inputs() -> None:
    with pytest.raises(CalderArtifactError, match="must be a mapping"):
        calder._mapping(None, name="value")
    with pytest.raises(CalderArtifactError, match="steps must be a list"):
        calder._step_ids({"steps": None})
    with pytest.raises(CalderArtifactError, match="preprocess step must be a mapping"):
        calder._step_ids({"steps": [None]})


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "cannot read Calder configuration"),
        ("bad_yaml", "cannot read Calder configuration"),
        ("bad_root", "root is not a mapping"),
        ("preprocess_diverges", "do not share one preprocess"),
        ("graph_diverges", "do not share one graph"),
        ("dataset", "not pinned to offline MNIST"),
        ("steps", "step sequence differs"),
        ("output", "retain flattened MNIST features"),
        ("preprocess_seed", "seed/fit scope is not frozen"),
        ("preprocess_cache", "preprocessing cache must be enabled"),
        ("graph_cache", "graph cache must be enabled"),
        ("graph_seed", "graph seed/cache policy is not frozen"),
        ("graph_spec", "graph specification differs"),
    ],
)
def test_calder_family_validation_rejects_protocol_drift(tmp_path, case: str, message: str) -> None:
    reproduction_root = _copy_calder_family(tmp_path)
    first = reproduction_root / calder.CALDER_CONFIGS[0]
    if case == "missing":
        first.unlink()
    elif case == "bad_yaml":
        first.write_text("[", encoding="utf-8")
    elif case == "bad_root":
        first.write_text("[]\n", encoding="utf-8")
    elif case == "preprocess_diverges":
        _mutate_yaml(first, lambda raw: raw["preprocess"].update(seed=999))
    elif case == "graph_diverges":
        _mutate_yaml(first, lambda raw: raw["graph"].update(seed=999))
    elif case == "dataset":
        _mutate_yaml(first, lambda raw: raw["dataset"].update(id="fashion_mnist"))
    elif case == "steps":
        _mutate_all_calder(
            tmp_path,
            lambda raw: raw["preprocess"]["plan"]["steps"].pop(0),
        )
    elif case == "output":
        _mutate_all_calder(
            tmp_path,
            lambda raw: raw["preprocess"]["plan"].update(output_key="features.other"),
        )
    elif case == "preprocess_seed":
        _mutate_all_calder(tmp_path, lambda raw: raw["preprocess"].update(seed=2))
    elif case == "preprocess_cache":
        _mutate_all_calder(tmp_path, lambda raw: raw["preprocess"].update(cache=False))
    elif case == "graph_cache":
        _mutate_all_calder(tmp_path, lambda raw: raw["graph"].update(enabled=False))
    elif case == "graph_seed":
        _mutate_all_calder(tmp_path, lambda raw: raw["graph"].update(seed=2))
    else:
        _mutate_all_calder(
            tmp_path,
            lambda raw: raw["graph"]["spec"].update(k=11),
        )

    with pytest.raises(CalderArtifactError, match=message):
        load_calder_config_family(tmp_path)


def test_artifact_inventory_is_stable_and_detects_mutation(tmp_path) -> None:
    root = tmp_path / "artifact"
    _write_tree(root, {"b.bin": b"two", "nested/a.bin": b"one"})
    inventory = artifact_tree_inventory(root)

    assert [entry["path"] for entry in inventory["files"]] == ["b.bin", "nested/a.bin"]
    assert len(inventory["tree_sha256"]) == 64
    verify_artifact_tree(inventory)

    (root / "b.bin").write_bytes(b"changed")
    with pytest.raises(CalderArtifactError, match="differs from its SHA lock"):
        verify_artifact_tree(inventory)
    with pytest.raises(CalderArtifactError, match="inventory has no root"):
        verify_artifact_tree({})


def test_artifact_inventory_rejects_partial_graph_work(tmp_path) -> None:
    root = tmp_path / "graph"
    _write_tree(root, {"manifest.json": b"{}", "_work/knn/chunk.npz": b"partial"})
    with pytest.raises(CalderArtifactError, match="incomplete"):
        artifact_tree_inventory(root)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("root_symlink", "root must not be a symlink"),
        ("missing", "root is missing"),
        ("file_root", "root is not a directory"),
        ("nested_symlink", "contains a symlink"),
        ("special", "contains a special file"),
        ("empty", "tree is empty"),
        ("temporary", "tree is incomplete"),
    ],
)
def test_artifact_inventory_rejects_uncommitted_trees(tmp_path, case: str, message: str) -> None:
    root = tmp_path / "artifact"
    if case == "root_symlink":
        target = tmp_path / "target"
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)
    elif case == "missing":
        pass
    elif case == "file_root":
        root.write_bytes(b"file")
    elif case == "nested_symlink":
        root.mkdir()
        (root / "target").write_bytes(b"value")
        (root / "link").symlink_to(root / "target")
    elif case == "special":
        root.mkdir()
        os.mkfifo(root / "fifo")
    elif case == "empty":
        root.mkdir()
    else:
        _write_tree(root, {"unfinished.tmp": b"partial"})

    with pytest.raises(CalderArtifactError, match=message):
        artifact_tree_inventory(root)


def test_immutable_lock_is_idempotent_but_never_replaced(tmp_path) -> None:
    path = tmp_path / "lock.json"
    write_immutable_json(path, {"value": 1})
    write_immutable_json(path, {"value": 1})
    with pytest.raises(CalderArtifactError, match="refusing to replace"):
        write_immutable_json(path, {"value": 2})


def test_immutable_lock_handles_concurrent_publish(tmp_path, monkeypatch) -> None:
    same = tmp_path / "same.json"

    def publish_same(source, destination) -> None:
        Path(destination).write_bytes(Path(source).read_bytes())
        raise FileExistsError

    monkeypatch.setattr(calder.os, "link", publish_same)
    write_immutable_json(same, {"value": 1})

    different = tmp_path / "different.json"

    def publish_different(_source, destination) -> None:
        Path(destination).write_text("{}\n", encoding="utf-8")
        raise FileExistsError

    monkeypatch.setattr(calder.os, "link", publish_different)
    with pytest.raises(CalderArtifactError, match="concurrent immutable lock differs"):
        write_immutable_json(different, {"value": 1})


def test_immutable_lock_cleans_up_when_temporary_creation_fails(tmp_path, monkeypatch) -> None:
    def fail_temporary(**_kwargs):
        raise OSError("no temporary file")

    monkeypatch.setattr(calder.tempfile, "NamedTemporaryFile", fail_temporary)
    with pytest.raises(OSError, match="no temporary"):
        write_immutable_json(tmp_path / "lock.json", {"value": 1})


def test_immutable_effective_config_is_idempotent_but_never_replaced(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    calder._write_immutable_text(path, "value: 1\n")
    calder._write_immutable_text(path, "value: 1\n")
    with pytest.raises(CalderArtifactError, match="effective configuration"):
        calder._write_immutable_text(path, "value: 2\n")


def test_immutable_effective_config_handles_concurrent_publish(tmp_path, monkeypatch) -> None:
    same = tmp_path / "same.yaml"

    def publish_same(source, destination) -> None:
        Path(destination).write_bytes(Path(source).read_bytes())
        raise FileExistsError

    monkeypatch.setattr(calder.os, "link", publish_same)
    calder._write_immutable_text(same, "value: 1\n")

    different = tmp_path / "different.yaml"

    def publish_different(_source, destination) -> None:
        Path(destination).write_text("other: true\n", encoding="utf-8")
        raise FileExistsError

    monkeypatch.setattr(calder.os, "link", publish_different)
    with pytest.raises(CalderArtifactError, match="concurrent effective configuration"):
        calder._write_immutable_text(different, "value: 1\n")


def test_json_reader_rejects_missing_invalid_and_non_mapping_files(tmp_path) -> None:
    with pytest.raises(CalderArtifactError, match="invalid Calder artifact lock"):
        calder._read_json(tmp_path / "missing.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(CalderArtifactError, match="invalid Calder artifact lock"):
        calder._read_json(invalid)
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]\n", encoding="utf-8")
    with pytest.raises(CalderArtifactError, match="root is not a mapping"):
        calder._read_json(sequence)


def test_complete_lock_rehashes_all_three_artifacts(tmp_path, monkeypatch) -> None:
    lock = _valid_lock(tmp_path, monkeypatch)
    verify_calder_artifact_lock(lock)

    changed = copy.deepcopy(lock)
    changed["pins"]["graph_fingerprint"] = "4" * 64
    with pytest.raises(CalderArtifactError, match="lock SHA-256 differs"):
        verify_calder_artifact_lock(changed)


def test_lock_canonicalizes_equivalent_precomputed_paths(tmp_path, monkeypatch) -> None:
    actual_root = tmp_path / "actual"
    actual_root.mkdir()
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(actual_root, target_is_directory=True)

    lock = _valid_lock(alias_root, monkeypatch)

    verify_calder_artifact_lock(lock)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("schema", "invalid Calder artifact lock schema"),
        ("missing_hash", "lock SHA-256 differs"),
        ("pins", "pins must be a mapping"),
        ("artifacts", "artifacts must be a mapping"),
        ("artifact_set", "must contain protocol inputs, preprocess, and graph"),
        ("inventory", "artifact inventory must be a mapping"),
        ("graph_pin", "graph cache manifest fingerprint differs"),
        ("preprocess_pin", "graph cache preprocess fingerprint differs"),
        ("official_commit", "official GraphLearning commit differs"),
        ("official_knn", "official GraphLearning kNN SHA-256 differs"),
        ("official_permutations", "official GraphLearning permutations SHA-256 differs"),
        ("safe_permutations", "safe permutation artifact SHA-256 differs"),
    ],
)
def test_lock_validation_rejects_schema_and_manifest_drift(
    tmp_path, monkeypatch, case: str, message: str
) -> None:
    lock = _valid_lock(tmp_path, monkeypatch)
    if case == "schema":
        lock["schema_version"] = 99
    elif case == "missing_hash":
        lock.pop("lock_sha256")
    elif case == "pins":
        lock["pins"] = None
        lock = seal_calder_artifact_lock(lock)
    elif case == "artifacts":
        lock["artifacts"] = None
        lock = seal_calder_artifact_lock(lock)
    elif case == "artifact_set":
        lock["artifacts"].pop("preprocess")
        lock = seal_calder_artifact_lock(lock)
    elif case == "inventory":
        lock["artifacts"]["preprocess"] = None
        lock = seal_calder_artifact_lock(lock)
    elif case == "graph_pin":
        lock["pins"]["graph_fingerprint"] = "other"
        lock = seal_calder_artifact_lock(lock)
    elif case == "preprocess_pin":
        lock["pins"]["preprocess_fingerprint"] = "other"
        lock = seal_calder_artifact_lock(lock)
    elif case == "official_commit":
        lock["pins"]["official_commit"] = "other"
        lock = seal_calder_artifact_lock(lock)
    elif case == "official_knn":
        lock["pins"]["official_knn_sha256"] = "other"
        lock = seal_calder_artifact_lock(lock)
    elif case == "official_permutations":
        lock["pins"]["official_permutations_sha256"] = "other"
        lock = seal_calder_artifact_lock(lock)
    else:
        lock["pins"]["permutations_artifact_sha256"] = "other"
        lock = seal_calder_artifact_lock(lock)

    with pytest.raises(CalderArtifactError, match=message):
        verify_calder_artifact_lock(lock)


def test_path_environment_and_runtime_guards(tmp_path, monkeypatch) -> None:
    inside = tmp_path / "inside"
    assert calder._require_under(inside, tmp_path, name="value") == inside.resolve()
    with pytest.raises(CalderArtifactError, match="must be below"):
        calder._require_under(tmp_path.parent, tmp_path, name="value")

    monkeypatch.delenv("CALDER_TEST_PATH", raising=False)
    with pytest.raises(CalderArtifactError, match="CALDER_TEST_PATH is required"):
        calder._required_environment_path("CALDER_TEST_PATH")
    monkeypatch.setenv("CALDER_TEST_PATH", str(inside))
    assert calder._required_environment_path("CALDER_TEST_PATH") == inside

    monkeypatch.setattr(
        "bench.utils.runtime.collect_runtime_versions",
        lambda **_kwargs: {"git_sha": None, "git_dirty": False},
    )
    with pytest.raises(CalderArtifactError, match="clean Git snapshot"):
        calder._runtime_provenance(tmp_path)
    monkeypatch.setattr(
        "bench.utils.runtime.collect_runtime_versions",
        lambda **_kwargs: {"git_sha": "abc", "git_dirty": False},
    )
    git_runtime = calder._runtime_provenance(tmp_path)
    assert git_runtime["git_sha"] == "abc"
    assert git_runtime["source_identity"] == {"kind": "git", "sha": "abc"}
    monkeypatch.setattr(
        "bench.utils.runtime.collect_runtime_versions",
        lambda **_kwargs: {
            "git_sha": None,
            "git_dirty": None,
            "distribution_sha256": "d" * 64,
        },
    )
    wheel_runtime = calder._runtime_provenance(tmp_path)
    assert wheel_runtime["source_identity"] == {
        "kind": "installed_distribution",
        "sha256": "d" * 64,
    }


def test_exclusive_preparation_lock_rejects_a_second_owner(tmp_path) -> None:
    output = tmp_path / "lock.json"
    with (
        calder._exclusive_preparation_lock(output),
        pytest.raises(CalderArtifactError, match="another Calder artifact preparation"),
        calder._exclusive_preparation_lock(output),
    ):
        pytest.fail("the second owner must not acquire the lock")


def _run_mocked_preparation(monkeypatch, tmp_path: Path, *, scenario: str, explicit: bool):
    import modssc.data_loader as data_loader
    from bench.orchestrators import dataset as dataset_orchestrator
    from bench.orchestrators import graph as graph_orchestrator
    from bench.orchestrators import preprocess as preprocess_orchestrator
    from bench.orchestrators import sampling as sampling_orchestrator
    from bench.schema import ExperimentConfig
    from bench.utils import io as bench_io
    from modssc.graph.fingerprint import fingerprint_dict
    from modssc.graph.specs import GraphBuilderSpec

    scratch = tmp_path / "scratch"
    dataset_cache = scratch / "modssc_cache/datasets"
    preprocess_base = scratch / "modssc_cache/preprocess/calder2020-mnist-table1"
    graph_base = scratch / "modssc_cache/graph/calder2020-mnist-table1"
    preprocess_cache = preprocess_base / "dataset-fingerprint"
    preprocess_fingerprint = "preprocess:" + "b" * 64
    official_root = tmp_path / "protocol-inputs"
    for directory in (dataset_cache, preprocess_cache, graph_base, official_root):
        directory.mkdir(parents=True, exist_ok=True)
    _write_tree(
        official_root,
        {
            "MANIFEST.json": b"protocol-inputs",
            "graph/mnist-vae-knn30.npz": b"graph",
        },
    )
    _write_tree(preprocess_cache, {"steps/features.npy": b"features"})

    family = load_calder_config_family(REPO_ROOT)
    canonical = family.canonical_raw
    run_seed = 7
    sampling_seed = 1 if explicit else None
    preprocess_seed = 1 if explicit else None
    graph_seed = 1 if explicit else None
    graph_cfg = SimpleNamespace(
        cache_dir=str(graph_base),
        seed=graph_seed,
        spec=copy.deepcopy(canonical["graph"]["spec"]),
        cache=True,
    )
    graph_cfg.spec["precomputed_path"] = (
        None if scenario == "official_path" else str(official_root / "graph/mnist-vae-knn30.npz")
    )
    prepared_fingerprint_value = "prepared-fingerprint"
    canonical_graph_spec = GraphBuilderSpec.from_dict(graph_cfg.spec).to_dict()
    graph_spec_fingerprint = fingerprint_dict(canonical_graph_spec)
    graph_fingerprint = fingerprint_dict(
        {
            "dataset_fingerprint": prepared_fingerprint_value,
            "preprocess_fingerprint": preprocess_fingerprint,
            "spec": canonical_graph_spec,
            "seed": 1 if explicit else run_seed,
        }
    )
    graph_cache = graph_base / graph_fingerprint
    _write_tree(
        graph_cache,
        {
            "manifest.json": json.dumps(
                {
                    "fingerprint": graph_fingerprint,
                    "dataset_fingerprint": prepared_fingerprint_value,
                    "preprocess_fingerprint": preprocess_fingerprint,
                    "spec": canonical_graph_spec,
                    "spec_fingerprint": graph_spec_fingerprint,
                    "seed": 1 if explicit else run_seed,
                }
            ).encode(),
            "edge_index.npy": b"edges",
        },
    )
    if scenario == "graph_none":
        graph_cfg = None
    cfg = SimpleNamespace(
        run=SimpleNamespace(seed=run_seed),
        dataset=SimpleNamespace(
            id="mnist",
            cache_dir=None if scenario == "cache_missing" else str(dataset_cache),
            options={},
        ),
        sampling=SimpleNamespace(
            seed=sampling_seed,
            plan=copy.deepcopy(canonical["sampling"]["plan"]),
        ),
        preprocess=SimpleNamespace(
            seed=preprocess_seed,
            fit_on="train",
            cache=True,
            cache_dir=str(preprocess_base),
            plan=copy.deepcopy(canonical["preprocess"]["plan"]),
        ),
        graph=graph_cfg,
    )

    source_fingerprint = None if scenario == "source_fingerprint" else "source-fingerprint"
    source_dataset = SimpleNamespace(
        meta={"dataset_fingerprint": source_fingerprint},
        train=SimpleNamespace(y=np.zeros(1, dtype=np.int64)),
        test=SimpleNamespace(),
    )
    prepared_fingerprint = (
        None if scenario == "prepared_fingerprint" else prepared_fingerprint_value
    )
    prepared_dataset = SimpleNamespace(
        meta={"dataset_fingerprint": prepared_fingerprint},
        train=SimpleNamespace(y=np.zeros(70_000, dtype=np.int8)),
        test=SimpleNamespace() if scenario == "pool" else None,
    )

    production_result = SimpleNamespace(
        preprocess_fingerprint=preprocess_fingerprint,
        cache_dir=None if scenario == "preprocess_cache_dir" else str(preprocess_cache),
    )

    first_graph_fingerprint = None if scenario == "graph_fingerprint" else graph_fingerprint
    graph_result = SimpleNamespace(
        meta={"fingerprint": first_graph_fingerprint},
        n_nodes=1 if scenario == "graph_shape" else 70_000,
        edge_index=np.zeros((2, 1), dtype=np.int64),
    )
    verified_graph = SimpleNamespace(
        meta={"fingerprint": "different" if scenario == "graph_reload" else graph_fingerprint}
    )
    graph_results = iter((graph_result, verified_graph))
    preprocess_calls = []
    graph_calls = []

    def run_preprocess(*args, **kwargs):
        preprocess_calls.append((args, kwargs))
        return production_result

    def build_graph(*args, **kwargs):
        graph_calls.append((args, kwargs))
        return next(graph_results)

    monkeypatch.setenv("MODSSC_SCRATCH", str(scratch))
    monkeypatch.setattr(
        calder,
        "_runtime_provenance",
        lambda _root: {
            "git_sha": "abc",
            "git_dirty": False,
            "source_identity": {"kind": "git", "sha": "abc"},
        },
    )
    monkeypatch.setattr(calder, "load_calder_config_family", lambda _root: family)
    monkeypatch.setattr(bench_io, "load_yaml", lambda _path: {})
    monkeypatch.setattr(
        ExperimentConfig,
        "from_dict",
        classmethod(lambda _cls, _raw: cfg),
    )
    monkeypatch.setattr(
        dataset_orchestrator,
        "load",
        lambda _cfg: (source_dataset, {"id": "mnist"}),
    )
    monkeypatch.setattr(
        data_loader,
        "verify_dataset_content",
        lambda *_args, **_kwargs: {"content_sha256": "content"},
    )
    official_evidence = {
        "commit": "official-commit",
        "knn_sha256": "a" * 64,
        "permutations_sha256": "b" * 64,
        "permutations_artifact_sha256": "c" * 64,
    }

    def verify_official(*_args, **_kwargs):
        if scenario == "official_verify":
            raise RuntimeError("bad official artifact")
        return official_evidence

    monkeypatch.setattr(
        "bench.campaign.protocols.calder.official.verify_calder_official_assets",
        verify_official,
    )
    monkeypatch.setattr(
        sampling_orchestrator,
        "prepare_dataset",
        lambda _dataset, **_kwargs: prepared_dataset,
    )
    monkeypatch.setattr(
        sampling_orchestrator,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        preprocess_orchestrator,
        "resolve_fit_indices",
        lambda **_kwargs: np.arange(10, dtype=np.int64),
    )
    monkeypatch.setattr(preprocess_orchestrator, "run", run_preprocess)
    monkeypatch.setattr(graph_orchestrator, "build", build_graph)

    lock = calder._prepare_calder_artifacts(tmp_path)
    return lock, preprocess_calls, graph_calls


@pytest.mark.parametrize("explicit", [False, True])
def test_prepare_calder_artifacts_pins_official_source_and_verifies_graph(
    tmp_path, monkeypatch, explicit: bool
) -> None:
    lock, preprocess_calls, graph_calls = _run_mocked_preparation(
        monkeypatch,
        tmp_path,
        scenario="ok",
        explicit=explicit,
    )

    expected_seed = 1 if explicit else 7
    assert lock["pins"]["official_commit"] == "official-commit"
    assert lock["dataset"]["n_nodes"] == 70_000
    assert [call[1]["seed"] for call in preprocess_calls] == [expected_seed]
    assert graph_calls[0][1]["require_cache_hit"] is False
    assert graph_calls[1][1]["require_cache_hit"] is True
    assert [call[1]["seed"] for call in graph_calls] == [expected_seed, expected_seed]
    verify_calder_artifact_lock(lock)


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("graph_none", "has no graph section"),
        ("cache_missing", "caches must be explicit"),
        ("source_fingerprint", "no source fingerprint"),
        ("prepared_fingerprint", "no fingerprint"),
        ("pool", "70,000-node MNIST pool"),
        ("official_path", "has no official precomputed_path"),
        ("official_verify", "Calder protocol inputs failed verification"),
        ("graph_fingerprint", "graph did not report a fingerprint"),
        ("graph_shape", "empty or has the wrong node count"),
        ("graph_reload", "reloaded a different fingerprint"),
        ("preprocess_cache_dir", "did not report its cache directory"),
    ],
)
def test_prepare_calder_artifacts_fails_closed(
    tmp_path, monkeypatch, scenario: str, message: str
) -> None:
    with pytest.raises(CalderArtifactError, match=message):
        _run_mocked_preparation(
            monkeypatch,
            tmp_path,
            scenario=scenario,
            explicit=True,
        )


def test_prepare_and_verify_lock_files_are_idempotent(tmp_path, monkeypatch) -> None:
    local_root = tmp_path / "local"
    work = local_root / "work"
    scratch = local_root / "scratch"
    work.mkdir(parents=True)
    scratch.mkdir()
    output = work / "artifacts/calder.json"
    lock = _valid_lock(tmp_path / "source", monkeypatch)
    monkeypatch.setenv("MODSSC_WORK", str(work))
    monkeypatch.setenv("MODSSC_SCRATCH", str(scratch))
    monkeypatch.setattr(calder.socket, "gethostname", lambda: "local-test-host")
    monkeypatch.setattr(calder, "_prepare_calder_artifacts", lambda _root: lock)

    assert (
        calder.prepare_calder_artifact_lock(
            repo_root=tmp_path,
            output=output,
            execution_site="local",
            local_root=local_root,
        )
        == lock
    )
    monkeypatch.setattr(
        calder,
        "_prepare_calder_artifacts",
        lambda _root: pytest.fail("an existing verified lock must be reused"),
    )
    assert (
        calder.prepare_calder_artifact_lock(
            repo_root=tmp_path,
            output=output,
            execution_site="local",
            local_root=local_root,
        )
        == lock
    )
    assert (
        calder.verify_calder_artifact_lock_file(
            output,
            execution_site="local",
            local_root=local_root,
        )
        == lock
    )


def _lock_with_source_family(tmp_path, monkeypatch) -> tuple[Path, dict]:
    repo_root = tmp_path / "repo"
    _copy_calder_family(repo_root)
    family = load_calder_config_family(repo_root)
    lock = _valid_lock(tmp_path / "artifacts", monkeypatch)
    lock["builder"]["config_files"] = list(family.files)
    return repo_root, seal_calder_artifact_lock(lock)


def test_materialize_effective_configs_pins_verified_cache_immutably(tmp_path, monkeypatch) -> None:
    repo_root, lock = _lock_with_source_family(tmp_path, monkeypatch)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    output = repo_root / "bench/generated/calder-test"

    manifest = calder.materialize_calder_effective_configs(
        repo_root=repo_root,
        lock=lock,
        output_dir=output,
    )

    assert manifest["kind"] == calder.EFFECTIVE_CONFIG_KIND
    assert manifest["artifact_lock_sha256"] == lock["lock_sha256"]
    assert manifest["artifact_builder"] == {
        "kind": "git",
        "sha": "calder-test-commit",
    }
    assert manifest["source_configs"] == lock["builder"]["config_files"]
    assert len(manifest["configs"]) == 10
    assert len({record["sha256"] for record in manifest["configs"]}) == 10
    assert all(
        record["repo_path"].startswith("bench/generated/calder-test/")
        for record in manifest["configs"]
    )
    for relative in calder.CALDER_CONFIGS:
        raw = yaml.safe_load((output / relative).read_text(encoding="utf-8"))
        assert raw["graph"]["require_cache_hit"] is True
        assert raw["graph"]["expected_fingerprint"] == lock["pins"]["graph_fingerprint"]
        assert (
            raw["graph"]["expected_preprocess_fingerprint"]
            == lock["pins"]["preprocess_fingerprint"]
        )
        assert raw["graph"]["spec"] == {
            **calder._EXPECTED_GRAPH_SPEC,
            "radius": None,
            "weights": {"kind": "knn_gaussian", "sigma": None},
            "n_anchors": None,
            "anchors_k": 5,
            "anchors_method": "random",
            "candidate_limit": 1000,
            "faiss_exact": False,
            "faiss_hnsw_m": 32,
            "faiss_ef_search": 64,
            "faiss_ef_construction": 200,
        }
    assert json.loads((output / "MANIFEST.json").read_text(encoding="utf-8")) == manifest
    monkeypatch.delenv("MODSSC_ROOT", raising=False)
    from bench.utils.io import load_yaml

    effective = load_yaml(output / calder.CALDER_CONFIGS[0])
    assert effective["sampling"]["plan"]["labeling"]["fixed_indices_artifact"]["path"].startswith(
        str(repo_root)
    )
    assert effective["graph"]["spec"]["precomputed_path"].startswith(str(repo_root))

    assert (
        calder.materialize_calder_effective_configs(
            repo_root=repo_root,
            lock=lock,
            output_dir=output,
        )
        == manifest
    )

    first = output / calder.CALDER_CONFIGS[0]
    first.write_text("changed: true\n", encoding="utf-8")
    with pytest.raises(CalderArtifactError, match="effective configuration"):
        calder.materialize_calder_effective_configs(
            repo_root=repo_root,
            lock=lock,
            output_dir=output,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda spec: spec.update({"unknown_scientific_knob": 999}),
        lambda spec: spec.update({"self_loops": "false"}),
    ],
)
def test_materialized_graph_spec_rejects_unverified_fields(mutation) -> None:
    raw = dict(calder._EXPECTED_GRAPH_SPEC)
    mutation(raw)
    with pytest.raises(CalderArtifactError, match="differs from the frozen protocol"):
        calder.materialized_calder_graph_spec(raw)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("builder", "builder must be a mapping"),
        ("source_identity", "builder identity is inconsistent"),
        ("source_drift", "differ from the artifact builder snapshot"),
    ],
)
def test_materialize_effective_configs_rejects_unbound_source_snapshot(
    tmp_path, monkeypatch, case: str, message: str
) -> None:
    repo_root, lock = _lock_with_source_family(tmp_path, monkeypatch)
    if case == "builder":
        lock["builder"] = None
    elif case == "source_identity":
        lock["builder"]["source_identity"] = {"kind": "unknown"}
    else:
        first = repo_root / "bench/configs/reproductions" / calder.CALDER_CONFIGS[0]
        _mutate_yaml(
            first,
            lambda raw: raw["method"]["params"].update(cg_tol=9.0e-5),
        )
    lock = seal_calder_artifact_lock(lock)

    with pytest.raises(CalderArtifactError, match=message):
        calder.materialize_calder_effective_configs(
            repo_root=repo_root,
            lock=lock,
            output_dir=repo_root / "bench/generated/calder-test",
        )


def test_materialize_effective_configs_must_stay_inside_repo_generated(
    tmp_path, monkeypatch
) -> None:
    repo_root, lock = _lock_with_source_family(tmp_path, monkeypatch)
    with pytest.raises(CalderArtifactError, match="must be below"):
        calder.materialize_calder_effective_configs(
            repo_root=repo_root,
            lock=lock,
            output_dir=tmp_path / "outside",
        )


def test_cli_prepare_verify_and_error_paths(tmp_path, monkeypatch, capsys) -> None:
    summary_lock = {
        "lock_sha256": "digest",
        "pins": {"graph_fingerprint": "graph"},
        "graph": {"n_nodes": 70_000},
    }
    local_root = tmp_path / "local"
    work = local_root / "work"
    scratch = local_root / "scratch"
    repo_root = work / "repo"
    repo_root.mkdir(parents=True)
    scratch.mkdir()
    monkeypatch.setenv("MODSSC_WORK", str(work))
    monkeypatch.setenv("MODSSC_SCRATCH", str(scratch))
    calls: dict[str, object] = {}

    def prepare(**kwargs):
        calls["prepare"] = kwargs
        return summary_lock

    monkeypatch.setattr(calder, "prepare_calder_artifact_lock", prepare)
    assert (
        calder.main(
            [
                "prepare",
                "--repo-root",
                str(repo_root),
                "--output",
                str(work / "lock.json"),
                "--execution-site",
                "local",
                "--local-root",
                str(local_root),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["lock_sha256"] == "digest"
    assert calls["prepare"] == {
        "repo_root": repo_root,
        "output": work / "lock.json",
        "execution_site": "local",
        "local_root": local_root,
    }

    def verify(path, **kwargs):
        calls.setdefault("verify", []).append((path, kwargs))
        return summary_lock

    monkeypatch.setattr(calder, "verify_calder_artifact_lock_file", verify)
    assert (
        calder.main(
            [
                "verify",
                "--output",
                str(work / "lock.json"),
                "--execution-site",
                "local",
                "--local-root",
                str(local_root),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["graph"]["n_nodes"] == 70_000

    materialized = {
        "kind": calder.EFFECTIVE_CONFIG_KIND,
        "configs": [{"path": "laplace.yaml"}],
    }
    monkeypatch.setattr(
        calder,
        "materialize_calder_effective_configs",
        lambda **_kwargs: materialized,
    )
    assert (
        calder.main(
            [
                "materialize",
                "--repo-root",
                str(repo_root),
                "--lock",
                str(work / "lock.json"),
                "--output-dir",
                str(repo_root / "bench/generated/calder-test"),
                "--execution-site",
                "local",
                "--local-root",
                str(local_root),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == materialized
    verify_calls = calls["verify"]
    assert isinstance(verify_calls, list)
    assert verify_calls == [
        (
            work / "lock.json",
            {"execution_site": "local", "local_root": local_root},
        ),
        (
            work / "lock.json",
            {"execution_site": "local", "local_root": local_root},
        ),
    ]

    def fail_prepare(**_kwargs):
        raise CalderArtifactError("deterministic failure")

    monkeypatch.setattr(calder, "prepare_calder_artifact_lock", fail_prepare)
    with pytest.raises(SystemExit) as exc_info:
        calder.main(
            [
                "prepare",
                "--repo-root",
                str(tmp_path),
                "--output",
                str(tmp_path / "lock.json"),
            ]
        )
    assert exc_info.value.code == 2
    assert "deterministic failure" in capsys.readouterr().err


def test_compute_guard_rejects_frontend_and_hostname_mismatch(tmp_path, monkeypatch) -> None:
    with pytest.raises(CalderArtifactError, match="scheduled compute allocation"):
        require_scheduled_compute_node(environ={}, hostname="slurm-gpu1")
    with pytest.raises(CalderArtifactError, match="allocated compute node"):
        require_scheduled_compute_node(
            environ={
                "MODSSC_EXECUTION_JOB_ID": "12",
                "MODSSC_EXECUTION_NODE": "gpu-node",
            },
            hostname="slurm-gpu1",
        )
    require_scheduled_compute_node(
        environ={
            "MODSSC_EXECUTION_JOB_ID": "12",
            "MODSSC_EXECUTION_NODE": "gpu-node",
        },
        hostname="gpu-node.site-specific.fr",
    )
    monkeypatch.setenv("MODSSC_EXECUTION_JOB_ID", "12")
    monkeypatch.setenv("MODSSC_EXECUTION_NODE", "gpu-node")
    monkeypatch.setattr(calder.socket, "gethostname", lambda: "gpu-node.site-specific.fr")
    require_scheduled_compute_node()


def test_local_execution_site_requires_explicit_containment_and_refuses_scheduled_allocation(
    tmp_path,
) -> None:
    local_root = tmp_path / "local"
    work = local_root / "work"
    scratch = local_root / "scratch"
    work.mkdir(parents=True)
    scratch.mkdir()
    environ = {
        "MODSSC_WORK": str(work),
        "MODSSC_SCRATCH": str(scratch),
    }

    require_execution_site(
        "local",
        local_root=local_root,
        environ=environ,
        hostname="melvin-mac.local",
    )
    with pytest.raises(CalderArtifactError, match="--local-root is required"):
        require_execution_site("local", environ=environ, hostname="melvin-mac.local")
    with pytest.raises(CalderArtifactError, match="MODSSC_WORK must be below"):
        require_execution_site(
            "local",
            local_root=local_root,
            environ={**environ, "MODSSC_WORK": str(tmp_path / "outside")},
            hostname="melvin-mac.local",
        )
    with pytest.raises(CalderArtifactError, match="refusing local Calder execution"):
        require_execution_site(
            "local",
            local_root=local_root,
            environ={
                **environ,
                "MODSSC_EXECUTION_JOB_ID": "12",
                "MODSSC_EXECUTION_NODE": "gpu-node",
            },
            hostname="gpu-node.site-specific.fr",
        )
    with pytest.raises(CalderArtifactError, match="unknown Calder execution site"):
        require_execution_site(
            "regional",  # type: ignore[arg-type]
            local_root=local_root,
            environ=environ,
            hostname="melvin-mac.local",
        )


def test_optional_scheduler_dispatcher_exposes_calder_operations_on_compute_nodes() -> None:
    script = (REPO_ROOT / "tools/hpc/slurm/run-operation.sh").read_text(encoding="utf-8")
    assert "SLURM_JOB_ID" in script
    assert "SLURMD_NODENAME" in script
    assert '"$MODSSC_PYTHON"' in script
    assert "calder-prepare)" in script
    assert "calder-verify)" in script
    assert "calder-materialize)" in script
    assert "-m tools.replication_audit.calder.artifacts prepare" in script
    assert "-m tools.replication_audit.calder.artifacts verify" in script
    assert "-m tools.replication_audit.calder.artifacts materialize" in script
    assert "pip install" not in script
