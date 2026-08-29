from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import modssc.graph.cache as cache_module
from modssc.graph.artifacts import DatasetViews, GraphArtifact
from modssc.graph.cache import GraphCache, GraphCacheError, ViewsCache
from modssc.graph.fingerprint import fingerprint_array


def _graph(*, weighted: bool = False) -> GraphArtifact:
    return GraphArtifact(
        n_nodes=3,
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        edge_weight=(np.array([0.25, 0.75], dtype=np.float32) if weighted else None),
    )


def _views(*, meta: dict[str, Any] | None = None) -> DatasetViews:
    return DatasetViews(
        views={"attr": np.arange(6, dtype=np.float32).reshape(3, 2)},
        y=np.zeros(3, dtype=np.int64),
        meta={} if meta is None else meta,
    )


def _manifest(entry: Path) -> dict[str, Any]:
    return json.loads((entry / "manifest.json").read_text())


def _rewrite_manifest(entry: Path, mutate) -> dict[str, Any]:
    manifest = _manifest(entry)
    mutate(manifest)
    envelope = manifest.get("_cache")
    if isinstance(envelope, dict):
        envelope["manifest_sha256"] = cache_module._manifest_sha256(manifest)
    cache_module._safe_write_json(entry / "manifest.json", manifest)
    return manifest


def _set_nested(manifest: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current: dict[str, Any] = manifest
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def _saved_graph(tmp_path: Path, name: str, *, weighted: bool = False) -> tuple[GraphCache, Path]:
    cache = GraphCache(root=tmp_path / name)
    entry = cache.save(fingerprint="fp", graph=_graph(weighted=weighted), manifest={})
    return cache, entry


def _saved_views(tmp_path: Path, name: str) -> tuple[ViewsCache, Path, DatasetViews]:
    views = _views()
    cache = ViewsCache(root=tmp_path / name)
    entry = cache.save(fingerprint="fp", views=views, manifest={})
    return cache, entry, views


def test_fingerprint_array_rejects_object_content() -> None:
    with pytest.raises(TypeError, match="Object arrays"):
        fingerprint_array(np.array([[object()]], dtype=object))


def test_low_level_cache_helper_error_branches(tmp_path, monkeypatch) -> None:
    assert cache_module._manifest_sha256({"plain": True})
    assert cache_module.array_content_sha256(np.arange(3, dtype=np.int64))

    with monkeypatch.context() as scoped:
        scoped.setattr(cache_module.os, "name", "nt")
        cache_module._fsync_directory(tmp_path)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            cache_module.os, "open", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError())
        )
        cache_module._fsync_directory(tmp_path)

    manifest_target = tmp_path / "manifest-target.json"
    manifest_target.write_text("{}")
    manifest_link = tmp_path / "manifest-link.json"
    try:
        manifest_link.symlink_to(manifest_target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(GraphCacheError, match="symlinked cache manifest"):
        cache_module._safe_read_json(manifest_link)

    def missing_distribution(name: str) -> str:
        raise cache_module.importlib.metadata.PackageNotFoundError(name)

    with monkeypatch.context() as scoped:
        scoped.setattr(cache_module.importlib.metadata, "version", missing_distribution)
        assert cache_module._distribution_version("missing-a", "missing-b") is None

    monkeypatch.setattr(cache_module, "_distribution_version", lambda *_names: "test-version")
    identity = cache_module.graph_implementation_identity(component="construction", backend="torch")
    assert identity["dependencies"]["torch"] == "test-version"
    faiss_identity = cache_module.graph_implementation_identity(
        component="construction", backend="faiss"
    )
    assert faiss_identity["dependencies"]["faiss"] == "test-version"
    annoy_identity = cache_module.graph_implementation_identity(
        component="construction", backend="annoy"
    )
    assert annoy_identity["dependencies"]["annoy"] == "test-version"

    with pytest.raises(GraphCacheError, match="Object arrays"):
        cache_module._array_descriptor(np.array([object()], dtype=object))

    data_target = tmp_path / "data-target.bin"
    data_target.write_bytes(b"payload")
    data_link = tmp_path / "data-link.bin"
    data_link.symlink_to(data_target)
    with pytest.raises(GraphCacheError, match="symlinked cache file"):
        cache_module._file_descriptor(data_link)
    with pytest.raises(GraphCacheError, match="Unable to authenticate"):
        cache_module._file_descriptor(tmp_path / "missing.bin")

    actual = data_target.stat()
    changed = SimpleNamespace(
        st_dev=actual.st_dev,
        st_ino=actual.st_ino,
        st_size=actual.st_size,
        st_mtime_ns=actual.st_mtime_ns,
        st_ctime_ns=actual.st_ctime_ns + 1,
    )
    states = iter((actual, actual, changed))
    original_stat = Path.stat

    def changing_stat(path: Path, *args, **kwargs):
        if path == data_target:
            return next(states)
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "stat", changing_stat)
        with pytest.raises(GraphCacheError, match="changed while hashing"):
            cache_module._file_descriptor(data_target)

    with pytest.raises(GraphCacheError, match="reserved"):
        cache_module._build_authenticated_manifest(
            kind="graph",
            fingerprint="fp",
            payload={"_cache": {}},
            entry=tmp_path,
            files=[],
            arrays={},
            content_sha256="content",
        )
    with pytest.raises(GraphCacheError, match="Manifest fingerprint differs"):
        cache_module._build_authenticated_manifest(
            kind="graph",
            fingerprint="fp",
            payload={"fingerprint": "other"},
            entry=tmp_path,
            files=[],
            arrays={},
            content_sha256="content",
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("_cache", "schema_version"), -1, "Unsupported"),
        (("_cache", "kind"), "views", "kind mismatch"),
        (("fingerprint",), "other", "Manifest fingerprint differs"),
        (("_cache", "fingerprint"), "other", "Authenticated fingerprint differs"),
        (("_cache", "files"), {}, "no authenticated files"),
    ],
)
def test_authenticated_manifest_header_validation(tmp_path, path, value, message) -> None:
    _cache, entry = _saved_graph(tmp_path, message.replace(" ", "-"))
    _rewrite_manifest(entry, lambda manifest: _set_nested(manifest, path, value))
    with pytest.raises(GraphCacheError, match=message):
        cache_module._validate_authenticated_manifest(
            entry=entry,
            fingerprint="fp",
            kind="graph",
            verify_file_hashes=True,
        )


def test_authenticated_manifest_file_descriptor_validation(tmp_path, monkeypatch) -> None:
    _cache, entry = _saved_graph(tmp_path, "descriptors")
    data_path = entry / "edge_index.npy"

    _rewrite_manifest(
        entry,
        lambda manifest: manifest["_cache"]["files"].__setitem__("edge_index.npy", "bad"),
    )
    with pytest.raises(GraphCacheError, match="Invalid file descriptor"):
        cache_module._validate_authenticated_manifest(
            entry=entry, fingerprint="fp", kind="graph", verify_file_hashes=True
        )

    _cache, entry = _saved_graph(tmp_path, "unsafe-name")
    descriptor = _manifest(entry)["_cache"]["files"]["edge_index.npy"]
    _rewrite_manifest(
        entry,
        lambda manifest: manifest["_cache"].__setitem__("files", {"../unsafe": descriptor}),
    )
    original_iterdir = Path.iterdir

    def unsafe_iterdir(path: Path):
        if path == entry:
            return iter((SimpleNamespace(name="../unsafe"),))
        return original_iterdir(path)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "iterdir", unsafe_iterdir)
        with pytest.raises(GraphCacheError, match="Unsafe cached file name"):
            cache_module._validate_authenticated_manifest(
                entry=entry, fingerprint="fp", kind="graph", verify_file_hashes=True
            )

    _cache, entry = _saved_graph(tmp_path, "file-symlink")
    data_path = entry / "edge_index.npy"
    target = tmp_path / "external-edge-index.npy"
    data_path.replace(target)
    data_path.symlink_to(target)
    with pytest.raises(GraphCacheError, match="Missing or unsafe"):
        cache_module._validate_authenticated_manifest(
            entry=entry, fingerprint="fp", kind="graph", verify_file_hashes=True
        )

    _cache, entry = _saved_graph(tmp_path, "stat-error")
    data_path = entry / "edge_index.npy"
    original_is_file = Path.is_file
    original_is_symlink = Path.is_symlink
    original_stat = Path.stat

    def is_file(path: Path) -> bool:
        return True if path == data_path else original_is_file(path)

    def stat_error(path: Path, *args, **kwargs):
        if path == data_path:
            raise OSError("stat failed")
        return original_stat(path, *args, **kwargs)

    def is_symlink(path: Path) -> bool:
        return False if path == data_path else original_is_symlink(path)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "is_file", is_file)
        scoped.setattr(Path, "is_symlink", is_symlink)
        scoped.setattr(Path, "stat", stat_error)
        with pytest.raises(GraphCacheError, match="Unable to stat"):
            cache_module._validate_authenticated_manifest(
                entry=entry, fingerprint="fp", kind="graph", verify_file_hashes=True
            )


@pytest.mark.parametrize("existing", [False, True])
def test_publish_failure_restores_or_leaves_destination_absent(
    tmp_path, monkeypatch, existing
) -> None:
    root = tmp_path / ("existing" if existing else "new")
    cache = GraphCache(root=root)
    destination = cache.entry_dir("fp")
    if existing:
        destination.mkdir(parents=True)
        (destination / "legacy.txt").write_text("legacy")

    original_replace = cache_module.os.replace

    def fail_staging_publish(source, target) -> None:
        source_path = Path(source)
        target_path = Path(target)
        if ".staging-" in source_path.name and target_path == destination:
            raise OSError("publish failed")
        original_replace(source, target)

    monkeypatch.setattr(cache_module.os, "replace", fail_staging_publish)
    with pytest.raises(OSError, match="publish failed"):
        cache.save(fingerprint="fp", graph=_graph(), manifest={})
    assert destination.exists() is existing
    if existing:
        assert (destination / "legacy.txt").read_text() == "legacy"


def test_same_content_with_different_identity_is_a_collision(tmp_path) -> None:
    cache = GraphCache(root=tmp_path)
    cache.save(
        fingerprint="fp",
        graph=_graph(),
        manifest={"dataset_fingerprint": "dataset-a"},
    )
    with pytest.raises(GraphCacheError, match="identity field"):
        cache.save(
            fingerprint="fp",
            graph=_graph(),
            manifest={"dataset_fingerprint": "dataset-b"},
        )


def test_graph_save_rejects_metadata_fingerprint(tmp_path) -> None:
    cache = GraphCache(root=tmp_path)
    graph = GraphArtifact(
        n_nodes=3,
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        meta={"fingerprint": "other"},
    )
    with pytest.raises(GraphCacheError, match="metadata fingerprint"):
        cache.save(fingerprint="fp", graph=graph, manifest={})


def test_graph_edge_deserialization_errors(tmp_path, monkeypatch) -> None:
    cache = GraphCache(root=tmp_path)

    single = tmp_path / "single"
    single.mkdir()
    (single / "edge_index.npy").write_bytes(b"invalid")
    with pytest.raises(GraphCacheError, match="Corrupted cached edge_index"):
        cache._load_edges_single(single)

    invalid_index = tmp_path / "invalid-index"
    invalid_index.mkdir()
    np.savez_compressed(invalid_index / "edges_0000.npz", edge_index=np.zeros((3, 1)))
    with pytest.raises(GraphCacheError, match="Invalid edge_index"):
        cache._load_edges_sharded(invalid_index, num_shards=1)

    invalid_weight = tmp_path / "invalid-weight"
    invalid_weight.mkdir()
    np.savez_compressed(
        invalid_weight / "edges_0000.npz",
        edge_index=np.zeros((2, 1)),
        edge_weight=np.zeros(2),
    )
    with pytest.raises(GraphCacheError, match="Invalid edge_weight"):
        cache._load_edges_sharded(invalid_weight, num_shards=1)

    corrupted = tmp_path / "corrupted"
    corrupted.mkdir()
    (corrupted / "edges_0000.npz").write_bytes(b"invalid")
    with pytest.raises(GraphCacheError, match="Corrupted edge shard"):
        cache._load_edges_sharded(corrupted, num_shards=1)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("_cache", "arrays"), None, "no authenticated array descriptors"),
        (("meta", "fingerprint"), "other", "metadata fingerprint"),
        (("_storage",), None, "storage descriptor"),
        (("_storage", "edge"), {"kind": "sharded", "num_shards": "bad"}, "shard count"),
        (("_storage", "edge", "kind"), "unknown", "storage kind"),
        (("_cache", "arrays", "edge_index", "sha256"), "bad", "edge_index logical"),
        (("n_nodes",), "bad", "n_nodes"),
        (("n_edges",), 999, "n_edges"),
        (("has_edge_weight",), True, "weight presence"),
        (("_cache", "content_sha256"), "bad", "content commitment"),
        (("graph_content_sha256",), "bad", "content fingerprint"),
    ],
)
def test_graph_load_rejects_authenticated_semantic_inconsistency(
    tmp_path, path, value, message
) -> None:
    cache, entry = _saved_graph(tmp_path, message.replace(" ", "-"))
    _rewrite_manifest(entry, lambda manifest: _set_nested(manifest, path, value))
    with pytest.raises(GraphCacheError, match=message):
        cache.load("fp")


def test_graph_load_rejects_weight_descriptor_and_nonfinite_weight(tmp_path) -> None:
    cache, entry = _saved_graph(tmp_path, "weight-descriptor", weighted=True)
    _rewrite_manifest(
        entry,
        lambda manifest: _set_nested(
            manifest, ("_cache", "arrays", "edge_weight", "sha256"), "bad"
        ),
    )
    with pytest.raises(GraphCacheError, match="edge_weight logical"):
        cache.load("fp")

    cache, entry = _saved_graph(tmp_path, "nonfinite", weighted=True)
    weight_path = entry / "edge_weight.npy"
    weight = np.load(weight_path, allow_pickle=False)
    weight[0] = np.nan
    np.save(weight_path, weight, allow_pickle=False)

    def authenticate_nonfinite(manifest: dict[str, Any]) -> None:
        manifest["_cache"]["files"]["edge_weight.npy"] = cache_module._file_descriptor(weight_path)
        manifest["_cache"]["arrays"]["edge_weight"] = cache_module._array_descriptor(weight)

    _rewrite_manifest(entry, authenticate_nonfinite)
    with pytest.raises(GraphCacheError, match="non-finite"):
        cache.load("fp")


def test_graph_load_detects_state_change_after_deserialization(tmp_path, monkeypatch) -> None:
    cache, _entry = _saved_graph(tmp_path, "state-change")
    original = cache_module._validate_authenticated_manifest
    calls = 0

    def changed_on_second_validation(**kwargs):
        nonlocal calls
        calls += 1
        manifest = original(**kwargs)
        if calls == 2:
            manifest = {**manifest, "changed": True}
        return manifest

    monkeypatch.setattr(
        cache_module, "_validate_authenticated_manifest", changed_on_second_validation
    )
    with pytest.raises(GraphCacheError, match="state changed"):
        cache.load("fp")


def test_views_save_rejects_object_and_metadata_fingerprint(tmp_path) -> None:
    cache = ViewsCache(root=tmp_path)
    object_views = DatasetViews(
        views={"object": np.array([[object()], [object()]], dtype=object)},
        y=np.zeros(2, dtype=np.int64),
    )
    with pytest.raises(GraphCacheError, match="Object arrays"):
        cache.save(fingerprint="fp", views=object_views, manifest={})
    with pytest.raises(GraphCacheError, match="metadata fingerprint"):
        cache.save(
            fingerprint="fp",
            views=_views(meta={"fingerprint": "other"}),
            manifest={},
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("_cache", "arrays"), None, "no authenticated array descriptors"),
        (("view_names",), None, "view names"),
        (("view_names",), ["other"], "view names"),
        (("_cache", "arrays", "attr", "sha256"), "bad", "logical content"),
        (("_cache", "content_sha256"), "bad", "content commitment"),
        (("views_content_sha256",), "bad", "content fingerprint"),
        (("n_nodes",), 999, "n_nodes"),
        (("meta", "fingerprint"), "other", "metadata fingerprint"),
    ],
)
def test_views_load_rejects_authenticated_semantic_inconsistency(
    tmp_path, path, value, message
) -> None:
    cache, entry, views = _saved_views(tmp_path, message.replace(" ", "-"))
    _rewrite_manifest(entry, lambda manifest: _set_nested(manifest, path, value))
    with pytest.raises(GraphCacheError, match=message):
        cache.load("fp", y=views.y, masks={})


def test_views_load_rejects_corrupted_archive_after_authentication(tmp_path) -> None:
    cache, entry, views = _saved_views(tmp_path, "corrupted-archive")
    archive = entry / "views.npz"
    archive.write_bytes(b"not-an-npz")

    def authenticate_corruption(manifest: dict[str, Any]) -> None:
        manifest["_cache"]["files"]["views.npz"] = cache_module._file_descriptor(archive)

    _rewrite_manifest(entry, authenticate_corruption)
    with pytest.raises(GraphCacheError, match="Corrupted cached views"):
        cache.load("fp", y=views.y, masks={})


def test_views_load_detects_state_change_after_deserialization(tmp_path, monkeypatch) -> None:
    cache, _entry, views = _saved_views(tmp_path, "views-state-change")
    original = cache_module._validate_authenticated_manifest
    calls = 0

    def changed_on_second_validation(**kwargs):
        nonlocal calls
        calls += 1
        manifest = original(**kwargs)
        if calls == 2:
            manifest = {**manifest, "changed": True}
        return manifest

    monkeypatch.setattr(
        cache_module, "_validate_authenticated_manifest", changed_on_second_validation
    )
    with pytest.raises(GraphCacheError, match="state changed"):
        cache.load("fp", y=views.y, masks={})


def test_cache_list_keeps_directory_name_for_nonstr_fingerprint(tmp_path) -> None:
    entry = tmp_path / "entry"
    entry.mkdir()
    (entry / "manifest.json").write_text('{"fingerprint": 123}')
    assert GraphCache(root=tmp_path).list() == ["entry"]
