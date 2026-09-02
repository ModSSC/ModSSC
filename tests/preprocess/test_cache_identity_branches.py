from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
from scipy import sparse

import modssc.preprocess.services.pipeline as pipeline_module


class _OpaqueValue:
    pass


def _digest(value: object) -> str:
    digest = hashlib.sha256()
    pipeline_module._update_content_digest(digest, value)
    return digest.hexdigest()


def test_content_digest_frames_sparse_and_object_variants() -> None:
    matrix = sparse.coo_matrix(
        (
            np.array([1.5, 2.5], dtype=np.float64),
            (np.array([0, 1]), np.array([1, 0])),
        ),
        shape=(2, 2),
    )
    assert _digest(matrix) != _digest(matrix.toarray())

    scalar_string = np.array("value", dtype=object)
    opaque = _OpaqueValue()
    assert _digest(scalar_string) != _digest(opaque)

    variants = np.empty(5, dtype=object)
    variants[:] = [np.int64(3), b"bytes", None, 2.5, _OpaqueValue()]
    first = _digest(variants)
    variants[1] = b"changed"
    assert _digest(variants) != first


def test_step_import_identity_handles_unresolvable_and_non_file_modules(
    tmp_path, monkeypatch
) -> None:
    pipeline_module._step_import_identity.cache_clear()
    monkeypatch.setattr(pipeline_module.importlib_metadata, "packages_distributions", lambda: {})

    def fail_find_spec(_name: str) -> object:
        raise ValueError("invalid module")

    monkeypatch.setattr(pipeline_module.importlib.util, "find_spec", fail_find_spec)
    unresolved = pipeline_module._step_import_identity("invalid.module:Step")
    assert "module_file_sha256" not in unresolved

    pipeline_module._step_import_identity.cache_clear()
    monkeypatch.setattr(
        pipeline_module.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=None),
    )
    namespace = pipeline_module._step_import_identity("namespace.module:Step")
    assert "module_file_sha256" not in namespace

    pipeline_module._step_import_identity.cache_clear()
    missing_source = tmp_path / "missing.py"
    monkeypatch.setattr(
        pipeline_module.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(missing_source)),
    )
    missing = pipeline_module._step_import_identity("missing.module:Step")
    assert "module_file_sha256" not in missing
    pipeline_module._step_import_identity.cache_clear()


def test_preprocess_cache_identity_accepts_unmapped_optional_extra(monkeypatch) -> None:
    class _Manifest:
        def to_dict(self) -> dict[str, object]:
            return {"versions": {}}

    captured: set[str] = set()

    def collect(distributions: set[str], *, require_complete: bool) -> _Manifest:
        assert require_complete is False
        captured.update(distributions)
        return _Manifest()

    step = SimpleNamespace(
        step_id="custom",
        spec=SimpleNamespace(
            required_extra="unmapped-extra",
            import_path="custom.module:Step",
        ),
    )
    monkeypatch.setattr(pipeline_module, "collect_software_manifest", collect)
    monkeypatch.setattr(pipeline_module, "_preprocess_implementation_sha256", lambda: "a" * 64)
    monkeypatch.setattr(
        pipeline_module,
        "_step_import_identity",
        lambda _import_path: {"module": "custom.module"},
    )

    identity = pipeline_module._preprocess_cache_identity((step,))
    assert identity["steps"][0]["id"] == "custom"
    assert captured == {"numpy", "scipy"}
