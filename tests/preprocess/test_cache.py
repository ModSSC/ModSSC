import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.preprocess.cache import (
    CacheManager,
    _load_value,
    _resolve_cache_device,
    _save_json,
    _save_value,
    default_cache_dir,
    json_load,
)
from modssc.preprocess.errors import OptionalDependencyError, PreprocessCacheError


def test_default_cache_dir_override():
    with patch.dict(os.environ, {"MODSSC_PREPROCESS_CACHE_DIR": "/tmp/custom_cache"}):
        path = default_cache_dir()
        assert path == Path("/tmp/custom_cache").resolve()


def test_default_cache_dir_heuristic():
    with patch("pathlib.Path.cwd") as mock_cwd:
        mock_path = MagicMock(spec=Path)
        mock_cwd.return_value = mock_path

        parent = MagicMock(spec=Path)
        mock_path.parents = [parent]

        def exists_side_effect(arg=None):
            return True

        pyproj_mock = MagicMock()
        pyproj_mock.exists.return_value = True

        parent.__truediv__.return_value = pyproj_mock

        cache_mock = MagicMock()
        preprocess_mock = MagicMock()
        parent.__truediv__.side_effect = (
            lambda x: pyproj_mock if x == "pyproject.toml" else cache_mock
        )
        cache_mock.__truediv__.return_value = preprocess_mock

        pass


def test_default_cache_dir_heuristic_simple(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    subdir = tmp_path / "src"
    subdir.mkdir()

    with patch("pathlib.Path.cwd", return_value=subdir):
        path = default_cache_dir()
        assert path == tmp_path / "cache" / "preprocess"


def test_default_cache_dir_fallback(tmp_path):
    with (
        patch("pathlib.Path.cwd", return_value=Path("/")),
        patch("pathlib.Path.exists", return_value=False),
        patch("modssc.preprocess.cache.user_cache_dir", return_value=str(tmp_path)),
    ):
        path = default_cache_dir()
        assert path == tmp_path / "preprocess"


def test_save_value_json_types(tmp_path):
    desc = _save_value(tmp_path / "str", "hello")
    assert desc["type"] == "json"
    assert json.loads((tmp_path / "str.json").read_text()) == "hello"

    desc = _save_value(tmp_path / "int", 123)
    assert desc["type"] == "json"

    desc = _save_value(tmp_path / "none", None)
    assert desc["type"] == "json"

    desc = _save_value(tmp_path / "tuple", (1, 2))
    assert desc["type"] == "json"
    assert json.loads((tmp_path / "tuple.json").read_text()) == [1, 2]


def test_save_json_with_extra(tmp_path):
    desc = _save_json(tmp_path / "payload", {"a": 1}, extra={"format": "ndarray"})
    assert desc["type"] == "json"
    assert desc["format"] == "ndarray"


def test_save_value_unsupported_json(tmp_path):
    class Obj:
        pass

    desc = _save_value(tmp_path / "bad", {"a": Obj()})
    assert desc is None


def test_save_value_object_array_large(tmp_path, monkeypatch):
    import modssc.preprocess.cache as cache_mod

    monkeypatch.setattr(cache_mod, "OBJECT_JSON_MAX_ITEMS", 1)
    arr = np.array(["a", "b"], dtype=object)
    desc = _save_value(tmp_path / "obj", arr)
    assert desc["type"] == "npy"
    assert desc["allow_pickle"] is True


def test_save_value_object_array_small(tmp_path):
    arr = np.array(["a"], dtype=object)
    desc = _save_value(tmp_path / "obj", arr)
    assert desc["type"] == "json"
    assert desc["format"] == "ndarray"


def test_save_value_object_array_tolist_error(tmp_path):
    class BadArray(np.ndarray):
        def tolist(self):
            raise TypeError("fail")

    arr = np.array(["a"], dtype=object).view(BadArray)
    desc = _save_value(tmp_path / "obj", arr)
    assert desc is None


def test_save_value_unsupported_type(tmp_path):
    class Obj:
        pass

    desc = _save_value(tmp_path / "obj", Obj())
    assert desc is None


def test_save_value_scipy_missing(tmp_path):
    with (
        patch("modssc.preprocess.cache._is_scipy_sparse", return_value=True),
        patch.dict(sys.modules, {"scipy": None, "scipy.sparse": None}),
    ):

        class Dummy:
            pass

        with pytest.raises(OptionalDependencyError, match="scipy sparse IO"):
            _save_value(tmp_path / "sparse", Dummy())


def test_load_value_errors(tmp_path):
    with pytest.raises(PreprocessCacheError, match="missing 'path'"):
        _load_value(tmp_path, {"type": "json"})

    with pytest.raises(PreprocessCacheError, match="Unsupported cached value type"):
        _load_value(tmp_path, {"type": "unknown", "path": "x"})

    with (
        patch("modssc.preprocess.cache.np.load", side_effect=ValueError("bad")),
        pytest.raises(PreprocessCacheError, match="Failed to load cached array"),
    ):
        _load_value(tmp_path, {"type": "npy", "path": "missing.npy"})


def test_load_value_scipy_missing(tmp_path):
    with (
        patch.dict(sys.modules, {"scipy": None}),
        pytest.raises(OptionalDependencyError, match="scipy sparse IO"),
    ):
        _load_value(tmp_path, {"type": "npz", "path": "x.npz"})


def test_save_step_outputs_skips_unsupported(tmp_path):
    cm = CacheManager(root=tmp_path, dataset_fingerprint="ds")

    class Obj:
        pass

    produced = {"valid": 1, "invalid": Obj()}

    cm.save_step_outputs(step_fingerprint="step1", split="train", produced=produced, manifest={})

    manifest_path = cm.step_dir("step1") / "manifest.json"
    data = json.loads(manifest_path.read_text())
    saved = data["saved"]["train"]

    assert "valid" in saved
    assert "invalid" not in saved


def test_save_step_outputs_manifest_update(tmp_path):
    cm = CacheManager(root=tmp_path, dataset_fingerprint="ds")

    cm.save_step_outputs(step_fingerprint="step1", split="train", produced={}, manifest={"a": 1})

    cm.save_step_outputs(
        step_fingerprint="step1",
        split="test",
        produced={},
        manifest={"b": 2, "saved": "should_be_ignored"},
    )

    manifest_path = cm.step_dir("step1") / "manifest.json"
    data = json.loads(manifest_path.read_text())

    assert data["a"] == 1
    assert data["b"] == 2
    assert isinstance(data["saved"], dict)
    assert "train" in data["saved"]
    assert "test" in data["saved"]


def test_load_step_outputs_missing_manifest(tmp_path):
    cm = CacheManager(root=tmp_path, dataset_fingerprint="ds")
    with pytest.raises(PreprocessCacheError, match="Missing cache manifest"):
        cm.load_step_outputs(step_fingerprint="missing", split="train")


def test_load_step_outputs_invalid_structure(tmp_path):
    cm = CacheManager(root=tmp_path, dataset_fingerprint="ds")
    step_dir = cm.step_dir("step1")
    step_dir.mkdir(parents=True)

    (step_dir / "manifest.json").write_text(json.dumps({"saved": {"train": "not_a_dict"}}))

    with pytest.raises(PreprocessCacheError, match="Invalid cache manifest structure"):
        cm.load_step_outputs(step_fingerprint="step1", split="train")

    (step_dir / "manifest.json").write_text(json.dumps({"saved": {"train": {"key": "not_a_dict"}}}))

    out = cm.load_step_outputs(step_fingerprint="step1", split="train")
    assert "key" not in out


def test_json_load_not_dict():
    with pytest.raises(PreprocessCacheError, match="Invalid JSON manifest"):
        json_load("[1, 2]")


def test_load_value_json(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")

    val = _load_value(tmp_path, {"type": "json", "path": "data.json"})
    assert val == {"foo": "bar"}


def test_load_value_json_ndarray(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps([1, 2]), encoding="utf-8")

    val = _load_value(tmp_path, {"type": "json", "path": "data.json", "format": "ndarray"})
    assert isinstance(val, np.ndarray)
    assert val.tolist() == [1, 2]


def test_load_value_npy(tmp_path):
    p = tmp_path / "data.npy"
    arr = np.array([1, 2, 3])
    np.save(p, arr)

    val = _load_value(tmp_path, {"type": "npy", "path": "data.npy"})
    assert np.array_equal(val, arr)


def test_load_value_npy_memmap(tmp_path, monkeypatch):
    import modssc.preprocess.cache as cache_mod

    monkeypatch.setattr(cache_mod, "MMAP_THRESHOLD_BYTES", 0)
    p = tmp_path / "data.npy"
    arr = np.array([1, 2, 3], dtype=np.int64)
    np.save(p, arr)

    def fake_load(fp, allow_pickle=False, mmap_mode=None):
        assert mmap_mode == "r"
        return arr

    monkeypatch.setattr(cache_mod.np, "load", fake_load)
    val = _load_value(tmp_path, {"type": "npy", "path": "data.npy"})
    assert np.array_equal(np.asarray(val), arr)


def test_load_value_npy_allow_pickle(tmp_path):
    p = tmp_path / "data.npy"
    arr = np.array(["a", {"b": 1}], dtype=object)
    np.save(p, arr, allow_pickle=True)

    val = _load_value(tmp_path, {"type": "npy", "path": "data.npy", "allow_pickle": True})
    assert val.dtype == object
    assert val.tolist() == arr.tolist()


def test_cache_manager_roundtrip(tmp_path):
    cm = CacheManager(root=tmp_path, dataset_fingerprint="ds1")

    produced = {
        "matrix": np.array([1, 2, 3]),
        "meta": {"a": 1},
        "score": 0.5,
        "complex/key": "value",
    }
    manifest = {"params": {"p": 1}}

    cm.save_step_outputs(
        step_fingerprint="step1", split="train", produced=produced, manifest=manifest
    )

    assert cm.has_step_outputs("step1", split="train")
    assert not cm.has_step_outputs("step1", split="test")

    loaded = cm.load_step_outputs(step_fingerprint="step1", split="train")

    assert np.array_equal(loaded["matrix"], produced["matrix"])
    assert loaded["meta"] == produced["meta"]
    assert loaded["score"] == produced["score"]
    assert loaded["complex/key"] == produced["complex/key"]

    manifest_path = cm.step_dir("step1") / "manifest.json"
    saved_manifest = json.loads(manifest_path.read_text())
    assert saved_manifest["params"] == manifest["params"]
    assert "saved" in saved_manifest
    assert "train" in saved_manifest["saved"]
    assert "complex/key" in saved_manifest["saved"]["train"]


def test_save_load_sparse_success(tmp_path):
    mock_sparse = MagicMock()
    mock_sparse.save_npz = MagicMock()
    mock_sparse.load_npz = MagicMock(return_value="loaded_sparse")

    mock_scipy = MagicMock()
    mock_scipy.sparse = mock_sparse

    class SparseObj:
        pass

    sparse_obj = SparseObj()

    with (
        patch.dict(sys.modules, {"scipy": mock_scipy, "scipy.sparse": mock_sparse}),
        patch("modssc.preprocess.cache._is_scipy_sparse", return_value=True),
    ):
        desc = _save_value(tmp_path / "mat", sparse_obj)
        assert desc["type"] == "npz"
        assert desc["path"] == "mat.npz"
        mock_sparse.save_npz.assert_called_once()

        val = _load_value(tmp_path, desc)
        assert val == "loaded_sparse"
        mock_sparse.load_npz.assert_called_once()


def test_cache_manager_factory():
    with patch("modssc.preprocess.cache.default_cache_dir", return_value=Path("/tmp/cache")):
        cm = CacheManager.for_dataset("ds1")
        assert cm.root == Path("/tmp/cache")
        assert cm.dataset_fingerprint == "ds1"


def test_require_torch_missing(monkeypatch):
    import importlib

    import modssc.preprocess.cache as cache_mod

    def _boom(name):
        raise ModuleNotFoundError("no torch")

    monkeypatch.setattr(importlib, "import_module", _boom)

    with pytest.raises(OptionalDependencyError, match="inductive-torch"):
        cache_mod._require_torch()


def test_require_torch_success():
    import modssc.preprocess.cache as cache_mod

    torch_mod = cache_mod._require_torch()
    assert torch_mod.__name__ == "torch"


def test_save_load_torch_tensor_npy(tmp_path, monkeypatch):
    import modssc.preprocess.cache as cache_mod

    class DummyTensor:
        __module__ = "torch.tensor"

        def __init__(self, arr):
            self._arr = np.asarray(arr)
            self.shape = self._arr.shape
            self.dtype = "float32"
            self.device = "cpu"

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    class DummyTorch:
        float32 = "float32"

        class _Cuda:
            def is_available(self):
                return True

        cuda = _Cuda()

        def device(self, name):
            return f"dev:{name}"

        def as_tensor(self, arr, device=None, dtype=None):
            return {"arr": np.asarray(arr), "device": device, "dtype": dtype}

    tensor = DummyTensor([1, 2, 3])
    desc = cache_mod._save_value(tmp_path / "tensor", tensor)
    assert desc["type"] == "torch_npy"

    monkeypatch.setattr(cache_mod, "_require_torch", lambda: DummyTorch())

    loaded = cache_mod._load_value(
        tmp_path,
        {"type": "torch_npy", "path": "tensor.npy", "dtype": "torch.float32", "device": "cuda"},
    )

    assert loaded["device"] == "dev:cuda"
    assert loaded["dtype"] == "float32"
    assert np.array_equal(loaded["arr"], np.asarray([1, 2, 3]))


def test_save_torch_tensor_without_detach_cpu(tmp_path):
    import modssc.preprocess.cache as cache_mod

    class DummyTensor:
        __module__ = "torch.tensor"

        def __init__(self, arr):
            self._arr = np.asarray(arr)
            self.shape = self._arr.shape
            self.dtype = "float32"
            self.device = "cpu"

        def numpy(self):
            return self._arr

    desc = cache_mod._save_value(tmp_path / "tensor", DummyTensor([1, 2]))
    assert desc["type"] == "torch_npy"


def test_save_load_torch_tensor_pt(tmp_path, monkeypatch):
    import modssc.preprocess.cache as cache_mod

    class DummyTensor:
        __module__ = "torch.tensor"

        def __init__(self):
            self.shape = (2,)
            self.dtype = "float32"
            self.device = "cpu"

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            raise RuntimeError("no numpy")

    class DummyObj:
        def __init__(self):
            self.moved = None

        def to(self, device):
            self.moved = device
            return f"moved:{device}"

    class DummyTorch:
        def __init__(self):
            self.saved = []

            class _Cuda:
                def is_available(self):
                    return True

            self.cuda = _Cuda()

        def save(self, obj, path):
            Path(path).write_text("pt")
            self.saved.append(path)

        def load(self, path, map_location="cpu"):
            return DummyObj()

        def device(self, name):
            return f"dev:{name}"

    dummy_torch = DummyTorch()
    monkeypatch.setattr(cache_mod, "_require_torch", lambda: dummy_torch)

    desc = cache_mod._save_value(tmp_path / "tensor", DummyTensor())
    assert desc["type"] == "torch_pt"

    moved = cache_mod._load_value(
        tmp_path, {"type": "torch_pt", "path": "tensor.pt", "device": "cuda"}
    )
    assert moved == "moved:dev:cuda"

    obj = cache_mod._load_value(tmp_path, {"type": "torch_pt", "path": "tensor.pt", "device": ""})
    assert isinstance(obj, DummyObj)


def test_safe_path_component_helpers(monkeypatch):
    import modssc.preprocess.cache as cache_mod

    assert cache_mod._safe_path_component("alpha") == "alpha"
    monkeypatch.setattr(cache_mod.os, "name", "nt", raising=False)
    assert cache_mod._safe_path_component("a<b>c") == "a_b_c"
    assert cache_mod._safe_path_component("..") == "_"


def test_safe_name_helper():
    import modssc.preprocess.cache as cache_mod

    assert cache_mod._safe_name("foo/bar..baz") == "foo_bar__baz"


def test_tensor_and_sparse_detection_helpers():
    import modssc.preprocess.cache as cache_mod

    class DummyTorch:
        __module__ = "torch.fake"

        def __init__(self):
            self.shape = (1,)
            self.dtype = "float32"
            self.device = "cpu"

    class DummySparse:
        __module__ = "scipy.sparse.csr"

    assert cache_mod._is_torch_tensor(DummyTorch()) is True
    assert cache_mod._is_torch_tensor(np.zeros((1,))) is False
    assert cache_mod._is_scipy_sparse(DummySparse()) is True
    assert cache_mod._is_scipy_sparse(object()) is False


class _DummyCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _DummyTorch:
    def __init__(self, cuda_available: bool) -> None:
        self.cuda = _DummyCuda(cuda_available)

    def device(self, name: str) -> str:
        return f"device:{name}"


def test_resolve_cache_device_empty_string():
    assert _resolve_cache_device(_DummyTorch(cuda_available=True), "") is None


def test_resolve_cache_device_cuda_unavailable():
    dummy = _DummyTorch(cuda_available=False)
    assert _resolve_cache_device(dummy, "cuda:0") == "device:cpu"


def test_resolve_cache_device_mps_unavailable(monkeypatch):
    cache_mod = sys.modules["modssc.preprocess.cache"]

    monkeypatch.setattr(cache_mod, "mps_is_available", lambda _: False)
    dummy = _DummyTorch(cuda_available=True)
    assert _resolve_cache_device(dummy, "mps:0") == "device:cpu"


def test_load_value_npy_raises_preprocess_cache_error_on_value_error(tmp_path):
    with (
        patch("modssc.preprocess.cache.np.load", side_effect=ValueError("Corrupted")),
        pytest.raises(PreprocessCacheError, match="Failed to load cached array"),
    ):
        _load_value(tmp_path, {"type": "npy", "path": "corrupted.npy"})


def test_load_value_torch_npy_mmap(tmp_path, monkeypatch):
    cache_mod = sys.modules["modssc.preprocess.cache"]

    # Enable mmap by setting threshold to 0
    monkeypatch.setattr(cache_mod, "MMAP_THRESHOLD_BYTES", 0)

    fp = tmp_path / "data.npy"
    np.save(fp, np.array([1.0, 2.0], dtype="float32"))

    desc = {"type": "torch_npy", "path": "data.npy", "dtype": "torch.float32"}

    # Mock torch
    mock_torch = MagicMock()
    # We need to simulate as_tensor returning something
    mock_torch.as_tensor.return_value = "tensor"
    # And getattr(torch, "float32")
    mock_torch.float32 = "float32_dtype"

    with (
        patch("modssc.preprocess.cache.np.load", wraps=np.load) as mock_load,
        patch("modssc.preprocess.cache._require_torch", return_value=mock_torch),
        patch("modssc.preprocess.cache._resolve_cache_device", return_value="cpu"),
    ):
        val = _load_value(tmp_path, desc)

    assert val == "tensor"
    mock_load.assert_called_with(fp, allow_pickle=False, mmap_mode="r")
