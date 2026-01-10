from __future__ import annotations

from modssc import device as device_mod


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = bool(available)

    def is_available(self) -> bool:
        return self._available


class _FakeMps:
    def __init__(self, available: bool) -> None:
        self._available = bool(available)

    def is_available(self) -> bool:
        return self._available


class _FakeBackends:
    def __init__(self, mps_available: bool) -> None:
        self.mps = _FakeMps(mps_available)


class _FakeTorch:
    def __init__(self, *, cuda_available: bool, mps_available: bool) -> None:
        self.cuda = _FakeCuda(cuda_available)
        self.backends = _FakeBackends(mps_available)

    def zeros(self, *args, **kwargs):
        return object()


def test_resolve_device_name_passthrough() -> None:
    assert device_mod.resolve_device_name("cpu") == "cpu"
    assert device_mod.resolve_device_name("cuda") == "cuda"
    assert device_mod.resolve_device_name("mps") == "mps"


def test_resolve_device_name_none() -> None:
    assert device_mod.resolve_device_name(None) is None


def test_resolve_device_name_auto_prefers_cuda() -> None:
    torch = _FakeTorch(cuda_available=True, mps_available=True)
    assert device_mod.resolve_device_name("auto", torch=torch) == "cuda"


def test_resolve_device_name_auto_prefers_mps() -> None:
    torch = _FakeTorch(cuda_available=False, mps_available=True)
    assert device_mod.resolve_device_name("auto", torch=torch) == "mps"


def test_resolve_device_name_auto_falls_back_cpu() -> None:
    torch = _FakeTorch(cuda_available=False, mps_available=False)
    assert device_mod.resolve_device_name("auto", torch=torch) == "cpu"


def test_resolve_device_name_auto_with_none() -> None:
    assert device_mod.resolve_device_name("auto", torch=None) == "cpu"


def test_resolve_device_name_auto_cached(monkeypatch) -> None:
    device_mod._cached_best_device_name.cache_clear()
    monkeypatch.setattr(
        device_mod,
        "_try_import_torch",
        lambda: _FakeTorch(cuda_available=False, mps_available=True),
    )
    assert device_mod.resolve_device_name("auto") == "mps"

    monkeypatch.setattr(
        device_mod,
        "_try_import_torch",
        lambda: _FakeTorch(cuda_available=True, mps_available=False),
    )
    assert device_mod.resolve_device_name("auto") == "mps"

    device_mod._cached_best_device_name.cache_clear()
    assert device_mod.resolve_device_name("auto") == "cuda"


def test_try_import_torch_success(monkeypatch) -> None:
    import importlib

    sentinel = object()
    monkeypatch.setattr(importlib, "import_module", lambda name: sentinel)
    assert device_mod._try_import_torch() is sentinel


def test_try_import_torch_failure(monkeypatch) -> None:
    import importlib

    def _boom(name: str):
        raise ImportError("nope")

    monkeypatch.setattr(importlib, "import_module", _boom)
    assert device_mod._try_import_torch() is None


def test_best_device_missing_is_available() -> None:
    class _NoIsAvailable:
        pass

    class _BackendsNoCheck:
        def __init__(self) -> None:
            self.mps = _NoIsAvailable()

    class _TorchNoCheck:
        def __init__(self) -> None:
            self.cuda = _NoIsAvailable()
            self.backends = _BackendsNoCheck()

    torch = _TorchNoCheck()
    assert device_mod._best_device_from_torch(torch) == "cpu"


def test_mps_is_available_with_none() -> None:
    device_mod.mps_is_available.cache_clear()
    assert device_mod.mps_is_available(None) is False


def test_mps_is_available_missing_mps() -> None:
    class _TorchNoBackends:
        pass

    device_mod.mps_is_available.cache_clear()
    assert device_mod.mps_is_available(_TorchNoBackends()) is False


def test_mps_is_available_is_built_false() -> None:
    class _MpsBuiltFalse:
        def is_built(self):
            return False

        def is_available(self):
            return True

    class _Backends:
        def __init__(self) -> None:
            self.mps = _MpsBuiltFalse()

    class _Torch:
        def __init__(self) -> None:
            self.backends = _Backends()

        def zeros(self, *args, **kwargs):
            return object()

    device_mod.mps_is_available.cache_clear()
    assert device_mod.mps_is_available(_Torch()) is False


def test_mps_is_available_is_built_raises() -> None:
    class _MpsBuiltRaises:
        def is_built(self):
            raise RuntimeError("boom")

        def is_available(self):
            return True

    class _Backends:
        def __init__(self) -> None:
            self.mps = _MpsBuiltRaises()

    class _Torch:
        def __init__(self) -> None:
            self.backends = _Backends()

        def zeros(self, *args, **kwargs):
            return object()

    device_mod.mps_is_available.cache_clear()
    assert device_mod.mps_is_available(_Torch()) is False


def test_mps_is_available_is_available_not_callable() -> None:
    class _MpsNoAvailable:
        is_available = True

    class _Backends:
        def __init__(self) -> None:
            self.mps = _MpsNoAvailable()

    class _Torch:
        def __init__(self) -> None:
            self.backends = _Backends()

        def zeros(self, *args, **kwargs):
            return object()

    device_mod.mps_is_available.cache_clear()
    assert device_mod.mps_is_available(_Torch()) is False


def test_mps_is_available_is_available_raises() -> None:
    class _MpsAvailableRaises:
        def is_available(self):
            raise RuntimeError("boom")

    class _Backends:
        def __init__(self) -> None:
            self.mps = _MpsAvailableRaises()

    class _Torch:
        def __init__(self) -> None:
            self.backends = _Backends()

        def zeros(self, *args, **kwargs):
            return object()

    device_mod.mps_is_available.cache_clear()
    assert device_mod.mps_is_available(_Torch()) is False


def test_mps_is_available_zeros_raises() -> None:
    class _MpsAvailableTrue:
        def is_available(self):
            return True

    class _Backends:
        def __init__(self) -> None:
            self.mps = _MpsAvailableTrue()

    class _Torch:
        def __init__(self) -> None:
            self.backends = _Backends()

        def zeros(self, *args, **kwargs):
            raise RuntimeError("boom")

    device_mod.mps_is_available.cache_clear()
    assert device_mod.mps_is_available(_Torch()) is False
