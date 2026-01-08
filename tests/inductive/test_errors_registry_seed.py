from __future__ import annotations

import random

import numpy as np
import pytest

from modssc.inductive import registry
from modssc.inductive.base import MethodInfo
from modssc.inductive.errors import (
    InductiveNotImplementedError,
    InductiveValidationError,
    OptionalDependencyError,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.seed import make_numpy_rng, seed_everything


def test_optional_dependency_error_str_includes_hint():
    err = OptionalDependencyError("torch", "inductive-torch")
    assert "Optional dependency" in str(err)
    assert 'pip install "modssc[inductive-torch]"' in str(err)
    err2 = OptionalDependencyError("torch", "inductive-torch", message="boom")
    assert "boom" in str(err2)


def test_inductive_not_implemented_error_message():
    err = InductiveNotImplementedError("foo", hint="missing")
    assert "foo" in str(err)
    assert "missing" in str(err)
    err2 = InductiveNotImplementedError("bar")
    assert "bar" in str(err2)


def test_optional_import_success():
    mod = optional_import("math", extra="inductive-torch")
    assert hasattr(mod, "sqrt")


def test_make_numpy_rng_seeded():
    rng = make_numpy_rng(123)
    assert rng.integers(0, 10) == 0


def test_seed_everything_success():
    seed_everything(7)
    r1 = random.randint(0, 10)
    n1 = int(np.random.randint(0, 10))
    seed_everything(7)
    r2 = random.randint(0, 10)
    n2 = int(np.random.randint(0, 10))
    assert r1 == r2
    assert n1 == n2


def test_seed_everything_cuda_and_deterministic(monkeypatch):
    import torch

    called = {"cuda": 0, "det": 0}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _manual_seed_all(_seed):
        called["cuda"] += 1

    monkeypatch.setattr(torch.cuda, "manual_seed_all", _manual_seed_all)

    def _use_det(_flag):
        called["det"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(torch, "use_deterministic_algorithms", _use_det)
    seed_everything(42, deterministic=True)
    assert called["cuda"] >= 1
    assert called["det"] == 1

    seed_everything(42, deterministic=False)
    assert called["cuda"] >= 2


def test_seed_everything_no_deterministic_flags(monkeypatch):
    import torch

    if hasattr(torch, "use_deterministic_algorithms"):
        monkeypatch.delattr(torch, "use_deterministic_algorithms", raising=False)
    if hasattr(torch.backends, "cudnn"):
        monkeypatch.delattr(torch.backends, "cudnn", raising=False)

    seed_everything(1, deterministic=True)


def test_seed_everything_missing_cudnn_backend(monkeypatch):
    import types

    import torch

    monkeypatch.setattr(torch, "backends", types.SimpleNamespace(), raising=False)
    seed_everything(1, deterministic=True)


def test_seed_everything_optional_dependency(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise OptionalDependencyError("torch", "inductive-torch")

    monkeypatch.setattr("modssc.inductive.seed.optional_import", _boom)
    seed_everything(123)


def test_registry_register_errors():
    with pytest.raises(ValueError):
        registry.register_method("", "modssc.inductive.methods.pseudo_label:PseudoLabelMethod")
    with pytest.raises(ValueError):
        registry.register_method("bad", "modssc.inductive.methods.pseudo_label")
    registry.register_method(
        "dup_id",
        "modssc.inductive.methods.pseudo_label:PseudoLabelMethod",
    )
    with pytest.raises(ValueError):
        registry.register_method(
            "dup_id",
            "modssc.inductive.methods.pi_model:PiModelMethod",
        )
    with pytest.raises(ValueError):
        registry.register_method(
            "status_id",
            "modssc.inductive.methods.pseudo_label:PseudoLabelMethod",
            status="unknown",  # type: ignore[arg-type]
        )


def test_registry_available_methods_and_get_class():
    methods_all = registry.available_methods(available_only=False)
    assert "vat" in methods_all
    assert "noisy_student" in methods_all
    methods_avail = registry.available_methods(available_only=True)
    assert "vat" in methods_avail
    assert "noisy_student" in methods_avail
    cls = registry.get_method_class("pseudo_label")
    assert hasattr(cls, "info")
    with pytest.raises(KeyError):
        registry.get_method_class("does_not_exist")


def test_registry_get_method_info_and_debug_registry():
    info = registry.get_method_info("pseudo_label")
    assert isinstance(info, MethodInfo)
    assert info.method_id == "pseudo_label"

    registry.register_method(
        "dummy_info",
        "tests.inductive.dummy_registry_module:DummyMethod",
    )
    with pytest.raises(TypeError):
        registry.get_method_info("dummy_info")

    debug = registry._debug_registry()
    assert "pseudo_label" in debug


def test_inductive_validation_error_is_value_error():
    err = InductiveValidationError("bad")
    assert isinstance(err, ValueError)
