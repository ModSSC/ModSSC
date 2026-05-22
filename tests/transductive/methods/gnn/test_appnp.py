from __future__ import annotations

import importlib

import pytest


def _assert_module_importable(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or ""
        if missing.startswith("modssc"):
            raise
        pytest.skip(f"Optional dependency missing while importing {module_name}: {missing}")
    except Exception as exc:
        if exc.__class__.__name__ == "OptionalDependencyError" or 'pip install "modssc[' in str(
            exc
        ):
            pytest.skip(f"Optional dependency missing while importing {module_name}: {exc}")
        raise


def test_module_importable() -> None:
    _assert_module_importable("modssc.transductive.methods.gnn.appnp")


def test_spec_defaults() -> None:
    module = _assert_module_importable("modssc.transductive.methods.gnn.appnp")
    spec = module.APPNPSpec()
    assert spec.hidden_dim == 64
    assert spec.dropout == pytest.approx(0.5)
    assert spec.k == 10
    assert spec.alpha == pytest.approx(0.1)
    assert spec.lr == pytest.approx(0.01)
    assert spec.weight_decay == pytest.approx(5e-3)
    assert spec.max_epochs == 10000
    assert spec.patience == 100
    assert spec.norm_mode == "sym"
    assert spec.adjacency_dropout == pytest.approx(0.0)
    assert spec.weight_decay_scope == "first_layer"
    assert spec.selection_metric == "val_acc_then_loss_reset_any"
