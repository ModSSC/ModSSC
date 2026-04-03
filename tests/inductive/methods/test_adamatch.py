import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modssc.inductive.methods.adamatch import AdaMatchMethod, AdaMatchSpec
from modssc.inductive.types import DeviceSpec

try:
    import torch
except ImportError:
    torch = None


def _stop_after_slice(monkeypatch):
    monkeypatch.setattr("modssc.inductive.methods.adamatch.ensure_torch_data", lambda d, device: d)
    monkeypatch.setattr(
        "modssc.inductive.methods.adamatch.ensure_float_tensor", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "modssc.inductive.methods.adamatch.ensure_model_bundle", lambda bundle: bundle
    )
    monkeypatch.setattr(
        "modssc.inductive.methods.adamatch.ensure_model_device", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "modssc.inductive.methods.adamatch.cycle_batches",
        lambda X, y, **_kwargs: iter([(X, y)]),
    )
    monkeypatch.setattr(
        "modssc.inductive.methods.adamatch.cycle_batch_indices",
        lambda *_args, **_kwargs: iter([torch.tensor([0])]),
    )
    monkeypatch.setattr(
        "modssc.inductive.methods.adamatch.extract_logits",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )


@pytest.mark.skipif(torch is None, reason="Torch not available")
def test_adamatch_slice_dict_no_geometric(monkeypatch):
    """Test _slice_inputs with dictionary input when torch_geometric is missing."""

    def mock_optional_import(name, extra=None):
        if name == "torch":
            return torch
        if name == "torch_geometric":
            raise ImportError(f"No module named {name}")
        return MagicMock()

    monkeypatch.setattr("modssc.inductive.methods.adamatch.optional_import", mock_optional_import)
    _stop_after_slice(monkeypatch)

    # Force ImportError for torch_geometric.utils within the method
    # by ensuring it is NOT in sys.modules
    with patch.dict(sys.modules):
        keys_to_remove = [k for k in sys.modules if k.startswith("torch_geometric")]
        for k in keys_to_remove:
            del sys.modules[k]

        # We also need to prevent import from finding it
        import builtins

        real_import = builtins.__import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch_geometric.utils" or (
                name == "torch_geometric" and "utils" in fromlist
            ):
                raise ImportError(f"No module named {name}")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=mock_import):
            spec = AdaMatchSpec(
                model_bundle=MagicMock(model=MagicMock(), optimizer=MagicMock()),
                max_epochs=1,
                batch_size=2,
            )
            method = AdaMatchMethod(spec)

            X_l = {
                "x": torch.randn(10, 5),
                "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                "y": torch.zeros(10),
                "other": torch.randn(10, 2),
            }
            y_l = torch.zeros(10, dtype=torch.long)
            X_u_w = {
                "x": torch.randn(10, 5),
                "edge_index": torch.tensor([[0, 1]], dtype=torch.long),
                "other": torch.randn(10, 2),
            }
            X_u_s = X_u_w

            data = SimpleNamespace(
                X_l=X_l, y_l=y_l, X_u_w=X_u_w, X_u_s=X_u_s, X_u=None, views=None, meta=None
            )

            with pytest.raises(RuntimeError, match="stop"):
                method.fit(data, device=DeviceSpec("cpu"))


@pytest.mark.skipif(torch is None, reason="Torch not available")
def test_adamatch_slice_dict_with_geometric(monkeypatch):
    """Test _slice_inputs with dictionary input when torch_geometric IS present."""

    # Mock optional_import
    def mock_optional_import(name, extra=None):
        if name == "torch":
            return torch
        return MagicMock()

    monkeypatch.setattr("modssc.inductive.methods.adamatch.optional_import", mock_optional_import)
    _stop_after_slice(monkeypatch)

    # Create the mock subgraph function
    mock_subgraph = MagicMock(return_value=(torch.tensor([[0], [1]]), torch.tensor([0, 1])))

    # Setup sys.modules so 'from torch_geometric.utils import subgraph' works
    pyg = types.ModuleType("torch_geometric")
    utils = types.ModuleType("torch_geometric.utils")
    utils.subgraph = mock_subgraph
    pyg.utils = utils

    # Patch sys.modules. Need to patch BOTH top level and submodule
    with patch.dict(sys.modules, {"torch_geometric": pyg, "torch_geometric.utils": utils}):
        spec = AdaMatchSpec(
            model_bundle=MagicMock(model=MagicMock(), optimizer=MagicMock()),
            max_epochs=1,
            batch_size=2,
        )
        method = AdaMatchMethod(spec)

        X_l = {
            "x": torch.randn(10, 5),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        }
        y_l = torch.zeros(10, dtype=torch.long)
        X_u_w = X_l
        X_u_s = X_l

        data = SimpleNamespace(
            X_l=X_l, y_l=y_l, X_u_w=X_u_w, X_u_s=X_u_s, X_u=None, views=None, meta=None
        )

        with pytest.raises(RuntimeError, match="stop"):
            method.fit(data, device=DeviceSpec("cpu"))

        assert mock_subgraph.called, "subgraph should have been called"


@pytest.mark.skipif(torch is None, reason="Torch not available")
def test_adamatch_slice_dict_with_geometric_x_only(monkeypatch):
    def mock_optional_import(name, extra=None):
        if name == "torch":
            return torch
        return MagicMock()

    monkeypatch.setattr("modssc.inductive.methods.adamatch.optional_import", mock_optional_import)
    _stop_after_slice(monkeypatch)

    pyg = types.ModuleType("torch_geometric")
    utils = types.ModuleType("torch_geometric.utils")
    utils.subgraph = MagicMock()
    pyg.utils = utils

    with patch.dict(sys.modules, {"torch_geometric": pyg, "torch_geometric.utils": utils}):
        spec = AdaMatchSpec(
            model_bundle=MagicMock(model=MagicMock(), optimizer=MagicMock()),
            max_epochs=1,
            batch_size=2,
        )
        method = AdaMatchMethod(spec)
        X_l = torch.randn(10, 5)
        y_l = torch.zeros(10, dtype=torch.long)
        X_u_w = {"x": torch.randn(10, 5), "other": torch.randn(10, 2)}
        X_u_s = X_u_w
        data = SimpleNamespace(
            X_l=X_l, y_l=y_l, X_u_w=X_u_w, X_u_s=X_u_s, X_u=None, views=None, meta=None
        )

        with pytest.raises(RuntimeError, match="stop"):
            method.fit(data, device=DeviceSpec("cpu"))

        assert not utils.subgraph.called


@pytest.mark.skipif(torch is None, reason="Torch not available")
def test_adamatch_slice_dict_with_geometric_edge_index_only(monkeypatch):
    class _FeatDict(dict):
        @property
        def shape(self):
            return self["feat"].shape

    def mock_optional_import(name, extra=None):
        if name == "torch":
            return torch
        return MagicMock()

    monkeypatch.setattr("modssc.inductive.methods.adamatch.optional_import", mock_optional_import)
    _stop_after_slice(monkeypatch)

    mock_subgraph = MagicMock(return_value=(torch.tensor([[0], [0]]), torch.tensor([0])))
    pyg = types.ModuleType("torch_geometric")
    utils = types.ModuleType("torch_geometric.utils")
    utils.subgraph = mock_subgraph
    pyg.utils = utils

    with patch.dict(sys.modules, {"torch_geometric": pyg, "torch_geometric.utils": utils}):
        spec = AdaMatchSpec(
            model_bundle=MagicMock(model=MagicMock(), optimizer=MagicMock()),
            max_epochs=1,
            batch_size=2,
        )
        method = AdaMatchMethod(spec)
        X_l = torch.randn(10, 5)
        y_l = torch.zeros(10, dtype=torch.long)
        X_u_w = _FeatDict(
            {
                "feat": torch.randn(10, 5),
                "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            }
        )
        X_u_s = X_u_w
        data = SimpleNamespace(
            X_l=X_l, y_l=y_l, X_u_w=X_u_w, X_u_s=X_u_s, X_u=None, views=None, meta=None
        )

        with pytest.raises(RuntimeError, match="stop"):
            method.fit(data, device=DeviceSpec("cpu"))

        assert mock_subgraph.called
