from __future__ import annotations

import importlib

import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from modssc.inductive.deep.types import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods import mixmatch


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
    _assert_module_importable("modssc.inductive.methods.mixmatch")


def _make_bundle(model: torch.nn.Module, *, meta=None) -> TorchModelBundle:
    return TorchModelBundle(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        meta=meta,
    )


def test_mixmatch_bundle_helpers_cover_discrete_input_variants() -> None:
    plain_bundle = _make_bundle(torch.nn.Linear(2, 2), meta=["not-a-mapping"])
    assert mixmatch._bundle_prefers_manifold_mixup(plain_bundle) is False

    class _EmbModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.fc = torch.nn.Linear(4, 2)

        def get_input_embeddings(self, x):
            return self.embedding(x.to(dtype=torch.long))

        def forward(self, x):
            emb = self.get_input_embeddings(x)
            return self.fc(emb.mean(dim=1))

    class _WrappedEmbModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.module = _EmbModel()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return self.module(x)

    assert mixmatch._bundle_supports_discrete_inputs(_make_bundle(_EmbModel(), meta={})) is True
    assert (
        mixmatch._bundle_supports_discrete_inputs(_make_bundle(_WrappedEmbModel(), meta={})) is True
    )


def test_mixmatch_ensure_tensor_input_allow_discrete_branches() -> None:
    x_dict = {"x": torch.ones((2, 3), dtype=torch.float32), "meta": "keep"}
    mixmatch._ensure_mixmatch_tensor_input(x_dict, name="X_u_w", allow_discrete=True)

    with pytest.raises(InductiveValidationError, match="X_u_s must be a torch.Tensor"):
        mixmatch._ensure_mixmatch_tensor_input([1, 2, 3], name="X_u_s", allow_discrete=True)

    with pytest.raises(
        InductiveValidationError,
        match="floating point or integer token ids",
    ):
        mixmatch._ensure_mixmatch_tensor_input(
            torch.tensor([[True, False]]),
            name="X_u_s",
            allow_discrete=True,
        )


def test_mixmatch_forward_features_meta_validation_and_tuple_fallback() -> None:
    class _TupleNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.fc(x), x + 1.0

    x = torch.randn(2, 2)
    bundle_tuple = _make_bundle(_TupleNet(), meta=["not-a-mapping"])
    feats = mixmatch._forward_features(bundle_tuple, x)
    assert torch.allclose(feats, x + 1.0)

    bundle_bad = _make_bundle(
        torch.nn.Linear(2, 2),
        meta={"forward_features": lambda _x: "bad"},
    )
    with pytest.raises(InductiveValidationError, match="must return torch.Tensor"):
        mixmatch._forward_features(bundle_bad, x)
