from __future__ import annotations

import importlib

import numpy as np
import pytest

import modssc.preprocess.steps.core.vae as vae_module
from modssc.preprocess.errors import OptionalDependencyError, PreprocessValidationError
from modssc.preprocess.steps.core.vae import VaeStep
from modssc.preprocess.store import ArtifactStore


def test_module_importable() -> None:
    importlib.import_module("modssc.preprocess.steps.core.vae")


def test_vae_transform_before_fit() -> None:
    step = VaeStep()
    store = ArtifactStore()
    store.set("features.X", np.zeros((3, 2), dtype=np.float32))

    with pytest.raises(PreprocessValidationError, match="called before fit"):
        step.transform(store, rng=np.random.default_rng(0))


def test_vae_invalid_params() -> None:
    step = VaeStep(latent_dim=0)
    store = ArtifactStore()
    store.set("features.X", np.zeros((3, 2), dtype=np.float32))

    with pytest.raises(PreprocessValidationError, match="latent_dim must be > 0"):
        step.fit(store, fit_indices=np.array([0, 1]), rng=np.random.default_rng(0))


def test_vae_invalid_model_seed() -> None:
    step = VaeStep(model_seed=-1)
    store = ArtifactStore()
    store.set("features.X", np.zeros((3, 2), dtype=np.float32))

    with pytest.raises(PreprocessValidationError, match="model_seed must be >= 0"):
        step.fit(store, fit_indices=np.array([0, 1]), rng=np.random.default_rng(0))


def test_vae_invalid_fit_scope() -> None:
    step = VaeStep(fit_scope="split")
    store = ArtifactStore()
    store.set("features.X", np.zeros((3, 2), dtype=np.float32))

    with pytest.raises(PreprocessValidationError, match="fit_scope must be one of"):
        step.fit(store, fit_indices=np.array([0, 1]), rng=np.random.default_rng(0))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epochs": 0}, "epochs must be > 0"),
        ({"batch_size": 0}, "batch_size must be > 0"),
        ({"lr": 0.0}, "lr must be > 0"),
        ({"beta": -1.0}, "beta must be >= 0"),
        ({"dropout": 1.0}, "dropout must be in"),
        ({"max_fit_samples": 0}, "max_fit_samples must be > 0"),
        ({"expected_input_dim": 0}, "expected_input_dim must be > 0"),
        ({"hidden_dims": (4, 0)}, "hidden_dims must contain"),
        ({"input_scaling": "bad"}, "input_scaling must be one of"),
        ({"reconstruction_loss": "bad"}, "reconstruction_loss must be"),
        ({"decoder_output": "bad"}, "decoder_output must be"),
    ],
)
def test_vae_validation_matrix(kwargs: dict[str, object], message: str) -> None:
    step = VaeStep(**kwargs)
    store = ArtifactStore()
    store.set("features.X", np.zeros((3, 2), dtype=np.float32))

    with pytest.raises(PreprocessValidationError, match=message):
        step.fit(store, fit_indices=np.array([0, 1]), rng=np.random.default_rng(0))


def test_vae_unknown_preset() -> None:
    step = VaeStep(preset="missing")
    store = ArtifactStore()
    store.set("features.X", np.zeros((3, 784), dtype=np.float32))

    with pytest.raises(PreprocessValidationError, match="Unknown VAE preset"):
        step.fit(store, fit_indices=np.array([0, 1]), rng=np.random.default_rng(0))


def test_vae_preset_validates_expected_input_dim() -> None:
    step = VaeStep(preset="poisson_mnist")
    store = ArtifactStore()
    store.set("features.X", np.zeros((3, 512), dtype=np.float32))

    with pytest.raises(PreprocessValidationError, match="expected input_dim=784"):
        step.fit(store, fit_indices=np.array([0, 1]), rng=np.random.default_rng(0))


def test_vae_bce_requires_sigmoid() -> None:
    step = VaeStep(reconstruction_loss="bce", decoder_output="linear")
    store = ArtifactStore()
    store.set("features.X", np.zeros((3, 2), dtype=np.float32))

    with pytest.raises(PreprocessValidationError, match="requires decoder_output=sigmoid"):
        step.fit(store, fit_indices=np.array([0, 1]), rng=np.random.default_rng(0))


def test_vae_invalid_input_dim() -> None:
    step = VaeStep()
    store = ArtifactStore()
    store.set("features.X", np.array([1.0, 2.0, 3.0], dtype=np.float32))

    with pytest.raises(PreprocessValidationError, match="expects a 2D features.X matrix"):
        step.fit(store, fit_indices=np.array([0, 1]), rng=np.random.default_rng(0))


def test_vae_private_helpers_and_runtime_errors() -> None:
    class DenseLike:
        def toarray(self):
            return np.ones((2, 3), dtype=np.float32)

    class FallbackSubset:
        def __init__(self) -> None:
            self.values = np.arange(6, dtype=np.float32).reshape(3, 2)

        def __getitem__(self, _idx):
            raise RuntimeError("force fallback")

        def __array__(self, dtype=None):
            return np.asarray(self.values, dtype=dtype)

    class ArrayOnly:
        def __array__(self, dtype=None):
            return np.asarray(np.arange(6, dtype=np.float32).reshape(3, 2), dtype=dtype)

    np.testing.assert_array_equal(
        vae_module._as_dense_2d(DenseLike(), name="features.X"),
        np.ones((2, 3), dtype=np.float32),
    )
    with pytest.raises(PreprocessValidationError, match="fit_indices must be 1D"):
        vae_module._subset_rows(np.zeros((3, 2)), fit_indices=np.zeros((1, 1), dtype=np.int64))
    np.testing.assert_array_equal(
        vae_module._subset_rows(FallbackSubset(), fit_indices=np.array([0, 2])),
        np.array([[0.0, 1.0], [4.0, 5.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        vae_module._subset_rows(ArrayOnly(), fit_indices=np.array([1])),
        np.array([[2.0, 3.0]], dtype=np.float32),
    )
    cleaned = vae_module._clean_array(
        np.array([[np.nan, 1.0], [2.0, np.inf]], dtype=np.float32),
        impute=np.array([9.0, 8.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(cleaned, np.array([[9.0, 1.0], [2.0, 8.0]], dtype=np.float32))
    assert vae_module._safe_cache_component("../bad key!") == "bad_key"

    step = VaeStep()
    with pytest.raises(PreprocessValidationError, match="before fit"):
        step.runtime_artifacts()
    with pytest.raises(PreprocessValidationError, match="scaling state is missing"):
        step._scale_features(np.zeros((2, 2), dtype=np.float32))


def test_vae_fit_empty_selection() -> None:
    step = VaeStep()
    store = ArtifactStore()
    store.set("features.X", np.zeros((3, 2), dtype=np.float32))

    with pytest.raises(PreprocessValidationError, match="Cannot fit VAE on empty"):
        step.fit(store, fit_indices=np.array([], dtype=np.int64), rng=np.random.default_rng(0))


def test_vae_requires_torch_after_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_require_optional(**kwargs):
        raise OptionalDependencyError(extra=str(kwargs["extra"]), purpose=kwargs.get("purpose"))

    monkeypatch.setattr(vae_module, "require_optional", fake_require_optional)
    step = VaeStep(latent_dim=2, hidden_dims=(4,), epochs=1, batch_size=2, device="cpu")
    store = ArtifactStore()
    store.set("features.X", np.zeros((4, 3), dtype=np.float32))

    with pytest.raises(OptionalDependencyError):
        step.fit(store, fit_indices=np.array([0, 1, 2, 3]), rng=np.random.default_rng(0))


def test_vae_fit_transform_when_torch_available() -> None:
    pytest.importorskip("torch")
    step = VaeStep(
        latent_dim=2,
        hidden_dims=(4,),
        epochs=1,
        batch_size=2,
        device="cpu",
        model_cache=False,
    )
    store = ArtifactStore()
    X = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 1.0, 0.0], [3.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    store.set("features.X", X)

    try:
        step.fit(store, fit_indices=np.array([0, 1, 2, 3]), rng=np.random.default_rng(0))
    except OptionalDependencyError:
        pytest.skip("torch optional dependency is unavailable")

    result = step.transform(store, rng=np.random.default_rng(0))
    assert set(result) == {"features.vae", "features.vae.info"}
    assert result["features.vae"].shape == (4, 2)
    assert result["features.vae"].dtype == np.float32
    assert result["features.vae.info"]["training"]["uses_labels"] is False


def test_vae_paper_style_fit_transform_when_torch_available() -> None:
    pytest.importorskip("torch")
    step = VaeStep(
        latent_dim=2,
        hidden_dims=(4,),
        epochs=1,
        batch_size=2,
        device="cpu",
        model_cache=False,
        input_scaling="minmax",
        reconstruction_loss="bce",
        decoder_output="sigmoid",
    )
    store = ArtifactStore()
    X = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 1.0, 0.0], [3.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    store.set("features.X", X)

    try:
        step.fit(store, fit_indices=np.array([0, 1, 2, 3]), rng=np.random.default_rng(0))
    except OptionalDependencyError:
        pytest.skip("torch optional dependency is unavailable")

    result = step.transform(store, rng=np.random.default_rng(0))
    assert result["features.vae"].shape == (4, 2)
    assert np.isfinite(result["features.vae"]).all()


def test_vae_scaling_branches_and_nonfinite_fit_when_torch_available() -> None:
    pytest.importorskip("torch")
    X = np.array(
        [[np.nan, 1.0, 2.0], [1.0, np.inf, 2.0], [1.0, 1.0, 2.0], [1.0, 1.0, 2.0]],
        dtype=np.float32,
    )
    for scaling in ("none", "global_minmax"):
        store = ArtifactStore()
        store.set("features.X", X)
        step = VaeStep(
            latent_dim=2,
            hidden_dims=(4,),
            epochs=1,
            batch_size=2,
            device="cpu",
            model_cache=False,
            input_scaling=scaling,
            max_fit_samples=2,
            model_seed=11,
        )
        step.fit(store, fit_indices=np.array([0, 1, 2, 3]), rng=np.random.default_rng(0))
        result = step.transform(store, rng=np.random.default_rng(0))
        assert result["features.vae"].shape == (4, 2)

    constant_store = ArtifactStore()
    constant_store.set("features.X", np.ones((4, 3), dtype=np.float32))
    constant = VaeStep(
        latent_dim=2,
        hidden_dims=(4,),
        epochs=1,
        batch_size=2,
        device="cpu",
        model_cache=False,
        input_scaling="global_minmax",
    )
    constant.fit(constant_store, fit_indices=np.array([0, 1, 2, 3]), rng=np.random.default_rng(0))
    assert float(np.asarray(constant.scale_)) == 1.0


def test_vae_model_helpers_cache_failures_and_dropout_when_torch_available(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    step = VaeStep(latent_dim=2, hidden_dims=(4,), dropout=0.2, decoder_output="sigmoid")
    model = step._make_model(torch, input_dim=3)
    recon, mu, logvar = model(torch.zeros((2, 3), dtype=torch.float32))
    assert recon.shape == (2, 3)
    assert mu.shape == (2, 2)
    assert logvar.shape == (2, 2)

    step._save_cached_model(torch, cache_dir=tmp_path / "empty", info={})
    assert not (tmp_path / "empty").exists()

    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (corrupt_dir / "model.pt").write_bytes(b"not a model")
    (corrupt_dir / "state.npz").write_bytes(b"not npz")
    assert step._load_cached_model(torch, cache_dir=corrupt_dir, input_dim=3, device="cpu") is False

    load_dir = tmp_path / "load"
    load_dir.mkdir()
    np.savez(
        load_dir / "state.npz",
        mean=np.zeros(3, dtype=np.float32),
        scale=np.ones(3, dtype=np.float32),
        impute=np.zeros(3, dtype=np.float32),
    )
    (load_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (load_dir / "model.pt").write_bytes(b"placeholder")
    expected_state = model.state_dict()
    calls = {"count": 0}

    def fake_load(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TypeError("old torch")
        return {"state_dict": expected_state}

    monkeypatch.setattr(torch, "load", fake_load)
    cached = VaeStep(latent_dim=2, hidden_dims=(4,), dropout=0.2, decoder_output="sigmoid")
    assert cached._load_cached_model(torch, cache_dir=load_dir, input_dim=3, device="cpu") is True
    assert calls["count"] == 2


def test_vae_fit_scope_all_ignores_sampling_indices_when_torch_available() -> None:
    pytest.importorskip("torch")
    X = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 1.0, 0.0], [3.0, 2.0, 1.0]],
        dtype=np.float32,
    )

    first_store = ArtifactStore()
    first_store.set("features.X", X)
    first = VaeStep(
        latent_dim=2,
        hidden_dims=(4,),
        epochs=1,
        batch_size=2,
        device="cpu",
        model_cache=False,
        model_seed=7,
        fit_scope="all",
    )
    first.fit(first_store, fit_indices=np.array([0, 1]), rng=np.random.default_rng(0))
    first_info = first.transform(first_store, rng=np.random.default_rng(0))["features.vae.info"]

    second_store = ArtifactStore()
    second_store.set("features.X", X)
    second = VaeStep(
        latent_dim=2,
        hidden_dims=(4,),
        epochs=1,
        batch_size=2,
        device="cpu",
        model_cache=False,
        model_seed=7,
        fit_scope="all",
    )
    second.fit(second_store, fit_indices=np.array([2, 3]), rng=np.random.default_rng(999))
    second_info = second.transform(second_store, rng=np.random.default_rng(0))["features.vae.info"]

    assert first_info["training"]["n_fit_samples"] == 4
    assert first_info["training"]["fit_scope"] == "all"
    assert first_info["fingerprint"] == second_info["fingerprint"]


def test_vae_poisson_mnist_preset_when_torch_available() -> None:
    pytest.importorskip("torch")
    step = VaeStep(
        preset="poisson_mnist",
        epochs=1,
        device="cpu",
        model_cache=False,
    )
    store = ArtifactStore()
    X = np.arange(4 * 784, dtype=np.float32).reshape(4, 784)
    store.set("features.X", X)

    try:
        step.fit(store, fit_indices=np.array([0, 1, 2, 3]), rng=np.random.default_rng(0))
    except OptionalDependencyError:
        pytest.skip("torch optional dependency is unavailable")

    result = step.transform(store, rng=np.random.default_rng(0))
    assert result["features.vae"].shape == (4, 20)
    info = result["features.vae.info"]
    assert info["params"]["preset"] == "poisson_mnist"
    assert info["params"]["hidden_dims"] == [400]
    assert info["params"]["input_scaling"] == "global_minmax"
    assert info["params"]["reconstruction_loss"] == "bce"


def test_vae_model_cache_reuses_checkpoint_when_torch_available(tmp_path) -> None:
    pytest.importorskip("torch")
    X = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 1.0, 0.0], [3.0, 2.0, 1.0]],
        dtype=np.float32,
    )

    first_store = ArtifactStore()
    first_store.set("features.X", X)
    first = VaeStep(
        latent_dim=2,
        hidden_dims=(4,),
        epochs=1,
        batch_size=2,
        device="cpu",
        model_cache_dir=str(tmp_path),
        cache_key="toy",
        model_seed=123,
    )
    first.fit(first_store, fit_indices=np.array([0, 1, 2, 3]), rng=np.random.default_rng(0))
    first_result = first.transform(first_store, rng=np.random.default_rng(0))

    second_store = ArtifactStore()
    second_store.set("features.X", X)
    second = VaeStep(
        latent_dim=2,
        hidden_dims=(4,),
        epochs=1,
        batch_size=2,
        device="cpu",
        model_cache_dir=str(tmp_path),
        cache_key="toy",
        model_seed=123,
    )
    second.fit(second_store, fit_indices=np.array([0, 1, 2, 3]), rng=np.random.default_rng(999))
    second_result = second.transform(second_store, rng=np.random.default_rng(0))

    assert second_result["features.vae.info"]["cache"]["hit"] is True
    assert (
        first_result["features.vae.info"]["fingerprint"]
        == second_result["features.vae.info"]["fingerprint"]
    )
    np.testing.assert_allclose(first_result["features.vae"], second_result["features.vae"])
