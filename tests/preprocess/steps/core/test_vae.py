from __future__ import annotations

import importlib
import json

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


def test_vae_expected_model_fingerprint_must_not_be_empty() -> None:
    step = VaeStep(expected_model_fingerprint="")
    store = ArtifactStore()
    store.set("features.X", np.zeros((3, 2), dtype=np.float32))

    with pytest.raises(PreprocessValidationError, match="must not be empty"):
        step.fit(store, fit_indices=np.array([0, 1]), rng=np.random.default_rng(0))


def test_vae_expected_model_fingerprint_is_not_self_referential() -> None:
    first = VaeStep(expected_model_fingerprint="vae_first")
    second = VaeStep(expected_model_fingerprint="vae_second")

    assert "expected_model_fingerprint" not in first._params_for_fingerprint()
    assert first._params_for_fingerprint() == second._params_for_fingerprint()
    kwargs = {
        "fit_shape": (4, 3),
        "fit_data_hash": "data",
        "fit_indices_hash": "indices",
        "seed": 7,
    }
    assert first._model_fingerprint(**kwargs) == second._model_fingerprint(**kwargs)


def test_vae_model_fingerprint_commits_implementation_identity(monkeypatch) -> None:
    step = VaeStep()
    kwargs = {
        "fit_shape": (4, 3),
        "fit_data_hash": "data",
        "fit_indices_hash": "indices",
        "seed": 7,
    }
    monkeypatch.setattr(vae_module, "_vae_implementation_sha256", lambda: "a" * 64)
    first = step._model_fingerprint(**kwargs)
    monkeypatch.setattr(vae_module, "_vae_implementation_sha256", lambda: "b" * 64)
    second = step._model_fingerprint(**kwargs)

    assert first != second


def test_vae_rejects_a_computed_model_fingerprint_mismatch() -> None:
    step = VaeStep(
        latent_dim=2,
        hidden_dims=(4,),
        epochs=1,
        batch_size=2,
        model_seed=7,
        model_cache=False,
        expected_model_fingerprint="vae_wrong",
    )
    store = ArtifactStore()
    store.set("features.X", np.arange(12, dtype=np.float32).reshape(4, 3))

    with pytest.raises(PreprocessValidationError, match="fingerprint differs"):
        step.fit(store, fit_indices=np.arange(4), rng=np.random.default_rng(0))


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
    step = VaeStep(preset="graphlearning_mnist_vae2")
    store = ArtifactStore()
    store.set("features.X", np.zeros((3, 512), dtype=np.float32))

    with pytest.raises(PreprocessValidationError, match="expected input_dim=784"):
        step.fit(store, fit_indices=np.array([0, 1]), rng=np.random.default_rng(0))


def test_vae_poisson_mnist_alias_uses_canonical_graphlearning_preset() -> None:
    canonical = VaeStep(preset="graphlearning_mnist_vae2")
    legacy = VaeStep(preset="poisson_mnist")

    canonical._validate_params()
    legacy._validate_params()

    assert legacy.preset == "graphlearning_mnist_vae2"
    assert legacy._params_for_fingerprint() == canonical._params_for_fingerprint()


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
    output_info = result["features.vae.info"]["output"]
    assert output_info["content_sha256"] == vae_module._array_sha256(result["features.vae"])
    assert output_info["identity_schema_version"] == 1
    assert output_info["identity_fingerprint"].startswith("vae_output_")
    assert result["features.vae.info"]["runtime"]["torch_version"]
    assert result["features.vae.info"]["training_runtime"]["numpy_version"]


def test_vae_runtime_identity_commits_to_actual_latent_content() -> None:
    step = VaeStep()
    step.model_info_ = {
        "fingerprint": "vae_semantic",
        "training_runtime": {"torch_version": "training-runtime"},
        "runtime": {
            "python_version": "3.x",
            "numpy_version": "2.x",
            "torch_version": "inference-runtime",
            "device": "cpu",
        },
    }
    latent = np.arange(12, dtype=np.float32).reshape(6, 2)

    first = step.runtime_artifacts(
        produced={"features.vae": latent},
        split="train",
    )["features.vae.info"]
    repeated = step.runtime_artifacts(
        produced={"features.vae": latent.copy()},
        split="train",
    )["features.vae.info"]
    changed_latent = latent.copy()
    changed_latent[3, 1] += 1.0
    changed = step.runtime_artifacts(
        produced={"features.vae": changed_latent},
        split="train",
    )["features.vae.info"]

    assert first["output"]["content_sha256"] == vae_module._array_sha256(latent)
    assert first["output"]["identity_fingerprint"] == repeated["output"]["identity_fingerprint"]
    assert first["output"]["identity_fingerprint"] != changed["output"]["identity_fingerprint"]
    assert first["runtime"]["split"] == "train"
    assert first["runtime"]["torch_version"] == "inference-runtime"


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
    publisher = VaeStep(latent_dim=2, hidden_dims=(4,), dropout=0.2, decoder_output="sigmoid")
    publisher.model_ = model
    publisher.mean_ = np.zeros(3, dtype=np.float32)
    publisher.scale_ = np.ones(3, dtype=np.float32)
    publisher.impute_ = np.zeros(3, dtype=np.float32)
    publisher.device_ = "cpu"
    publisher._save_cached_model(
        torch,
        cache_dir=load_dir,
        info={"fingerprint": "vae_test"},
    )
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


def test_vae_verified_manifest_validation_branches_when_torch_available(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    invalid = VaeStep(model_cache=False, require_cache_hit=True)
    with pytest.raises(PreprocessValidationError, match="requires model_cache=true"):
        invalid._validate_params()

    cache_dir = tmp_path / "verified"
    cache_dir.mkdir()
    (cache_dir / "model.pt").write_bytes(b"model")
    (cache_dir / "state.npz").write_bytes(b"state")
    step = VaeStep(latent_dim=2, hidden_dims=(4,))

    (cache_dir / "manifest.json").write_text("[]", encoding="utf-8")
    assert not step._load_cached_model(
        torch,
        cache_dir=cache_dir,
        input_dim=3,
        device="cpu",
        expected_fingerprint="vae_expected",
    )
    (cache_dir / "manifest.json").write_text(
        '{"fingerprint":"vae_other","file_sha256":{}}', encoding="utf-8"
    )
    assert not step._load_cached_model(
        torch,
        cache_dir=cache_dir,
        input_dim=3,
        device="cpu",
        expected_fingerprint="vae_expected",
    )
    (cache_dir / "manifest.json").write_text('{"fingerprint":"vae_expected"}', encoding="utf-8")
    assert not step._load_cached_model(
        torch,
        cache_dir=cache_dir,
        input_dim=3,
        device="cpu",
        expected_fingerprint="vae_expected",
    )


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


def test_vae_fit_scope_all_max_fit_samples_uses_all_scope_indices_when_torch_available() -> None:
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
        max_fit_samples=3,
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
        max_fit_samples=3,
    )
    second.fit(second_store, fit_indices=np.array([2, 3]), rng=np.random.default_rng(999))
    second_info = second.transform(second_store, rng=np.random.default_rng(0))["features.vae.info"]

    assert first_info["training"]["n_fit_samples"] == 3
    assert first_info["fingerprint"] == second_info["fingerprint"]


def test_vae_graphlearning_mnist_vae2_preset_when_torch_available() -> None:
    pytest.importorskip("torch")
    step = VaeStep(
        preset="graphlearning_mnist_vae2",
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
    assert info["params"]["preset"] == "graphlearning_mnist_vae2"
    assert info["params"]["hidden_dims"] == [400]
    assert info["params"]["input_scaling"] == "global_minmax"
    assert info["params"]["reconstruction_loss"] == "bce"


@pytest.mark.parametrize("input_scaling", ["standardize", "global_minmax"])
def test_vae_model_cache_reuses_checkpoint_when_torch_available(
    tmp_path, input_scaling: str
) -> None:
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
        input_scaling=input_scaling,
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
        input_scaling=input_scaling,
        expected_model_fingerprint=first.model_fingerprint_,
    )
    second.fit(second_store, fit_indices=np.array([0, 1, 2, 3]), rng=np.random.default_rng(999))
    second_result = second.transform(second_store, rng=np.random.default_rng(0))

    assert second_result["features.vae.info"]["cache"]["hit"] is True
    assert second_result["features.vae.info"]["expected_fingerprint"] == first.model_fingerprint_
    assert "expected_model_fingerprint" not in second_result["features.vae.info"]["params"]
    assert (
        first_result["features.vae.info"]["fingerprint"]
        == second_result["features.vae.info"]["fingerprint"]
    )
    np.testing.assert_allclose(first_result["features.vae"], second_result["features.vae"])


def test_vae_frozen_cache_requires_verified_checkpoint_when_torch_available(tmp_path) -> None:
    pytest.importorskip("torch")
    X = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 1.0, 0.0], [3.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    store = ArtifactStore()
    store.set("features.X", X)
    frozen = VaeStep(
        latent_dim=2,
        hidden_dims=(4,),
        epochs=1,
        batch_size=2,
        device="cpu",
        model_cache_dir=str(tmp_path),
        cache_key="frozen",
        model_seed=123,
        require_cache_hit=True,
    )

    with pytest.raises(PreprocessValidationError, match="frozen VAE cache"):
        frozen.fit(store, fit_indices=np.arange(4), rng=np.random.default_rng(0))


def test_vae_frozen_cache_rejects_legacy_layout_when_torch_available(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "manifest.json").write_text("{}", encoding="utf-8")
    (legacy / "model.pt").write_bytes(b"legacy")
    np.savez(
        legacy / "state.npz",
        mean=np.zeros(3, dtype=np.float32),
        scale=np.ones(3, dtype=np.float32),
        impute=np.zeros(3, dtype=np.float32),
    )
    store = ArtifactStore()
    store.set("features.X", np.arange(12, dtype=np.float32).reshape(4, 3))
    frozen = VaeStep(
        latent_dim=2,
        hidden_dims=(4,),
        epochs=1,
        batch_size=2,
        device="cpu",
        model_seed=123,
        require_cache_hit=True,
    )
    monkeypatch.setattr(frozen, "_cache_dir_for", lambda _fingerprint: legacy)

    with pytest.raises(PreprocessValidationError, match="frozen VAE cache"):
        frozen.fit(store, fit_indices=np.arange(4), rng=np.random.default_rng(0))


def test_vae_verified_cache_rejects_tampering_when_torch_available(tmp_path) -> None:
    pytest.importorskip("torch")
    X = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 1.0, 0.0], [3.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    store = ArtifactStore()
    store.set("features.X", X)
    first = VaeStep(
        latent_dim=2,
        hidden_dims=(4,),
        epochs=1,
        batch_size=2,
        device="cpu",
        model_cache_dir=str(tmp_path),
        cache_key="tampered",
        model_seed=123,
    )
    first.fit(store, fit_indices=np.arange(4), rng=np.random.default_rng(0))
    assert first.model_cache_dir_ is not None
    pointer = json.loads((first.model_cache_dir_ / "CURRENT.json").read_text(encoding="utf-8"))
    state_path = first.model_cache_dir_ / "generations" / pointer["generation"] / "state.npz"
    with state_path.open("ab") as handle:
        handle.write(b"tampered")

    frozen = VaeStep(
        latent_dim=2,
        hidden_dims=(4,),
        epochs=1,
        batch_size=2,
        device="cpu",
        model_cache_dir=str(tmp_path),
        cache_key="tampered",
        model_seed=123,
        require_cache_hit=True,
    )
    with pytest.raises(PreprocessValidationError, match="frozen VAE cache"):
        frozen.fit(store, fit_indices=np.arange(4), rng=np.random.default_rng(0))
