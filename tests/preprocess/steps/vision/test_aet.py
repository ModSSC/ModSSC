from __future__ import annotations

import importlib
import os

import numpy as np
import pytest

import modssc.preprocess.steps.vision.aet as aet_module
from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.steps.vision.aet import AetStep, make_aet_regressor
from modssc.preprocess.store import ArtifactStore


def test_module_importable() -> None:
    importlib.import_module("modssc.preprocess.steps.vision.aet")


def test_aet_unknown_preset() -> None:
    step = AetStep(preset="missing")
    store = ArtifactStore()
    store.set("raw.X", np.zeros((1, 32, 32, 3), dtype=np.uint8))

    with pytest.raises(PreprocessValidationError, match="Unknown AET preset"):
        step.transform(store, rng=np.random.default_rng(0))


def test_aet_private_helpers_and_validation_errors(tmp_path) -> None:
    assert aet_module._safe_cache_component("../bad key!") == "bad_key"
    assert aet_module._default_aet_cache_dir().name == "aet_models"
    assert aet_module._default_precomputed_features_path().name == "cifar_aet.npz"
    assert aet_module._default_precomputed_labels_path().name == "cifar_labels.npz"
    assert aet_module._strip_state_prefix({}, "module.") == {}
    assert aet_module._strip_state_prefix({"layer": 1}, "module.") == {"layer": 1}
    assert aet_module._feature_dim("conv1") == 96 * 16 * 16
    assert aet_module._feature_dim("conv2") == 192 * 8 * 8
    assert aet_module._feature_dim("classifier") == 192
    with pytest.raises(PreprocessValidationError, match="Unknown AET feature layer"):
        aet_module._feature_dim("bad")

    npz_path = tmp_path / "data.npz"
    np.savez_compressed(npz_path, present=np.array([1]))
    with pytest.raises(PreprocessValidationError, match="Expected member"):
        aet_module._load_npz_member(npz_path, "missing")

    multi = tmp_path / "multi.npz"
    np.savez_compressed(multi, a=np.array([1]), b=np.array([2]))
    with pytest.raises(PreprocessValidationError, match="exactly one .npy"):
        aet_module._extract_single_npy_from_npz(multi, tmp_path / "out.npy")

    store = ArtifactStore()
    store.set("raw.X", np.zeros((1, 32, 32, 3), dtype=np.uint8))
    for kwargs, message in [
        ({"source": "bad"}, "source must be"),
        ({"batch_size": 0}, "batch_size must be > 0"),
        ({"train_offset": -1}, "train_offset and test_offset"),
        ({"expected_rows": 0}, "expected_rows must be > 0"),
        ({"feature_layer": "bad"}, "feature_layer must be one of"),
        ({"input_scaling": "bad"}, "input_scaling must be"),
    ]:
        with pytest.raises(PreprocessValidationError, match=message):
            AetStep(checkpoint_path=str(tmp_path / "missing.pth"), **kwargs).transform(
                store, rng=np.random.default_rng(0)
            )

    cached_root = AetStep(
        model_cache_dir=str(tmp_path / "models"),
        checkpoint_name="net_epoch_1499.pth",
    )
    assert cached_root._cache_root() == (tmp_path / "models").resolve()
    assert AetStep()._cache_root().name == "aet_models"
    default_checkpoint = cached_root._resolve_checkpoint_path()
    assert default_checkpoint.name == "net_epoch_1499.pth"
    assert cached_root._features_path().name == "cifar_aet.npz"
    assert cached_root._labels_path().name == "cifar_labels.npz"


def test_aet_float32_npy_cache_helper(tmp_path) -> None:
    source32 = tmp_path / "source32.npy"
    output32 = tmp_path / "source32.float32.npy"
    np.save(source32, np.ones((2, 3), dtype=np.float32))
    aet_module._ensure_float32_npy(source32, output32)
    assert not output32.exists()

    source64 = tmp_path / "source64.npy"
    valid_output = tmp_path / "valid.float32.npy"
    np.save(source64, np.arange(6, dtype=np.float64).reshape(2, 3))
    np.save(valid_output, np.full((2, 3), 9.0, dtype=np.float32))
    aet_module._ensure_float32_npy(source64, valid_output)
    meta_path = aet_module._float32_npy_cache_meta_path(valid_output)
    assert meta_path.exists()
    np.testing.assert_array_equal(
        np.load(valid_output, allow_pickle=False),
        np.arange(6, dtype=np.float32).reshape(2, 3),
    )

    aet_module._ensure_float32_npy(source64, valid_output)
    np.testing.assert_array_equal(
        np.load(valid_output, allow_pickle=False),
        np.arange(6, dtype=np.float32).reshape(2, 3),
    )

    np.save(source64, np.arange(6, 12, dtype=np.float64).reshape(2, 3))
    mtime_ns = source64.stat().st_mtime_ns + 1_000_000_000
    os.utime(source64, ns=(mtime_ns, mtime_ns))
    aet_module._ensure_float32_npy(source64, valid_output)
    np.testing.assert_array_equal(
        np.load(valid_output, allow_pickle=False),
        np.arange(6, 12, dtype=np.float32).reshape(2, 3),
    )

    rebuilt_output = tmp_path / "rebuilt.float32.npy"
    np.save(rebuilt_output, np.zeros((1, 3), dtype=np.float32))
    stale_tmp = rebuilt_output.with_suffix(rebuilt_output.suffix + ".tmp")
    stale_tmp.write_bytes(b"stale")
    aet_module._ensure_float32_npy(source64, rebuilt_output)
    assert not stale_tmp.exists()
    rebuilt = np.load(rebuilt_output, allow_pickle=False)
    assert rebuilt.dtype == np.float32
    np.testing.assert_array_equal(
        rebuilt,
        np.arange(6, 12, dtype=np.float32).reshape(2, 3),
    )


def test_aet_missing_checkpoint(tmp_path) -> None:
    step = AetStep(checkpoint_path=str(tmp_path / "missing.pth"))
    store = ArtifactStore()
    store.set("raw.X", np.zeros((1, 32, 32, 3), dtype=np.uint8))

    with pytest.raises(PreprocessValidationError, match="checkpoint not found"):
        step.transform(store, rng=np.random.default_rng(0))


def test_aet_transform_with_checkpoint(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "net_epoch_1499.pth"
    model = make_aet_regressor(torch)
    torch.save(model.state_dict(), checkpoint)

    step = AetStep(checkpoint_path=str(checkpoint), batch_size=2, device="cpu")
    store = ArtifactStore()
    X = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    X[:, 8:24, 8:24, :] = 255
    store.set("raw.X", X)

    result = step.transform(store, rng=np.random.default_rng(0))

    assert set(result) == {"features.aet", "features.aet.info"}
    assert result["features.aet"].shape == (3, 192 * 8 * 8)
    assert result["features.aet"].dtype == np.float32
    norms = np.linalg.norm(result["features.aet"], axis=1)
    np.testing.assert_allclose(norms, np.ones(3), rtol=1.0e-5, atol=1.0e-5)

    info = result["features.aet.info"]
    assert info["preset"] == "poisson_cifar10_projective"
    assert info["training"]["uses_labels"] is False
    assert info["checkpoint"]["path"] == str(checkpoint)
    assert len(info["checkpoint"]["sha256"]) == 64
    assert info["params"]["feature_layer"] == "conv2"


def test_aet_loads_dataparallel_state_dict(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "net_epoch_1499.pth"
    model = make_aet_regressor(torch)
    torch.save({f"module.{key}": value for key, value in model.state_dict().items()}, checkpoint)

    step = AetStep(
        checkpoint_path=str(checkpoint), batch_size=1, device="cpu", feature_layer="classifier"
    )
    store = ArtifactStore()
    store.set("raw.X", np.ones((1, 32, 32, 3), dtype=np.float32))

    result = step.transform(store, rng=np.random.default_rng(0))

    assert result["features.aet"].shape == (1, 192)


def test_aet_checkpoint_typeerror_fallback_and_invalid_payload(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "net_epoch_1499.pth"
    model = make_aet_regressor(torch)
    torch.save(model.state_dict(), checkpoint)
    state_dict = model.state_dict()
    calls = {"count": 0}
    original_load = torch.load
    original_save = torch.save

    def fake_load(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TypeError("old torch")
        return state_dict

    monkeypatch.setattr(torch, "load", fake_load)
    step = AetStep(
        checkpoint_path=str(checkpoint), batch_size=1, device="cpu", feature_layer="classifier"
    )
    store = ArtifactStore()
    store.set("raw.X", np.ones((1, 32, 32, 3), dtype=np.float32))
    result = step.transform(store, rng=np.random.default_rng(0))
    assert result["features.aet"].shape == (1, 192)
    assert calls["count"] == 2

    monkeypatch.setattr(torch, "load", original_load)
    bad_checkpoint = tmp_path / "bad.pth"
    original_save({"not": "a state dict"}, bad_checkpoint)
    with pytest.raises(RuntimeError):
        AetStep(checkpoint_path=str(bad_checkpoint), batch_size=1, device="cpu").transform(
            store, rng=np.random.default_rng(0)
        )


def test_aet_rejects_non_dict_checkpoint_payload(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "bad_payload.pth"
    torch.save([1, 2, 3], checkpoint)
    store = ArtifactStore()
    store.set("raw.X", np.ones((1, 32, 32, 3), dtype=np.float32))

    with pytest.raises(PreprocessValidationError, match="must contain a torch state_dict"):
        AetStep(checkpoint_path=str(checkpoint), batch_size=1, device="cpu").transform(
            store, rng=np.random.default_rng(0)
        )


def test_aet_rejects_non_cifar_preset_shape_after_checkpoint_load(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "net_epoch_1499.pth"
    torch.save(make_aet_regressor(torch).state_dict(), checkpoint)
    preset = dict(aet_module.AET_PRESETS["poisson_cifar10_projective"])
    preset["expected_shape"] = (1, 28, 28)
    monkeypatch.setitem(aet_module.AET_PRESETS, "poisson_cifar10_projective", preset)

    with pytest.raises(PreprocessValidationError, match="currently supports only"):
        AetStep(checkpoint_path=str(checkpoint), batch_size=1, device="cpu")._load_model()


def test_aet_requires_cifar_shape(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "net_epoch_1499.pth"
    torch.save(make_aet_regressor(torch).state_dict(), checkpoint)

    step = AetStep(checkpoint_path=str(checkpoint), batch_size=1, device="cpu")
    store = ArtifactStore()
    store.set("raw.X", np.zeros((1, 28, 28, 3), dtype=np.uint8))

    with pytest.raises(PreprocessValidationError, match="3x32x32"):
        step.transform(store, rng=np.random.default_rng(0))


def test_aet_runtime_artifacts_from_cached_output() -> None:
    step = AetStep()
    produced = {
        "features.aet": np.zeros((2, 3), dtype=np.float32),
        "features.aet.info": {
            "checkpoint": {"path": "/tmp/aet.pth", "sha256": "x"},
            "runtime": {"device": "cpu"},
        },
    }

    info = step.runtime_artifacts(produced=produced, split="train")["features.aet.info"]

    assert info["output"]["shape"] == [2, 3]
    assert info["runtime"]["split"] == "train"


def test_aet_runtime_artifacts_without_prior_fit(tmp_path) -> None:
    step = AetStep(checkpoint_path=str(tmp_path / "missing.pth"))
    info = step.runtime_artifacts(split="val")["features.aet.info"]
    assert info["checkpoint"]["exists"] is False
    assert info["runtime"]["split"] == "val"


def test_aet_precomputed(tmp_path) -> None:
    features_path = tmp_path / "cifar_aet.npz"
    labels_path = tmp_path / "cifar_labels.npz"
    features = np.arange(7 * 4, dtype=np.float64).reshape(7, 4)
    labels = np.array([1, 2, 3, 4, 8, 9, 0], dtype=np.int64)
    np.savez_compressed(features_path, data=features)
    np.savez_compressed(labels_path, labels=labels)

    step = AetStep(
        source="precomputed",
        features_path=str(features_path),
        labels_path=str(labels_path),
        train_offset=0,
        test_offset=4,
        expected_rows=7,
        unit_normalize=False,
    )
    store = ArtifactStore()
    store.set("raw.X", np.zeros((2, 32, 32, 3), dtype=np.uint8))
    store.set("raw.y", np.array([8, 9], dtype=np.int64))

    result = step.transform(store, rng=np.random.default_rng(0))

    np.testing.assert_array_equal(result["features.aet"], features[4:6].astype(np.float32))
    info = result["features.aet.info"]
    assert info["kind"] == "precomputed_aet"
    assert info["alignment"]["row_offset"] == 4
    assert info["training"]["uses_labels"] is False
    assert (tmp_path / "cifar_aet.npy").exists()

    refreshed = step.runtime_artifacts(
        produced={"features.aet": result["features.aet"]}, split="test"
    )["features.aet.info"]
    assert refreshed["runtime"]["split"] == "test"


def test_aet_precomputed_requires_ordered_labels(tmp_path) -> None:
    features_path = tmp_path / "cifar_aet.npz"
    labels_path = tmp_path / "cifar_labels.npz"
    np.savez_compressed(features_path, data=np.zeros((5, 2), dtype=np.float64))
    np.savez_compressed(labels_path, labels=np.array([1, 2, 3, 4, 5], dtype=np.int64))

    step = AetStep(
        source="precomputed",
        features_path=str(features_path),
        labels_path=str(labels_path),
        train_offset=0,
        test_offset=3,
        expected_rows=5,
    )
    store = ArtifactStore()
    store.set("raw.X", np.zeros((2, 32, 32, 3), dtype=np.uint8))
    store.set("raw.y", np.array([2, 4], dtype=np.int64))

    with pytest.raises(PreprocessValidationError, match="Could not align raw.y"):
        step.transform(store, rng=np.random.default_rng(0))


def test_aet_precomputed_error_paths_and_empty_unit_norm(tmp_path) -> None:
    store = ArtifactStore()
    store.set("raw.y", np.array([], dtype=np.int64))

    missing_features = AetStep(
        source="precomputed",
        features_path=str(tmp_path / "missing_features.npz"),
        labels_path=str(tmp_path / "missing_labels.npz"),
        expected_rows=0,
    )
    with pytest.raises(PreprocessValidationError, match="expected_rows must be > 0"):
        missing_features.transform(store, rng=np.random.default_rng(0))

    missing_features.expected_rows = 2
    with pytest.raises(PreprocessValidationError, match="features not found"):
        missing_features.transform(store, rng=np.random.default_rng(0))

    features_path = tmp_path / "cifar_aet.npz"
    labels_path = tmp_path / "cifar_labels.npz"
    np.savez_compressed(features_path, data=np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(PreprocessValidationError, match="labels not found"):
        AetStep(
            source="precomputed",
            features_path=str(features_path),
            labels_path=str(labels_path),
            expected_rows=2,
        ).transform(store, rng=np.random.default_rng(0))

    np.savez_compressed(labels_path, other=np.array([0, 1], dtype=np.int64))
    with pytest.raises(PreprocessValidationError, match="Expected member"):
        AetStep(
            source="precomputed",
            features_path=str(features_path),
            labels_path=str(labels_path),
            expected_rows=2,
        ).transform(store, rng=np.random.default_rng(0))

    np.savez_compressed(labels_path, labels=np.array([0], dtype=np.int64))
    with pytest.raises(PreprocessValidationError, match="Expected 2 precomputed CIFAR labels"):
        AetStep(
            source="precomputed",
            features_path=str(features_path),
            labels_path=str(labels_path),
            expected_rows=2,
        ).transform(store, rng=np.random.default_rng(0))

    wrong_features = tmp_path / "wrong_features.npz"
    wrong_npy = tmp_path / "wrong_features.npy"
    np.savez_compressed(wrong_features, data=np.zeros((1, 3), dtype=np.float32))
    np.save(wrong_npy, np.zeros((1, 3), dtype=np.float32))
    np.savez_compressed(labels_path, labels=np.array([0, 1], dtype=np.int64))
    with pytest.raises(PreprocessValidationError, match="Expected 2 AET rows"):
        AetStep(
            source="precomputed",
            features_path=str(wrong_features),
            labels_path=str(labels_path),
            extracted_npy_path=str(wrong_npy),
            expected_rows=2,
        ).transform(store, rng=np.random.default_rng(0))

    zero_features = tmp_path / "zero_features.npz"
    zero_npy = tmp_path / "zero_features.npy"
    np.savez_compressed(zero_features, data=np.zeros((2, 3), dtype=np.float32))
    np.save(zero_npy, np.zeros((2, 3), dtype=np.float32))
    result = AetStep(
        source="precomputed",
        features_path=str(zero_features),
        labels_path=str(labels_path),
        extracted_npy_path=str(zero_npy),
        expected_rows=2,
        unit_normalize=True,
    ).transform(store, rng=np.random.default_rng(0))
    assert result["features.aet"].shape == (0, 3)

    noncontig_features = tmp_path / "noncontig_features.npz"
    noncontig_npy = tmp_path / "noncontig_features.npy"
    np.savez_compressed(noncontig_features, data=np.arange(12, dtype=np.float32).reshape(4, 3))
    np.save(noncontig_npy, np.arange(12, dtype=np.float32).reshape(4, 3))
    np.savez_compressed(labels_path, labels=np.array([0, 1, 2, 3], dtype=np.int64))
    noncontig = AetStep(
        source="precomputed",
        features_path=str(noncontig_features),
        labels_path=str(labels_path),
        extracted_npy_path=str(noncontig_npy),
        expected_rows=4,
        unit_normalize=False,
    )
    noncontig._resolve_precomputed_indices = lambda labels, raw_y: (
        np.array([1, 3], dtype=np.int64),
        1,
    )
    store_noncontig = ArtifactStore()
    store_noncontig.set("raw.y", np.array([9, 9], dtype=np.int64))
    result = noncontig.transform(store_noncontig, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(
        result["features.aet"],
        np.array([[3.0, 4.0, 5.0], [9.0, 10.0, 11.0]], dtype=np.float32),
    )


def test_aet_prepare_images_and_empty_encode_paths() -> None:
    step = AetStep(input_scaling="none", feature_layer="classifier")
    with pytest.raises(PreprocessValidationError, match="image-like"):
        step._prepare_images(np.array([1.0, 2.0], dtype=np.float32))

    gray = np.ones((1, 1, 32, 32), dtype=np.float32)
    prepared = step._prepare_images(gray)
    assert prepared.shape == (1, 3, 32, 32)

    empty = step._prepare_images(np.zeros((0, 32, 32, 3), dtype=np.float32))
    assert empty.shape == (0, 3, 32, 32)

    class FakeTorch:
        @staticmethod
        def no_grad():
            raise AssertionError("empty encode must return before model execution")

    step.model_ = object()
    step.torch_ = FakeTorch()
    step.device_ = "cpu"
    encoded = step._encode(np.zeros((0, 32, 32, 3), dtype=np.float32))
    assert encoded.shape == (0, 192)


def test_aet_regressor_output_key_validation_and_default_forward() -> None:
    torch = pytest.importorskip("torch")
    model = make_aet_regressor(torch)
    with pytest.raises(ValueError, match="Empty list"):
        model.nin(torch.zeros((1, 3, 32, 32)), out_feat_keys=[])
    with pytest.raises(ValueError, match="does not exist"):
        model.nin(torch.zeros((1, 3, 32, 32)), out_feat_keys=["bad"])
    with pytest.raises(ValueError, match="Duplicate"):
        model.nin(torch.zeros((1, 3, 32, 32)), out_feat_keys=["conv1", "conv1"])

    f1, f2, pred = model(torch.zeros((1, 3, 32, 32)), torch.zeros((1, 3, 32, 32)))
    assert f1.shape == (1, 192)
    assert f2.shape == (1, 192)
    assert pred.shape == (1, 8)

    f1, f2 = model(
        torch.zeros((1, 3, 32, 32)),
        torch.zeros((1, 3, 32, 32)),
        out_feat_keys=["classifier"],
    )
    assert f1.shape == (1, 192)
    assert f2.shape == (1, 192)


def test_aet_encode_checkpoint_without_unit_normalization(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "net_epoch_1499.pth"
    torch.save(make_aet_regressor(torch).state_dict(), checkpoint)
    step = AetStep(
        checkpoint_path=str(checkpoint),
        batch_size=1,
        device="cpu",
        unit_normalize=False,
        feature_layer="classifier",
    )
    store = ArtifactStore()
    store.set("raw.X", np.ones((1, 32, 32, 3), dtype=np.float32))
    result = step.transform(store, rng=np.random.default_rng(0))
    assert result["features.aet"].shape == (1, 192)
