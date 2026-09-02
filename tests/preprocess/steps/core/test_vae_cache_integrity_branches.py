from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import modssc.preprocess.steps.core.vae as vae_module
from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.steps.core.vae import VaeStep


class _DummyModel:
    def __init__(self) -> None:
        self.loaded: object = None
        self.evaluated = False

    def to(self, _device: str) -> _DummyModel:
        return self

    def load_state_dict(self, state: object) -> None:
        self.loaded = state

    def state_dict(self) -> dict[str, object]:
        return {}

    def eval(self) -> None:
        self.evaluated = True


class _DummyTorch:
    __version__ = "test"

    def __init__(self, *, fail_save: bool = False) -> None:
        self.fail_save = fail_save

    def load(self, _path: Path, **_kwargs: object) -> dict[str, object]:
        return {"state_dict": {}}

    def save(self, _payload: object, path: Path) -> None:
        if self.fail_save:
            raise RuntimeError("save failed")
        path.write_bytes(b"model")


def _file_record(path: Path) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _write_signed(path: Path, payload: dict[str, object]) -> bytes:
    signed = vae_module._signed_cache_manifest(payload)
    raw = vae_module.stable_json_dumps(signed).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _valid_cache_layout(root: Path, *, global_minmax: bool = False) -> tuple[Path, Path, Path]:
    generation_dir = root / "generations" / "generation-valid"
    generation_dir.mkdir(parents=True)
    model_path = generation_dir / "model.pt"
    state_path = generation_dir / "state.npz"
    model_path.write_bytes(b"model")
    mean = np.asarray(0.0, dtype=np.float32) if global_minmax else np.zeros(2, dtype=np.float32)
    scale = np.asarray(1.0, dtype=np.float32) if global_minmax else np.ones(2, dtype=np.float32)
    impute = np.zeros(2, dtype=np.float32)
    np.savez(state_path, mean=mean, scale=scale, impute=impute)
    generation_path = generation_dir / vae_module.VAE_MODEL_CACHE_MANIFEST
    generation_raw = _write_signed(
        generation_path,
        {
            "schema_version": vae_module.VAE_MODEL_CACHE_SCHEMA_VERSION,
            "cache_kind": vae_module.VAE_MODEL_CACHE_KIND,
            "fingerprint": "fingerprint",
            "cache_identity": {"runtime": "expected"},
            "files": {
                "model.pt": _file_record(model_path),
                "state.npz": _file_record(state_path),
            },
            "state": {
                name: {
                    "dtype": "float32",
                    "shape": [int(value) for value in array.shape],
                }
                for name, array in (("mean", mean), ("scale", scale), ("impute", impute))
            },
            "info": {"training_runtime": {"device": "cpu"}},
        },
    )
    pointer_path = root / vae_module.VAE_MODEL_CACHE_POINTER
    _write_signed(
        pointer_path,
        {
            "schema_version": vae_module.VAE_MODEL_CACHE_SCHEMA_VERSION,
            "cache_kind": vae_module.VAE_MODEL_CACHE_KIND,
            "fingerprint": "fingerprint",
            "generation": generation_dir.name,
            "generation_manifest_sha256": hashlib.sha256(generation_raw).hexdigest(),
        },
    )
    return pointer_path, generation_path, generation_dir


def _mutate_signed(path: Path, mutation: object) -> bytes:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    return _write_signed(path, payload)


def _mutate_generation(
    pointer_path: Path,
    generation_path: Path,
    mutation: object,
) -> None:
    raw = _mutate_signed(generation_path, mutation)
    _mutate_signed(
        pointer_path,
        lambda pointer: pointer.update(generation_manifest_sha256=hashlib.sha256(raw).hexdigest()),
    )


def test_vae_manifest_json_and_file_guards(tmp_path: Path, monkeypatch) -> None:
    valid = vae_module._signed_cache_manifest(
        {
            "schema_version": vae_module.VAE_MODEL_CACHE_SCHEMA_VERSION,
            "cache_kind": vae_module.VAE_MODEL_CACHE_KIND,
        }
    )
    cases = (
        ({**valid, "schema_version": -1}, "unsupported test schema"),
        (
            vae_module._signed_cache_manifest({**valid, "cache_kind": "other"}),
            "invalid test cache kind",
        ),
        ({**valid, "manifest_sha256": "bad"}, "invalid test manifest digest"),
        (
            {**valid, "fingerprint": "tampered"},
            "test manifest digest differs",
        ),
    )
    for payload, message in cases:
        with pytest.raises(ValueError, match=message):
            vae_module._verify_cache_manifest(payload, label="test")

    with pytest.raises(ValueError, match="is missing"):
        vae_module._read_json_bytes(tmp_path / "missing.json", label="test JSON")
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="invalid JSON"):
        vae_module._read_json_bytes(invalid_json, label="test JSON")
    sequence_json = tmp_path / "sequence.json"
    sequence_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        vae_module._read_json_bytes(sequence_json, label="test JSON")

    file_path = tmp_path / "value.bin"
    file_path.write_bytes(b"aa")
    record = _file_record(file_path)
    with pytest.raises(ValueError, match="is missing"):
        vae_module._verified_file(tmp_path / "missing.bin", record, label="missing.bin")
    with pytest.raises(ValueError, match="invalid SHA-256"):
        vae_module._verified_file(file_path, {**record, "sha256": "bad"}, label="value.bin")
    with pytest.raises(ValueError, match="invalid size"):
        vae_module._verified_file(file_path, {**record, "size_bytes": True}, label="value.bin")
    with pytest.raises(ValueError, match="file hash mismatch"):
        vae_module._verified_file(file_path, {**record, "size_bytes": 3}, label="value.bin")

    expected = "a" * 64

    def mutate_while_hashing(path: Path) -> str:
        path.write_bytes(b"bb")
        return expected

    monkeypatch.setattr(vae_module, "_file_sha256", mutate_while_hashing)
    with pytest.raises(ValueError, match="changed while hashing"):
        vae_module._verified_file(
            file_path,
            {"sha256": expected, "size_bytes": 2},
            label="value.bin",
        )


def test_vae_default_and_explicit_cache_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(vae_module, "default_cache_dir", lambda: tmp_path)
    assert vae_module._default_vae_cache_dir() == tmp_path / "vae_models"
    default = VaeStep()
    assert default._cache_root() == tmp_path / "vae_models"
    assert default._cache_dir_for("vae:abc") == tmp_path / "vae_models" / "vae_abc"

    explicit = VaeStep(model_cache_dir=str(tmp_path / "models"), cache_key="paper:key")
    path = explicit._cache_dir_for("vae:0123456789abcdef")
    assert path.name == "paper_key-0123456789abcdef"


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("pointer-fingerprint-type", "pointer fingerprint is invalid"),
        ("pointer-fingerprint-mismatch", "does not match the requested model"),
        ("generation-name", "generation is invalid"),
        ("generation-sha", "generation manifest SHA-256 is invalid"),
        ("generation-missing", "generation is missing"),
        ("generation-file-hash", "generation manifest file hash mismatch"),
        ("generation-fingerprint", "generation fingerprint differs"),
        ("cache-identity", "runtime identity differs"),
        ("files", "file manifest is invalid"),
        ("file-record", "file records are invalid"),
        ("state", "state manifest is invalid"),
        ("state-record", "state record is invalid"),
        ("state-metadata", "state metadata differs"),
        ("state-shape", "state shape differs"),
        ("info", "cached VAE info is invalid"),
    ],
)
def test_vae_cache_load_fails_closed_for_every_authenticated_layer(
    tmp_path: Path,
    monkeypatch,
    caplog,
    case: str,
    message: str,
) -> None:
    pointer_path, generation_path, generation_dir = _valid_cache_layout(tmp_path)
    input_dim = 2
    if case == "pointer-fingerprint-type":
        _mutate_signed(pointer_path, lambda value: value.update(fingerprint=None))
    elif case == "pointer-fingerprint-mismatch":
        _mutate_signed(pointer_path, lambda value: value.update(fingerprint="other"))
    elif case == "generation-name":
        _mutate_signed(pointer_path, lambda value: value.update(generation="../bad"))
    elif case == "generation-sha":
        _mutate_signed(
            pointer_path,
            lambda value: value.update(generation_manifest_sha256="bad"),
        )
    elif case == "generation-missing":
        shutil.rmtree(generation_dir)
    elif case == "generation-file-hash":
        generation_path.write_bytes(generation_path.read_bytes() + b" ")
    elif case == "generation-fingerprint":
        _mutate_generation(
            pointer_path,
            generation_path,
            lambda value: value.update(fingerprint="other"),
        )
    elif case == "cache-identity":
        _mutate_generation(
            pointer_path,
            generation_path,
            lambda value: value.update(cache_identity={"runtime": "other"}),
        )
    elif case == "files":
        _mutate_generation(pointer_path, generation_path, lambda value: value.update(files={}))
    elif case == "file-record":
        _mutate_generation(
            pointer_path,
            generation_path,
            lambda value: value["files"].update({"model.pt": "invalid"}),
        )
    elif case == "state":
        _mutate_generation(pointer_path, generation_path, lambda value: value.update(state=[]))
    elif case == "state-record":
        _mutate_generation(
            pointer_path,
            generation_path,
            lambda value: value["state"].update(mean="invalid"),
        )
    elif case == "state-metadata":
        _mutate_generation(
            pointer_path,
            generation_path,
            lambda value: value["state"]["mean"].update(dtype="float64"),
        )
    elif case == "state-shape":
        input_dim = 3
    elif case == "info":
        _mutate_generation(pointer_path, generation_path, lambda value: value.update(info=[]))
    else:
        raise AssertionError(case)

    torch = _DummyTorch()
    step = VaeStep()
    model = _DummyModel()
    monkeypatch.setattr(
        vae_module,
        "_vae_cache_identity",
        lambda _torch, *, device: {"runtime": "expected"},
    )
    monkeypatch.setattr(step, "_make_model", lambda *_args, **_kwargs: model)

    with caplog.at_level("WARNING"):
        loaded = step._load_cached_model(
            torch,
            cache_dir=tmp_path,
            input_dim=input_dim,
            device="cpu",
            expected_fingerprint="fingerprint",
        )
    assert loaded is False
    assert message in caplog.text


def test_vae_cache_load_accepts_authenticated_global_minmax_state(
    tmp_path: Path, monkeypatch
) -> None:
    _valid_cache_layout(tmp_path, global_minmax=True)
    torch = _DummyTorch()
    step = VaeStep(input_scaling="global_minmax")
    model = _DummyModel()
    monkeypatch.setattr(
        vae_module,
        "_vae_cache_identity",
        lambda _torch, *, device: {"runtime": "expected"},
    )
    monkeypatch.setattr(step, "_make_model", lambda *_args, **_kwargs: model)

    loaded = step._load_cached_model(
        torch,
        cache_dir=tmp_path,
        input_dim=2,
        device="cpu",
        expected_fingerprint="fingerprint",
    )

    assert loaded is True
    assert step.mean_ is not None and step.mean_.shape == ()
    assert step.scale_ is not None and step.scale_.shape == ()
    assert step.impute_ is not None and step.impute_.shape == (2,)


def test_vae_cache_save_guards_idempotence_and_staging_cleanup(tmp_path: Path, monkeypatch) -> None:
    torch = _DummyTorch()
    step = VaeStep()
    step.model_ = _DummyModel()
    step.mean_ = np.zeros(2, dtype=np.float32)
    step.scale_ = np.ones(2, dtype=np.float32)
    step.impute_ = np.zeros(2, dtype=np.float32)
    step.device_ = "cpu"

    with pytest.raises(PreprocessValidationError, match="fingerprint is missing"):
        step._save_cached_model(torch, cache_dir=tmp_path / "missing", info={}, lock_held=True)

    cached_dir = tmp_path / "cached"
    monkeypatch.setattr(step, "_load_cached_model", lambda *_args, **_kwargs: True)
    step._save_cached_model(
        torch,
        cache_dir=cached_dir,
        info={"fingerprint": "fingerprint"},
        lock_held=True,
    )
    assert not (cached_dir / "generations").exists()

    failing_dir = tmp_path / "failing"
    monkeypatch.setattr(step, "_load_cached_model", lambda *_args, **_kwargs: False)
    with pytest.raises(RuntimeError, match="save failed"):
        step._save_cached_model(
            _DummyTorch(fail_save=True),
            cache_dir=failing_dir,
            info={"fingerprint": "fingerprint"},
            lock_held=True,
        )
    assert not list((failing_dir / "generations").glob(".staging-*"))
