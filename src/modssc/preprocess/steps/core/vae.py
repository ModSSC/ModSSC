from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from modssc.preprocess.cache import (
    _exclusive_cache_lock,
    _fsync_directory,
    _fsync_file,
    default_cache_dir,
)
from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.fingerprint import fingerprint, stable_json_dumps
from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.optional import require as require_optional
from modssc.preprocess.steps.base import fit_subset
from modssc.preprocess.store import ArtifactStore
from modssc.runtime.device import resolve_device_name
from modssc.utils.io import atomic_write_text

logger = logging.getLogger(__name__)

VAE_INFO_SCHEMA_VERSION = 3
VAE_OUTPUT_IDENTITY_SCHEMA_VERSION = 1
VAE_MODEL_CACHE_SCHEMA_VERSION = 1
VAE_MODEL_CACHE_KIND = "modssc.preprocess.vae-model"
VAE_MODEL_CACHE_POINTER = "CURRENT.json"
VAE_MODEL_CACHE_MANIFEST = "manifest.json"


def _as_dense_2d(X: Any, *, name: str) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    arr = np.asarray(to_numpy(X), dtype=np.float32)
    if arr.ndim != 2:
        raise PreprocessValidationError(f"VaeStep expects a 2D {name} matrix")
    return arr


def _clean_array(X: np.ndarray, *, impute: np.ndarray) -> np.ndarray:
    arr = np.array(X, dtype=np.float32, copy=True)
    finite = np.isfinite(arr)
    if not finite.all():
        np.copyto(arr, impute, where=~finite)
    return arr


def _subset_rows(X: Any, *, fit_indices: np.ndarray) -> Any:
    idx = np.asarray(fit_indices, dtype=np.int64)
    if idx.ndim != 1:
        raise PreprocessValidationError("fit_indices must be 1D")
    if hasattr(X, "__getitem__"):
        try:
            return X[idx]
        except Exception:
            pass
    return fit_subset(X, fit_indices=idx)


def _array_sha256(X: np.ndarray) -> str:
    arr = np.ascontiguousarray(X)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(tuple(int(dim) for dim in arr.shape)).encode("utf-8"))
    h.update(memoryview(arr).cast("B"))
    return h.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _signed_cache_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {str(key): value for key, value in payload.items() if key != "manifest_sha256"}
    digest = hashlib.sha256(stable_json_dumps(unsigned).encode("utf-8")).hexdigest()
    return {**unsigned, "manifest_sha256": digest}


def _verify_cache_manifest(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if payload.get("schema_version") != VAE_MODEL_CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported {label} schema version")
    if payload.get("cache_kind") != VAE_MODEL_CACHE_KIND:
        raise ValueError(f"invalid {label} cache kind")
    expected = payload.get("manifest_sha256")
    if not _is_sha256(expected):
        raise ValueError(f"invalid {label} manifest digest")
    unsigned = {str(key): value for key, value in payload.items() if key != "manifest_sha256"}
    observed = hashlib.sha256(stable_json_dumps(unsigned).encode("utf-8")).hexdigest()
    if observed != expected:
        raise ValueError(f"{label} manifest digest differs")
    return dict(payload)


def _read_json_bytes(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or is not a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a mapping")
    return raw, value


def _verified_file(path: Path, record: Mapping[str, Any], *, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"cached VAE {label} is missing or is not a regular file")
    expected_sha256 = record.get("sha256")
    expected_size = record.get("size_bytes")
    if not _is_sha256(expected_sha256):
        raise ValueError(f"cached VAE {label} has invalid SHA-256 metadata")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
        raise ValueError(f"cached VAE {label} has invalid size metadata")
    before = path.stat()
    if int(before.st_size) != expected_size or _file_sha256(path) != expected_sha256:
        raise ValueError(f"cached VAE file hash mismatch: {label}")
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"cached VAE {label} changed while hashing")


@lru_cache(maxsize=1)
def _vae_implementation_sha256() -> str:
    source = Path(__file__).resolve()
    return _file_sha256(source)


def _default_vae_cache_dir() -> Path:
    return default_cache_dir() / "vae_models"


def _runtime_dependencies(torch: Any, *, device: str) -> dict[str, str]:
    """Return the small runtime surface that can affect learned VAE features."""

    return {
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "device": str(device),
    }


def _vae_cache_identity(torch: Any, *, device: str) -> dict[str, Any]:
    return {
        "schema_version": VAE_INFO_SCHEMA_VERSION,
        "implementation_sha256": _vae_implementation_sha256(),
        "runtime": _runtime_dependencies(torch, device=device),
    }


VAE_PRESETS: dict[str, dict[str, Any]] = {
    "graphlearning_mnist_vae2": {
        "latent_dim": 20,
        "hidden_dims": (400,),
        "epochs": 100,
        "batch_size": 128,
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "beta": 1.0,
        "dropout": 0.0,
        "input_scaling": "global_minmax",
        "reconstruction_loss": "bce",
        "decoder_output": "sigmoid",
        "expected_input_dim": 784,
    },
    "poisson_fashionmnist": {
        "latent_dim": 30,
        "hidden_dims": (400,),
        "epochs": 100,
        "batch_size": 128,
        "lr": 1.0e-3,
        "beta": 1.0,
        "dropout": 0.0,
        "input_scaling": "global_minmax",
        "reconstruction_loss": "bce",
        "decoder_output": "sigmoid",
        "expected_input_dim": 784,
    },
}


def _safe_cache_component(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    name = "".join(cleaned).strip("._")
    return name or "vae"


@dataclass
class VaeStep:
    """Train a small dense VAE and expose latent means as ``features.vae``.

    The step intentionally leaves ``features.X`` unchanged. Downstream graph
    construction can opt into the VAE representation with
    ``graph.spec.feature_field: features.vae`` while the non-VAE branch keeps
    using ``features.X``.
    """

    latent_dim: int = 32
    hidden_dims: tuple[int, ...] | list[int] | None = None
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    beta: float = 1.0
    dropout: float = 0.0
    input_scaling: str | None = None
    reconstruction_loss: str = "mse"
    decoder_output: str = "linear"
    standardize: bool = True
    device: str | None = "auto"
    max_fit_samples: int | None = None
    preset: str | None = None
    expected_input_dim: int | None = None
    model_cache: bool = True
    model_cache_dir: str | None = None
    cache_key: str | None = None
    model_seed: int | None = None
    fit_scope: str = "selected"
    require_cache_hit: bool = field(default=False, kw_only=True)
    expected_model_fingerprint: str | None = field(default=None, kw_only=True)

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    impute_: np.ndarray | None = None
    model_: Any = field(default=None, init=False, repr=False)
    torch_: Any = field(default=None, init=False, repr=False)
    device_: str | None = field(default=None, init=False, repr=False)
    model_info_: dict[str, Any] | None = field(default=None, init=False, repr=False)
    model_cache_hit_: bool = field(default=False, init=False, repr=False)
    model_cache_dir_: Path | None = field(default=None, init=False, repr=False)
    model_fingerprint_: str | None = field(default=None, init=False, repr=False)
    model_training_runtime_: dict[str, str] | None = field(default=None, init=False, repr=False)

    def _apply_preset(self) -> None:
        if self.preset is None:
            return
        key = str(self.preset).strip().lower().replace("-", "_")
        aliases = {
            # Backward-compatible name used by the original Poisson cards.  It
            # resolves to the method-independent GraphLearning VAE2 recipe so
            # every graph method shares the same canonical cache identity.
            "poisson_mnist": "graphlearning_mnist_vae2",
            "poisson_fashion_mnist": "poisson_fashionmnist",
            "poisson_fmnist": "poisson_fashionmnist",
        }
        key = aliases.get(key, key)
        if key not in VAE_PRESETS:
            known = ", ".join(sorted(VAE_PRESETS))
            raise PreprocessValidationError(
                f"Unknown VAE preset {self.preset!r}; known presets: {known}"
            )
        self.preset = key
        for name, value in VAE_PRESETS[key].items():
            setattr(self, name, value)

    def _validate_params(self) -> None:
        self._apply_preset()
        if int(self.latent_dim) <= 0:
            raise PreprocessValidationError("latent_dim must be > 0")
        if int(self.epochs) <= 0:
            raise PreprocessValidationError("epochs must be > 0")
        if int(self.batch_size) <= 0:
            raise PreprocessValidationError("batch_size must be > 0")
        if float(self.lr) <= 0:
            raise PreprocessValidationError("lr must be > 0")
        if float(self.beta) < 0:
            raise PreprocessValidationError("beta must be >= 0")
        if not (0.0 <= float(self.dropout) < 1.0):
            raise PreprocessValidationError("dropout must be in [0, 1)")
        if self.max_fit_samples is not None and int(self.max_fit_samples) <= 0:
            raise PreprocessValidationError("max_fit_samples must be > 0 when provided")
        if self.expected_input_dim is not None and int(self.expected_input_dim) <= 0:
            raise PreprocessValidationError("expected_input_dim must be > 0 when provided")
        hidden_dims = self._hidden_dims()
        if not hidden_dims or any(int(dim) <= 0 for dim in hidden_dims):
            raise PreprocessValidationError("hidden_dims must contain positive integers")
        if self._input_scaling() not in {"none", "standardize", "minmax", "global_minmax"}:
            raise PreprocessValidationError(
                "input_scaling must be one of none, standardize, minmax, global_minmax"
            )
        if self.reconstruction_loss not in {"mse", "bce"}:
            raise PreprocessValidationError("reconstruction_loss must be mse or bce")
        if self.decoder_output not in {"linear", "sigmoid"}:
            raise PreprocessValidationError("decoder_output must be linear or sigmoid")
        if self.reconstruction_loss == "bce" and self.decoder_output != "sigmoid":
            raise PreprocessValidationError(
                "reconstruction_loss=bce requires decoder_output=sigmoid"
            )
        if self.model_seed is not None and int(self.model_seed) < 0:
            raise PreprocessValidationError("model_seed must be >= 0 when provided")
        if bool(self.require_cache_hit) and not bool(self.model_cache):
            raise PreprocessValidationError("require_cache_hit=true requires model_cache=true")
        if (
            self.expected_model_fingerprint is not None
            and not str(self.expected_model_fingerprint).strip()
        ):
            raise PreprocessValidationError("expected_model_fingerprint must not be empty")
        if str(self.fit_scope) not in {"selected", "all"}:
            raise PreprocessValidationError("fit_scope must be one of: selected, all")

    def _hidden_dims(self) -> tuple[int, ...]:
        if self.hidden_dims is None:
            return (256, 128)
        return tuple(int(dim) for dim in self.hidden_dims)

    def _input_scaling(self) -> str:
        if self.input_scaling is not None:
            return str(self.input_scaling)
        return "standardize" if self.standardize else "none"

    def _params_for_fingerprint(self) -> dict[str, Any]:
        return {
            "latent_dim": int(self.latent_dim),
            "hidden_dims": [int(dim) for dim in self._hidden_dims()],
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "beta": float(self.beta),
            "dropout": float(self.dropout),
            "input_scaling": self._input_scaling(),
            "reconstruction_loss": str(self.reconstruction_loss),
            "decoder_output": str(self.decoder_output),
            "max_fit_samples": None if self.max_fit_samples is None else int(self.max_fit_samples),
            "preset": None if self.preset is None else str(self.preset),
            "expected_input_dim": None
            if self.expected_input_dim is None
            else int(self.expected_input_dim),
            "model_seed": None if self.model_seed is None else int(self.model_seed),
            "fit_scope": str(self.fit_scope),
        }

    def _cache_root(self) -> Path:
        if self.model_cache_dir:
            return Path(self.model_cache_dir).expanduser().resolve()
        return _default_vae_cache_dir()

    def _cache_dir_for(self, model_fingerprint: str) -> Path:
        root = self._cache_root()
        suffix = model_fingerprint.split(":", 1)[-1][:16]
        if self.cache_key:
            return root / f"{_safe_cache_component(str(self.cache_key))}-{suffix}"
        return root / _safe_cache_component(model_fingerprint)

    def _model_fingerprint(
        self,
        *,
        fit_shape: tuple[int, int],
        fit_data_hash: str,
        fit_indices_hash: str,
        seed: int,
        cache_identity: Mapping[str, Any] | None = None,
    ) -> str:
        return fingerprint(
            {
                "kind": "dense_vae",
                "version": 2,
                "fit_shape": [int(fit_shape[0]), int(fit_shape[1])],
                "fit_data_sha256": fit_data_hash,
                "fit_indices_sha256": fit_indices_hash,
                "seed": int(seed),
                "params": self._params_for_fingerprint(),
                "cache_identity": (
                    dict(cache_identity)
                    if cache_identity is not None
                    else {
                        "schema_version": VAE_INFO_SCHEMA_VERSION,
                        "implementation_sha256": _vae_implementation_sha256(),
                    }
                ),
            },
            prefix="vae_",
        )

    def _build_info(
        self,
        *,
        model_fingerprint: str,
        fit_shape: tuple[int, int],
        fit_data_hash: str,
        fit_indices_hash: str,
        seed: int,
        device: str,
        torch: Any,
        cache_dir: Path | None,
        cache_hit: bool,
    ) -> dict[str, Any]:
        runtime = _runtime_dependencies(torch, device=device)
        training_runtime = dict(self.model_training_runtime_ or runtime)
        return {
            "kind": "dense_vae",
            "schema_version": VAE_INFO_SCHEMA_VERSION,
            "fingerprint": model_fingerprint,
            "expected_fingerprint": self.expected_model_fingerprint,
            "params": self._params_for_fingerprint(),
            "training": {
                "n_fit_samples": int(fit_shape[0]),
                "input_dim": int(fit_shape[1]),
                "fit_data_sha256": fit_data_hash,
                "fit_indices_sha256": fit_indices_hash,
                "fit_scope": str(self.fit_scope),
                "model_seed": int(seed),
                "uses_labels": False,
                "device": str(device),
            },
            "cache": {
                "enabled": bool(self.model_cache),
                "hit": bool(cache_hit),
                "dir": None if cache_dir is None else str(cache_dir),
                "cache_key": None if self.cache_key is None else str(self.cache_key),
            },
            "training_runtime": training_runtime,
            "runtime": runtime,
        }

    def _runtime_info(self, produced: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.model_info_ is None:
            raise PreprocessValidationError("VaeStep runtime metadata requested before fit()")
        info = dict(self.model_info_)
        features = produced.get("features.vae") if produced is not None else None
        feature_array = None if features is None else np.asarray(to_numpy(features))
        shape = None if feature_array is None else feature_array.shape
        dtype = None if feature_array is None else feature_array.dtype
        info["output"] = {
            "artifact": "features.vae",
            "shape": None if shape is None else [int(dim) for dim in shape],
            "dtype": None if dtype is None else str(dtype),
            "content_sha256": (None if feature_array is None else _array_sha256(feature_array)),
        }
        return info

    def runtime_artifacts(
        self, *, produced: dict[str, Any] | None = None, split: str = ""
    ) -> dict[str, Any]:
        info = self._runtime_info(produced=produced)
        runtime = dict(info.get("runtime", {}))
        runtime["split"] = str(split)
        info["runtime"] = runtime
        output = dict(info["output"])
        content_sha256 = output.get("content_sha256")
        output["identity_schema_version"] = VAE_OUTPUT_IDENTITY_SCHEMA_VERSION
        output["identity_fingerprint"] = (
            None
            if content_sha256 is None
            else fingerprint(
                {
                    "kind": "dense_vae_latent_output",
                    "version": VAE_OUTPUT_IDENTITY_SCHEMA_VERSION,
                    "model_fingerprint": info["fingerprint"],
                    "content_sha256": content_sha256,
                    "shape": output["shape"],
                    "dtype": output["dtype"],
                    "training_runtime": info.get("training_runtime"),
                    "runtime": runtime,
                },
                prefix="vae_output_",
            )
        )
        info["output"] = output
        return {"features.vae.info": info}

    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise PreprocessValidationError("VaeStep scaling state is missing")
        Z = (X - self.mean_) / self.scale_
        if self.reconstruction_loss == "bce":
            Z = np.clip(Z, 0.0, 1.0)
        return Z.astype(np.float32, copy=False)

    def _make_model(self, torch: Any, input_dim: int) -> Any:
        nn = torch.nn
        hidden_dims = self._hidden_dims()
        dropout = float(self.dropout)
        decoder_output = str(self.decoder_output)

        class DenseVAE(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                encoder_layers: list[Any] = []
                prev = int(input_dim)
                for hidden in hidden_dims:
                    encoder_layers.append(nn.Linear(prev, int(hidden)))
                    encoder_layers.append(nn.ReLU())
                    if dropout > 0:
                        encoder_layers.append(nn.Dropout(dropout))
                    prev = int(hidden)
                self.encoder = nn.Sequential(*encoder_layers)
                self.mu = nn.Linear(prev, int(self_outer.latent_dim))
                self.logvar = nn.Linear(prev, int(self_outer.latent_dim))

                decoder_layers: list[Any] = []
                prev = int(self_outer.latent_dim)
                for hidden in reversed(hidden_dims):
                    decoder_layers.append(nn.Linear(prev, int(hidden)))
                    decoder_layers.append(nn.ReLU())
                    if dropout > 0:
                        decoder_layers.append(nn.Dropout(dropout))
                    prev = int(hidden)
                decoder_layers.append(nn.Linear(prev, int(input_dim)))
                if decoder_output == "sigmoid":
                    decoder_layers.append(nn.Sigmoid())
                self.decoder = nn.Sequential(*decoder_layers)

            def encode(self, x: Any) -> tuple[Any, Any]:
                h = self.encoder(x)
                return self.mu(h), self.logvar(h)

            def reparameterize(self, mu: Any, logvar: Any) -> Any:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z: Any) -> Any:
                return self.decoder(z)

            def forward(self, x: Any) -> tuple[Any, Any, Any]:
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar

        self_outer = self
        return DenseVAE()

    def _load_cached_model(
        self,
        torch: Any,
        *,
        cache_dir: Path,
        input_dim: int,
        device: str,
        expected_fingerprint: str | None = None,
    ) -> bool:
        pointer_path = cache_dir / VAE_MODEL_CACHE_POINTER
        if not pointer_path.is_file() or pointer_path.is_symlink():
            return False

        try:
            _pointer_raw, pointer_value = _read_json_bytes(pointer_path, label="cached VAE pointer")
            pointer = _verify_cache_manifest(pointer_value, label="cached VAE pointer")
            fingerprint_value = pointer.get("fingerprint")
            if not isinstance(fingerprint_value, str):
                raise ValueError("cached VAE pointer fingerprint is invalid")
            if expected_fingerprint is not None and fingerprint_value != str(expected_fingerprint):
                raise ValueError("cached VAE fingerprint does not match the requested model")

            generation_name = pointer.get("generation")
            generation_manifest_sha256 = pointer.get("generation_manifest_sha256")
            if (
                not isinstance(generation_name, str)
                or not generation_name
                or Path(generation_name).name != generation_name
            ):
                raise ValueError("cached VAE generation is invalid")
            if not _is_sha256(generation_manifest_sha256):
                raise ValueError("cached VAE generation manifest SHA-256 is invalid")
            generation_dir = cache_dir / "generations" / generation_name
            if not generation_dir.is_dir() or generation_dir.is_symlink():
                raise ValueError("cached VAE generation is missing")
            generation_path = generation_dir / VAE_MODEL_CACHE_MANIFEST
            generation_raw, generation_value = _read_json_bytes(
                generation_path, label="cached VAE generation manifest"
            )
            if hashlib.sha256(generation_raw).hexdigest() != generation_manifest_sha256:
                raise ValueError("cached VAE generation manifest file hash mismatch")
            generation = _verify_cache_manifest(generation_value, label="cached VAE generation")
            if generation.get("fingerprint") != fingerprint_value:
                raise ValueError("cached VAE generation fingerprint differs")
            expected_identity = _vae_cache_identity(torch, device=device)
            if generation.get("cache_identity") != expected_identity:
                raise ValueError("cached VAE implementation or runtime identity differs")

            files = generation.get("files")
            if not isinstance(files, dict) or set(files) != {"model.pt", "state.npz"}:
                raise ValueError("cached VAE file manifest is invalid")
            model_record = files["model.pt"]
            state_record = files["state.npz"]
            if not isinstance(model_record, dict) or not isinstance(state_record, dict):
                raise ValueError("cached VAE file records are invalid")
            model_path = generation_dir / "model.pt"
            state_path = generation_dir / "state.npz"
            _verified_file(model_path, model_record, label="model.pt")
            _verified_file(state_path, state_record, label="state.npz")

            with np.load(state_path, allow_pickle=False) as state_np:
                state_arrays = {
                    "mean": np.asarray(state_np["mean"]),
                    "scale": np.asarray(state_np["scale"]),
                    "impute": np.asarray(state_np["impute"]),
                }
            state_metadata = generation.get("state")
            if not isinstance(state_metadata, dict) or set(state_metadata) != set(state_arrays):
                raise ValueError("cached VAE normalization state manifest is invalid")
            for name, array in state_arrays.items():
                record = state_metadata[name]
                if not isinstance(record, dict):
                    raise ValueError("cached VAE normalization state record is invalid")
                if record.get("dtype") != str(array.dtype) or record.get("shape") != [
                    int(value) for value in array.shape
                ]:
                    raise ValueError(f"cached VAE normalization state metadata differs: {name}")
            mean = np.asarray(state_arrays["mean"], dtype=np.float32)
            scale = np.asarray(state_arrays["scale"], dtype=np.float32)
            impute = np.asarray(state_arrays["impute"], dtype=np.float32)
            vector_shape = (int(input_dim),)
            scaling_shape = () if self._input_scaling() == "global_minmax" else vector_shape
            if (
                mean.shape != scaling_shape
                or scale.shape != scaling_shape
                or impute.shape != vector_shape
            ):
                raise ValueError("cached VAE normalization state shape differs")
            model = self._make_model(torch, input_dim=int(input_dim)).to(device)
            try:
                payload = torch.load(model_path, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(model_path, map_location="cpu")
            state_dict = payload.get("state_dict") if isinstance(payload, dict) else payload
            model.load_state_dict(state_dict)
            model.eval()
            _verified_file(model_path, model_record, label="model.pt")
            _verified_file(state_path, state_record, label="state.npz")
            info = generation.get("info")
            if not isinstance(info, dict):
                raise ValueError("cached VAE info is invalid")
            raw_training_runtime = info.get("training_runtime", info.get("runtime"))
            self.mean_ = mean
            self.scale_ = scale
            self.impute_ = impute
            self.model_ = model
            self.torch_ = torch
            self.device_ = device
            self.model_training_runtime_ = (
                {str(key): str(value) for key, value in raw_training_runtime.items()}
                if isinstance(raw_training_runtime, dict)
                else None
            )
            return True
        except Exception as exc:
            logger.warning("Ignoring corrupt VAE cache at %s: %s", cache_dir, exc)
            return False

    def _save_cached_model(
        self,
        torch: Any,
        *,
        cache_dir: Path,
        info: dict[str, Any],
        lock_held: bool = False,
    ) -> None:
        if self.model_ is None or self.mean_ is None or self.scale_ is None or self.impute_ is None:
            return
        if not lock_held:
            with _exclusive_cache_lock(cache_dir / ".cache.lock"):
                self._save_cached_model(torch, cache_dir=cache_dir, info=info, lock_held=True)
            return

        cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint_value = info.get("fingerprint")
        if not isinstance(fingerprint_value, str) or not fingerprint_value:
            raise PreprocessValidationError("VAE cache info fingerprint is missing")
        if self._load_cached_model(
            torch,
            cache_dir=cache_dir,
            input_dim=int(np.asarray(self.impute_).size),
            device=str(self.device_ or "cpu"),
            expected_fingerprint=fingerprint_value,
        ):
            return

        generations = cache_dir / "generations"
        generations.mkdir(parents=True, exist_ok=True)
        staging = generations / f".staging-{os.getpid()}-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            model_path = staging / "model.pt"
            state_path = staging / "state.npz"
            torch.save({"state_dict": self.model_.state_dict()}, model_path)
            _fsync_file(model_path)
            with state_path.open("wb") as handle:
                np.savez(
                    handle,
                    mean=np.asarray(self.mean_, dtype=np.float32),
                    scale=np.asarray(self.scale_, dtype=np.float32),
                    impute=np.asarray(self.impute_, dtype=np.float32),
                )
                handle.flush()
                os.fsync(handle.fileno())
            files = {
                path.name: {
                    "sha256": _file_sha256(path),
                    "size_bytes": int(path.stat().st_size),
                }
                for path in (model_path, state_path)
            }
            generation = _signed_cache_manifest(
                {
                    "schema_version": VAE_MODEL_CACHE_SCHEMA_VERSION,
                    "cache_kind": VAE_MODEL_CACHE_KIND,
                    "fingerprint": fingerprint_value,
                    "cache_identity": _vae_cache_identity(torch, device=str(self.device_ or "cpu")),
                    "created_at": datetime.now(UTC).isoformat(),
                    "files": files,
                    "state": {
                        "mean": {
                            "dtype": str(np.asarray(self.mean_).dtype),
                            "shape": [int(value) for value in np.asarray(self.mean_).shape],
                        },
                        "scale": {
                            "dtype": str(np.asarray(self.scale_).dtype),
                            "shape": [int(value) for value in np.asarray(self.scale_).shape],
                        },
                        "impute": {
                            "dtype": str(np.asarray(self.impute_).dtype),
                            "shape": [int(value) for value in np.asarray(self.impute_).shape],
                        },
                    },
                    "info": dict(info),
                }
            )
            generation_path = staging / VAE_MODEL_CACHE_MANIFEST
            atomic_write_text(generation_path, stable_json_dumps(generation))
            generation_manifest_sha256 = _file_sha256(generation_path)
            _fsync_directory(staging)

            generation_name = f"model-{generation['manifest_sha256']}-{uuid.uuid4().hex}"
            generation_dir = generations / generation_name
            os.replace(staging, generation_dir)
            _fsync_directory(generations)
            pointer = _signed_cache_manifest(
                {
                    "schema_version": VAE_MODEL_CACHE_SCHEMA_VERSION,
                    "cache_kind": VAE_MODEL_CACHE_KIND,
                    "fingerprint": fingerprint_value,
                    "generation": generation_name,
                    "generation_manifest_sha256": generation_manifest_sha256,
                }
            )
            atomic_write_text(cache_dir / VAE_MODEL_CACHE_POINTER, stable_json_dumps(pointer))
            _fsync_directory(cache_dir)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def _fit_model_with_cache_lock(
        self,
        *,
        torch: Any,
        device: str,
        X_fit: np.ndarray,
        fit_data_hash: str,
        fit_indices_hash: str,
        seed: int,
        model_fingerprint: str,
        cache_dir: Path | None,
    ) -> None:
        if cache_dir is None:
            self._fit_model_locked(
                torch=torch,
                device=device,
                X_fit=X_fit,
                fit_data_hash=fit_data_hash,
                fit_indices_hash=fit_indices_hash,
                seed=seed,
                model_fingerprint=model_fingerprint,
                cache_dir=None,
            )
            return
        with _exclusive_cache_lock(cache_dir / ".cache.lock"):
            self._fit_model_locked(
                torch=torch,
                device=device,
                X_fit=X_fit,
                fit_data_hash=fit_data_hash,
                fit_indices_hash=fit_indices_hash,
                seed=seed,
                model_fingerprint=model_fingerprint,
                cache_dir=cache_dir,
            )

    def _fit_model_locked(
        self,
        *,
        torch: Any,
        device: str,
        X_fit: np.ndarray,
        fit_data_hash: str,
        fit_indices_hash: str,
        seed: int,
        model_fingerprint: str,
        cache_dir: Path | None,
    ) -> None:
        if cache_dir is not None and self._load_cached_model(
            torch,
            cache_dir=cache_dir,
            input_dim=int(X_fit.shape[1]),
            device=device,
            expected_fingerprint=model_fingerprint,
        ):
            self.model_cache_hit_ = True
            self.model_info_ = self._build_info(
                model_fingerprint=model_fingerprint,
                fit_shape=(int(X_fit.shape[0]), int(X_fit.shape[1])),
                fit_data_hash=fit_data_hash,
                fit_indices_hash=fit_indices_hash,
                seed=seed,
                device=device,
                torch=torch,
                cache_dir=cache_dir,
                cache_hit=True,
            )
            logger.info(
                "Loaded cached VAE model: fingerprint=%s dir=%s",
                model_fingerprint,
                cache_dir,
            )
            return

        if cache_dir is not None and bool(self.require_cache_hit):
            raise PreprocessValidationError(
                "The frozen VAE cache is missing, corrupt, or has a different fingerprint: "
                f"{cache_dir}. Build and verify it during preflight; scientific jobs are not "
                "allowed to train it implicitly."
            )

        self.model_cache_hit_ = False
        scaled_fit = self._scale_features(X_fit)
        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)

        model = self._make_model(torch, input_dim=int(scaled_fit.shape[1])).to(device)
        model.train()

        tensor = torch.as_tensor(scaled_fit, dtype=torch.float32)
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataset = torch.utils.data.TensorDataset(tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(self.batch_size),
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )

        for _epoch in range(int(self.epochs)):
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                recon, mu, logvar = model(batch)
                if self.reconstruction_loss == "bce":
                    recon_loss = torch.nn.functional.binary_cross_entropy(
                        recon, batch, reduction="sum"
                    )
                    kl_loss = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
                else:
                    recon_loss = torch.nn.functional.mse_loss(recon, batch, reduction="mean")
                    kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + float(self.beta) * kl_loss
                loss.backward()
                optimizer.step()

        model.eval()
        self.model_ = model
        self.torch_ = torch
        self.device_ = device
        self.model_training_runtime_ = _runtime_dependencies(torch, device=device)
        self.model_info_ = self._build_info(
            model_fingerprint=model_fingerprint,
            fit_shape=(int(X_fit.shape[0]), int(X_fit.shape[1])),
            fit_data_hash=fit_data_hash,
            fit_indices_hash=fit_indices_hash,
            seed=seed,
            device=device,
            torch=torch,
            cache_dir=cache_dir,
            cache_hit=False,
        )
        if cache_dir is not None:
            self._save_cached_model(
                torch,
                cache_dir=cache_dir,
                info=self.model_info_,
                lock_held=True,
            )
            logger.info(
                "Saved VAE model cache: fingerprint=%s dir=%s",
                model_fingerprint,
                cache_dir,
            )

    def fit(
        self, store: ArtifactStore, *, fit_indices: np.ndarray, rng: np.random.Generator
    ) -> None:
        self._validate_params()

        X_source = store.require("features.X")
        fit_indices_arr = np.asarray(fit_indices, dtype=np.int64).reshape(-1)
        if str(self.fit_scope) == "all":
            X_fit = _as_dense_2d(X_source, name="features.X")
            fit_indices_used = np.arange(int(X_fit.shape[0]), dtype=np.int64)
        else:
            fit_indices_used = fit_indices_arr
            X_fit = _as_dense_2d(_subset_rows(X_source, fit_indices=fit_indices), name="features.X")
        if int(X_fit.shape[0]) == 0:
            raise PreprocessValidationError("Cannot fit VAE on empty selection")
        if self.expected_input_dim is not None and int(X_fit.shape[1]) != int(
            self.expected_input_dim
        ):
            raise PreprocessValidationError(
                "VaeStep expected input_dim="
                f"{int(self.expected_input_dim)} for preset/config, got {int(X_fit.shape[1])}"
            )

        seed = (
            int(self.model_seed)
            if self.model_seed is not None
            else int(rng.integers(0, np.iinfo(np.int32).max))
        )
        fit_rng = np.random.default_rng(seed)
        if self.max_fit_samples is not None and int(X_fit.shape[0]) > int(self.max_fit_samples):
            idx = fit_rng.choice(int(X_fit.shape[0]), size=int(self.max_fit_samples), replace=False)
            idx = np.asarray(idx, dtype=np.int64)
            X_fit = X_fit[idx]
            fit_indices_used = fit_indices_used[idx]

        finite = np.isfinite(X_fit)
        if finite.all():
            impute = X_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
        else:
            impute = np.mean(X_fit, axis=0, where=finite, dtype=np.float64)
            impute = np.where(np.isfinite(impute), impute, 0.0).astype(np.float32)
        X_fit = _clean_array(X_fit, impute=impute)
        fit_data_hash = _array_sha256(X_fit)
        fit_indices_hash = _array_sha256(fit_indices_used.astype(np.int64, copy=False))

        scaling = self._input_scaling()
        if scaling == "standardize":
            mean = X_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
            scale = X_fit.std(axis=0, dtype=np.float64).astype(np.float32)
            scale = np.where(scale > 1.0e-6, scale, 1.0).astype(np.float32)
        elif scaling == "minmax":
            mean = X_fit.min(axis=0).astype(np.float32)
            scale = (X_fit.max(axis=0) - mean).astype(np.float32)
            scale = np.where(scale > 1.0e-6, scale, 1.0).astype(np.float32)
        elif scaling == "global_minmax":
            global_min = np.asarray(X_fit.min(), dtype=np.float32)
            global_scale = np.asarray(X_fit.max() - float(global_min), dtype=np.float32)
            if float(global_scale) <= 1.0e-6:
                global_scale = np.asarray(1.0, dtype=np.float32)
            mean = global_min
            scale = global_scale
        else:
            mean = np.zeros(int(X_fit.shape[1]), dtype=np.float32)
            scale = np.ones(int(X_fit.shape[1]), dtype=np.float32)

        self.mean_ = mean
        self.scale_ = scale
        self.impute_ = impute

        torch = require_optional(
            module="torch",
            extra="inductive-torch",
            purpose="core.vae preprocessing step",
        )
        device = resolve_device_name(self.device, torch=torch) or "cpu"
        cache_identity = _vae_cache_identity(torch, device=device)
        model_fingerprint = self._model_fingerprint(
            fit_shape=(int(X_fit.shape[0]), int(X_fit.shape[1])),
            fit_data_hash=fit_data_hash,
            fit_indices_hash=fit_indices_hash,
            seed=seed,
            cache_identity=cache_identity,
        )
        if self.expected_model_fingerprint is not None and model_fingerprint != str(
            self.expected_model_fingerprint
        ):
            raise PreprocessValidationError(
                "VAE model fingerprint differs from expected_model_fingerprint: "
                f"computed {model_fingerprint}, expected {self.expected_model_fingerprint}"
            )
        cache_dir = self._cache_dir_for(model_fingerprint) if self.model_cache else None
        self.model_fingerprint_ = model_fingerprint
        self.model_cache_dir_ = cache_dir
        self._fit_model_with_cache_lock(
            torch=torch,
            device=device,
            X_fit=X_fit,
            fit_data_hash=fit_data_hash,
            fit_indices_hash=fit_indices_hash,
            seed=seed,
            model_fingerprint=model_fingerprint,
            cache_dir=cache_dir,
        )

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        del rng
        if (
            self.model_ is None
            or self.mean_ is None
            or self.scale_ is None
            or self.impute_ is None
            or self.torch_ is None
            or self.device_ is None
            or self.model_info_ is None
        ):
            raise PreprocessValidationError("VaeStep.transform called before fit()")

        X = _as_dense_2d(store.require("features.X"), name="features.X")
        n_samples = int(X.shape[0])
        Z = np.empty((n_samples, int(self.latent_dim)), dtype=np.float32)

        torch = self.torch_
        with torch.no_grad():
            for start in range(0, n_samples, int(self.batch_size)):
                end = min(start + int(self.batch_size), n_samples)
                chunk = _clean_array(X[start:end], impute=self.impute_)
                chunk = self._scale_features(chunk)
                batch = torch.as_tensor(chunk, dtype=torch.float32, device=self.device_)
                mu, _logvar = self.model_.encode(batch)
                Z[start:end] = mu.detach().cpu().numpy().astype(np.float32, copy=False)

        produced = {"features.vae": Z}
        produced.update(self.runtime_artifacts(produced=produced))
        return produced
