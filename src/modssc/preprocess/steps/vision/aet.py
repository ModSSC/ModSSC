from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from modssc.preprocess.cache import default_cache_dir
from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.optional import require as require_optional
from modssc.preprocess.steps.vision.layout import prepare_image_array
from modssc.preprocess.store import ArtifactStore
from modssc.runtime.device import resolve_device_name

AET_PRESETS: dict[str, dict[str, Any]] = {
    "poisson_cifar10_projective": {
        "variant": "projective",
        "checkpoint_name": "net_epoch_1499.pth",
        "feature_layer": "conv2",
        "input_scaling": "cifar10",
        "unit_normalize": True,
        "expected_shape": (3, 32, 32),
    },
}

_VALID_FEATURE_LAYERS = {"conv1", "conv2", "conv3", "conv4", "classifier"}
_PRECOMPUTED_CIFAR_ROWS = 60_000
_SHA256_HEX = frozenset("0123456789abcdef")


def _safe_cache_component(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    name = "".join(cleaned).strip("._")
    return name or "aet"


def _default_aet_cache_dir() -> Path:
    return default_cache_dir() / "aet_models"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_npz_member(path: Path, member_name: str) -> Any:
    with np.load(path, allow_pickle=False) as npz:
        if member_name not in npz:
            raise PreprocessValidationError(
                f"Expected member {member_name!r} in {path}, got {sorted(npz.files)!r}"
            )
        return npz[member_name]


def _extract_single_npy_from_npz(npz_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(npz_path) as zf:
            members = [name for name in zf.namelist() if name.endswith(".npy")]
            if len(members) != 1:
                raise PreprocessValidationError(
                    f"Expected exactly one .npy member in {npz_path}, got {members!r}"
                )
            with zf.open(members[0]) as src, tmp_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():  # pragma: no cover - best-effort cleanup after failed extraction.
            tmp_path.unlink()


def _float32_npy_cache_path(npy_path: Path) -> Path:
    return npy_path.with_name(f"{npy_path.stem}.float32.npy")


def _float32_npy_cache_meta_path(npy_path: Path) -> Path:
    return npy_path.with_suffix(npy_path.suffix + ".json")


def _npy_source_identity(
    source_path: Path, *, source_npz_sha256: str | None = None
) -> dict[str, Any]:
    return {
        "npy_sha256": _file_sha256(source_path),
        "source_npz_sha256": source_npz_sha256,
    }


def _read_npy_cache_meta(meta_path: Path) -> dict[str, Any] | None:
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _npy_cache_matches(
    *,
    output_path: Path,
    meta_path: Path,
    kind: str,
    source_identity: dict[str, Any],
) -> bool:
    if not output_path.is_file():
        return False
    payload = _read_npy_cache_meta(meta_path)
    if payload is None:
        return False
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != kind
        or payload.get("source") != source_identity
    ):
        return False
    expected_output_sha256 = payload.get("output_npy_sha256")
    return (
        isinstance(expected_output_sha256, str)
        and _file_sha256(output_path) == expected_output_sha256
    )


def _write_npy_cache_meta(
    meta_path: Path,
    *,
    kind: str,
    source_identity: dict[str, Any],
    output_path: Path,
) -> None:
    tmp_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": kind,
                "source": source_identity,
                "output_npy_sha256": _file_sha256(output_path),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    tmp_path.replace(meta_path)


def _ensure_extracted_npy(
    npz_path: Path,
    output_path: Path,
    *,
    source_npz_sha256: str,
) -> None:
    meta_path = _float32_npy_cache_meta_path(output_path)
    source_identity = {"source_npz_sha256": source_npz_sha256}
    if _npy_cache_matches(
        output_path=output_path,
        meta_path=meta_path,
        kind="npz-extraction",
        source_identity=source_identity,
    ):
        return
    _extract_single_npy_from_npz(npz_path, output_path)
    _write_npy_cache_meta(
        meta_path,
        kind="npz-extraction",
        source_identity=source_identity,
        output_path=output_path,
    )


def _ensure_float32_npy(
    source_path: Path,
    output_path: Path,
    *,
    source_npz_sha256: str | None = None,
) -> None:
    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if source.dtype == np.float32:
        return
    source_identity = _npy_source_identity(
        source_path,
        source_npz_sha256=source_npz_sha256,
    )
    meta_path = _float32_npy_cache_meta_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        try:
            cached = np.load(output_path, mmap_mode="r", allow_pickle=False)
            cache_valid = (
                tuple(cached.shape) == tuple(source.shape)
                and cached.dtype == np.float32
                and _npy_cache_matches(
                    output_path=output_path,
                    meta_path=meta_path,
                    kind="float32-conversion",
                    source_identity=source_identity,
                )
            )
        except (OSError, ValueError):
            cache_valid = False
        if cache_valid:
            return
        output_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    target = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.float32,
        shape=tuple(source.shape),
    )
    try:
        for start in range(0, int(source.shape[0]), 512):
            end = min(start + 512, int(source.shape[0]))
            target[start:end] = np.asarray(source[start:end], dtype=np.float32)
        target.flush()
        del target
        tmp_path.replace(output_path)
        _write_npy_cache_meta(
            meta_path,
            kind="float32-conversion",
            source_identity=source_identity,
            output_path=output_path,
        )
    finally:
        if tmp_path.exists():  # pragma: no cover - best-effort cleanup after failed conversion.
            tmp_path.unlink()


def _default_precomputed_features_path() -> Path:
    return default_cache_dir() / "pretrained_features" / "aet" / "cifar_aet.npz"


def _default_precomputed_labels_path() -> Path:
    return default_cache_dir() / "pretrained_features" / "aet" / "cifar_labels.npz"


def _strip_state_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if all(str(k).startswith(prefix) for k in state_dict):
        return {str(k)[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _feature_dim(layer: str) -> int:
    if layer == "conv1":
        return 96 * 16 * 16
    if layer in {"conv2", "conv3", "conv4"}:
        return 192 * 8 * 8
    if layer == "classifier":
        return 192
    raise PreprocessValidationError(f"Unknown AET feature layer: {layer!r}")


def make_aet_regressor(torch: Any) -> Any:
    """Build the CIFAR AET Network-In-Network regressor."""

    nn = torch.nn
    functional = torch.nn.functional

    class BasicBlock(nn.Module):
        def __init__(self, in_planes: int, out_planes: int, kernel_size: int) -> None:
            super().__init__()
            padding = (int(kernel_size) - 1) // 2
            self.layers = nn.Sequential(
                nn.Conv2d(
                    int(in_planes),
                    int(out_planes),
                    kernel_size=int(kernel_size),
                    stride=1,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(int(out_planes)),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: Any) -> Any:
            return self.layers(x)

    class GlobalAveragePooling(nn.Module):
        def forward(self, feat: Any) -> Any:
            num_channels = feat.size(1)
            return functional.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)

    class NetworkInNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            n_channels = 192
            n_channels2 = 160
            n_channels3 = 96
            blocks: list[Any] = [nn.Sequential() for _ in range(4)]

            blocks[0].add_module("Block1_ConvB1", BasicBlock(3, n_channels, 5))
            blocks[0].add_module("Block1_ConvB2", BasicBlock(n_channels, n_channels2, 1))
            blocks[0].add_module("Block1_ConvB3", BasicBlock(n_channels2, n_channels3, 1))
            blocks[0].add_module("Block1_MaxPool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

            blocks[1].add_module("Block2_ConvB1", BasicBlock(n_channels3, n_channels, 5))
            blocks[1].add_module("Block2_ConvB2", BasicBlock(n_channels, n_channels, 1))
            blocks[1].add_module("Block2_ConvB3", BasicBlock(n_channels, n_channels, 1))
            blocks[1].add_module("Block2_AvgPool", nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

            blocks[2].add_module("Block3_ConvB1", BasicBlock(n_channels, n_channels, 3))
            blocks[2].add_module("Block3_ConvB2", BasicBlock(n_channels, n_channels, 1))
            blocks[2].add_module("Block3_ConvB3", BasicBlock(n_channels, n_channels, 1))

            blocks[3].add_module("Block4_ConvB1", BasicBlock(n_channels, n_channels, 3))
            blocks[3].add_module("Block4_ConvB2", BasicBlock(n_channels, n_channels, 1))
            blocks[3].add_module("Block4_ConvB3", BasicBlock(n_channels, n_channels, 1))

            blocks.append(nn.Sequential())
            blocks[-1].add_module("GlobalAveragePooling", GlobalAveragePooling())

            self._feature_blocks = nn.ModuleList(blocks)
            self.all_feat_names = ["conv1", "conv2", "conv3", "conv4", "classifier"]

        def _parse_out_keys_arg(self, out_feat_keys: list[str] | None) -> tuple[list[str], int]:
            out_keys = [self.all_feat_names[-1]] if out_feat_keys is None else list(out_feat_keys)
            if not out_keys:
                raise ValueError("Empty list of output feature keys.")
            for i, key in enumerate(out_keys):
                if key not in self.all_feat_names:
                    raise ValueError(
                        f"Feature {key!r} does not exist. Existing features: {self.all_feat_names}"
                    )
                if key in out_keys[:i]:
                    raise ValueError(f"Duplicate output feature key: {key!r}")
            return out_keys, max(self.all_feat_names.index(key) for key in out_keys)

        def forward(self, x: Any, out_feat_keys: list[str] | None = None) -> Any:
            out_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
            out_feats = [None] * len(out_keys)
            feat = x
            for i in range(max_out_feat + 1):
                feat = self._feature_blocks[i](feat)
                key = self.all_feat_names[i]
                if key in out_keys:
                    out_feats[out_keys.index(key)] = feat
            return out_feats[0] if len(out_feats) == 1 else out_feats

    class Regressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.nin = NetworkInNetwork()
            self.fc = nn.Linear(384, 8)

        def forward(self, x1: Any, x2: Any, out_feat_keys: list[str] | None = None) -> Any:
            f1 = self.nin(x1, out_feat_keys)
            f2 = self.nin(x2, out_feat_keys)
            if out_feat_keys is None:
                pred = self.fc(torch.cat((f1, f2), dim=1))
                return f1, f2, pred
            return f1, f2

    return Regressor()


@dataclass
class AetStep:
    """Extract CIFAR AET features from an AET checkpoint."""

    source: str = "checkpoint"
    preset: str = "poisson_cifar10_projective"
    checkpoint_path: str | None = None
    checkpoint_name: str | None = None
    model_cache_dir: str | None = None
    features_path: str | None = None
    labels_path: str | None = None
    extracted_npy_path: str | None = None
    train_offset: int = 0
    test_offset: int = 50_000
    expected_rows: int = _PRECOMPUTED_CIFAR_ROWS
    feature_layer: str | None = None
    input_scaling: str | None = None
    unit_normalize: bool | None = None
    batch_size: int = 128
    device: str | None = "auto"
    expected_features_sha256: str | None = field(default=None, kw_only=True)
    expected_labels_sha256: str | None = field(default=None, kw_only=True)

    model_: Any = field(default=None, init=False, repr=False)
    torch_: Any = field(default=None, init=False, repr=False)
    device_: str | None = field(default=None, init=False, repr=False)
    checkpoint_path_: Path | None = field(default=None, init=False, repr=False)
    checkpoint_sha256_: str | None = field(default=None, init=False, repr=False)
    info_: dict[str, Any] | None = field(default=None, init=False, repr=False)
    precomputed_info_: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def _apply_preset(self) -> dict[str, Any]:
        key = str(self.preset).strip().lower().replace("-", "_")
        aliases = {
            "poisson_cifar10": "poisson_cifar10_projective",
            "aet_cifar10_projective": "poisson_cifar10_projective",
        }
        key = aliases.get(key, key)
        if key not in AET_PRESETS:
            known = ", ".join(sorted(AET_PRESETS))
            raise PreprocessValidationError(
                f"Unknown AET preset {self.preset!r}; known presets: {known}"
            )
        self.preset = key
        preset = dict(AET_PRESETS[key])
        self.checkpoint_name = self.checkpoint_name or str(preset["checkpoint_name"])
        self.feature_layer = self.feature_layer or str(preset["feature_layer"])
        self.input_scaling = self.input_scaling or str(preset["input_scaling"])
        if self.unit_normalize is None:
            self.unit_normalize = bool(preset["unit_normalize"])
        return preset

    def _validate_params(self) -> dict[str, Any]:
        preset = self._apply_preset()
        if self.source not in {"checkpoint", "precomputed"}:
            raise PreprocessValidationError("source must be checkpoint or precomputed")
        if int(self.batch_size) <= 0:
            raise PreprocessValidationError("batch_size must be > 0")
        if int(self.train_offset) < 0 or int(self.test_offset) < 0:
            raise PreprocessValidationError("train_offset and test_offset must be >= 0")
        if int(self.expected_rows) <= 0:
            raise PreprocessValidationError("expected_rows must be > 0")
        for field_name in ("expected_features_sha256", "expected_labels_sha256"):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized = str(value).strip().lower()
            if len(normalized) != 64 or any(ch not in _SHA256_HEX for ch in normalized):
                raise PreprocessValidationError(f"{field_name} must be a SHA-256 hex digest")
            setattr(self, field_name, normalized)
        if self.feature_layer not in _VALID_FEATURE_LAYERS:
            known = ", ".join(sorted(_VALID_FEATURE_LAYERS))
            raise PreprocessValidationError(f"feature_layer must be one of: {known}")
        if self.input_scaling not in {"cifar10", "none"}:
            raise PreprocessValidationError("input_scaling must be cifar10 or none")
        return preset

    def _cache_root(self) -> Path:
        if self.model_cache_dir:
            return Path(self.model_cache_dir).expanduser().resolve()
        return _default_aet_cache_dir()

    def _resolve_checkpoint_path(self) -> Path:
        if self.checkpoint_path:
            return Path(self.checkpoint_path).expanduser().resolve()
        root = self._cache_root()
        return root / _safe_cache_component(str(self.preset)) / str(self.checkpoint_name)

    def _checkpoint_info(self, path: Path) -> dict[str, Any]:
        exists = path.exists()
        sha = _file_sha256(path) if exists else None
        self.checkpoint_path_ = path
        self.checkpoint_sha256_ = sha
        return {
            "path": str(path),
            "exists": bool(exists),
            "sha256": sha,
            "expected_default_dir": str(
                _default_aet_cache_dir() / _safe_cache_component(str(self.preset))
            ),
            "official_code": "https://github.com/maple-research-lab/AET",
        }

    def _features_path(self) -> Path:
        if self.features_path:
            return Path(self.features_path).expanduser().resolve()
        return _default_precomputed_features_path()

    def _labels_path(self) -> Path:
        if self.labels_path:
            return Path(self.labels_path).expanduser().resolve()
        return _default_precomputed_labels_path()

    def _extracted_npy_path(self, features_path: Path) -> Path:
        if self.extracted_npy_path:
            return Path(self.extracted_npy_path).expanduser().resolve()
        return features_path.with_suffix(".npy")

    def _build_info(
        self,
        *,
        checkpoint: dict[str, Any],
        device: str | None,
        produced: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        features = produced.get("features.aet") if produced is not None else None
        shape = getattr(features, "shape", None)
        dtype = getattr(features, "dtype", None)
        return {
            "kind": "aet_cifar_nin",
            "schema_version": 1,
            "preset": str(self.preset),
            "source": str(self.source),
            "params": {
                "feature_layer": str(self.feature_layer),
                "input_scaling": str(self.input_scaling),
                "unit_normalize": bool(self.unit_normalize),
                "batch_size": int(self.batch_size),
                "expected_shape": [3, 32, 32],
            },
            "training": {
                "external_checkpoint": True,
                "uses_labels": False,
                "paper": "Zhang et al. 2019 AET vs AED; Poisson paper uses default AET for CIFAR-10",
                "paper_defaults": {
                    "architecture": "Network-In-Network, 4 blocks",
                    "transformation": "projective",
                    "epochs": 1500,
                    "optimizer": "SGD",
                    "batch_size": 512,
                    "lr_schedule": [240, 480, 640, 800, 1000],
                },
            },
            "checkpoint": checkpoint,
            "runtime": {"device": None if device is None else str(device)},
            "output": {
                "artifact": "features.aet",
                "shape": None if shape is None else [int(dim) for dim in shape],
                "dtype": None if dtype is None else str(dtype),
            },
        }

    def _build_precomputed_info(
        self,
        *,
        features_path: Path,
        labels_path: Path,
        extracted_npy_path: Path,
        features_sha256: str,
        labels_sha256: str,
        row_offset: int,
        produced: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        features = produced.get("features.aet") if produced is not None else None
        shape = getattr(features, "shape", None)
        dtype = getattr(features, "dtype", None)
        return {
            "kind": "precomputed_aet",
            "schema_version": 1,
            "preset": str(self.preset),
            "source": "precomputed",
            "params": {
                "feature_layer": str(self.feature_layer),
                "unit_normalize": bool(self.unit_normalize),
                "train_offset": int(self.train_offset),
                "test_offset": int(self.test_offset),
                "expected_rows": int(self.expected_rows),
            },
            "training": {
                "external_features": True,
                "uses_labels": False,
                "paper": "Poisson paper CIFAR-10 branch uses AET features normalized to unit vectors",
            },
            "alignment": {
                "uses_dataset_labels_for_row_alignment": True,
                "row_offset": int(row_offset),
            },
            "files": {
                "features_npz": {
                    "path": str(features_path),
                    "sha256": features_sha256,
                    "expected_sha256": self.expected_features_sha256,
                    "source_url": "https://www-users.cse.umn.edu/~jwcalder/Data/cifar_aet.npz",
                },
                "labels_npz": {
                    "path": str(labels_path),
                    "sha256": labels_sha256,
                    "expected_sha256": self.expected_labels_sha256,
                    "source_url": "https://www-users.cse.umn.edu/~jwcalder/Data/cifar_labels.npz",
                },
                "features_npy_cache": str(extracted_npy_path),
            },
            "output": {
                "artifact": "features.aet",
                "shape": None if shape is None else [int(dim) for dim in shape],
                "dtype": None if dtype is None else str(dtype),
            },
        }

    def _load_model(self) -> None:
        preset = self._validate_params()
        checkpoint_path = self._resolve_checkpoint_path()
        checkpoint = self._checkpoint_info(checkpoint_path)
        if not checkpoint_path.exists():
            raise PreprocessValidationError(
                "AetStep checkpoint not found at "
                f"{checkpoint_path}. Put the AET checkpoint there or set checkpoint_path."
            )
        torch = require_optional(
            module="torch",
            extra="preprocess",
            purpose="vision.aet preprocessing step",
        )

        device = resolve_device_name(self.device, torch=torch) or "cpu"
        model = make_aet_regressor(torch)
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = (
            payload["state_dict"]
            if isinstance(payload, dict) and "state_dict" in payload
            else payload
        )
        if not isinstance(state_dict, dict):
            raise PreprocessValidationError("AetStep checkpoint must contain a torch state_dict")
        state_dict = _strip_state_prefix(state_dict, "module.")
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        self.model_ = model
        self.torch_ = torch
        self.device_ = str(device)
        self.info_ = self._build_info(checkpoint=checkpoint, device=str(device))
        if tuple(preset["expected_shape"]) != (3, 32, 32):
            raise PreprocessValidationError("AetStep currently supports only 3x32x32 CIFAR images")

    def _resolve_precomputed_indices(
        self, labels: np.ndarray, raw_y: Any
    ) -> tuple[np.ndarray, int]:
        y = np.asarray(to_numpy(raw_y)).reshape(-1).astype(np.int64, copy=False)
        candidates = (int(self.train_offset), int(self.test_offset))
        for offset in candidates:
            end = offset + int(y.shape[0])
            if end <= int(labels.shape[0]) and np.array_equal(labels[offset:end], y):
                return np.arange(offset, end, dtype=np.int64), offset
        raise PreprocessValidationError(
            "Could not align raw.y with CIFAR labels. This precomputed AET "
            "mode expects unshuffled CIFAR-10 train/test order; avoid dataset.options.seed "
            "or use the checkpoint-backed AET path."
        )

    def _encode_precomputed(self, store: ArtifactStore) -> np.ndarray:
        self._validate_params()
        features_path = self._features_path()
        labels_path = self._labels_path()
        if not features_path.exists():
            raise PreprocessValidationError(
                f"Precomputed AET features not found at {features_path}"
            )
        if not labels_path.exists():
            raise PreprocessValidationError(f"Precomputed CIFAR labels not found at {labels_path}")

        features_sha = _file_sha256(features_path)
        if (
            self.expected_features_sha256 is not None
            and features_sha != self.expected_features_sha256
        ):
            raise PreprocessValidationError(
                f"Precomputed AET features SHA-256 mismatch at {features_path}"
            )
        labels_sha = _file_sha256(labels_path)
        if self.expected_labels_sha256 is not None and labels_sha != self.expected_labels_sha256:
            raise PreprocessValidationError(
                f"Precomputed CIFAR labels SHA-256 mismatch at {labels_path}"
            )

        extracted_npy_path = self._extracted_npy_path(features_path)
        _ensure_extracted_npy(
            features_path,
            extracted_npy_path,
            source_npz_sha256=features_sha,
        )
        features_npy_path = extracted_npy_path
        probe = np.load(features_npy_path, mmap_mode="r", allow_pickle=False)
        if probe.dtype != np.float32:
            features_npy_path = _float32_npy_cache_path(extracted_npy_path)
            _ensure_float32_npy(
                extracted_npy_path,
                features_npy_path,
                source_npz_sha256=features_sha,
            )

        labels = np.asarray(_load_npz_member(labels_path, "labels"), dtype=np.int64)
        expected_rows = int(self.expected_rows)
        if int(labels.shape[0]) != expected_rows:
            raise PreprocessValidationError(
                f"Expected {expected_rows} precomputed CIFAR labels, got {labels.shape}"
            )
        indices, row_offset = self._resolve_precomputed_indices(labels, store.require("raw.y"))

        features = np.load(features_npy_path, mmap_mode="r", allow_pickle=False)
        if int(features.shape[0]) != expected_rows:
            raise PreprocessValidationError(
                f"Expected {expected_rows} AET rows, got {features.shape}"
            )
        if indices.size == 0:
            Z = np.empty((0, int(features.shape[1])), dtype=np.float32)
        else:
            start = int(indices[0])
            if np.array_equal(indices, np.arange(start, start + int(indices.size))):
                Z = np.array(
                    features[start : start + int(indices.size)],
                    dtype=np.float32,
                    copy=True,
                )
            else:
                Z = np.array(features[indices], dtype=np.float32, copy=True)

        if self.unit_normalize:
            norms = np.linalg.norm(Z, axis=1, keepdims=True)
            norms = np.where(norms > 1.0e-12, norms, 1.0).astype(np.float32)
            Z /= norms

        self.precomputed_info_ = self._build_precomputed_info(
            features_path=features_path,
            labels_path=labels_path,
            extracted_npy_path=features_npy_path,
            features_sha256=features_sha,
            labels_sha256=labels_sha,
            row_offset=row_offset,
            produced={"features.aet": Z},
        )
        return Z

    def _prepare_images(self, X: Any) -> np.ndarray:
        arr = np.asarray(to_numpy(X), dtype=np.float32)
        prepared = prepare_image_array(arr)
        if prepared is None:
            raise PreprocessValidationError("AetStep expects image-like raw.X")
        arr4, _single, layout = prepared
        if layout == "NHWC":
            arr4 = np.transpose(arr4, (0, 3, 1, 2))
        if int(arr4.shape[1]) == 1:
            arr4 = np.repeat(arr4, 3, axis=1)
        if tuple(int(v) for v in arr4.shape[1:]) != (3, 32, 32):
            raise PreprocessValidationError(
                "AetStep expects images shaped as 3x32x32 after layout conversion; "
                "use vision.ensure_num_channels and vision.resize before vision.aet"
            )
        if int(arr4.shape[0]) == 0:
            return np.ascontiguousarray(arr4, dtype=np.float32)
        if self.input_scaling == "cifar10":
            if float(np.nanmax(arr4)) > 2.0:
                arr4 = arr4 / 255.0
            mean = np.asarray((0.4914, 0.4822, 0.4465), dtype=np.float32).reshape(1, 3, 1, 1)
            std = np.asarray((0.2023, 0.1994, 0.2010), dtype=np.float32).reshape(1, 3, 1, 1)
            arr4 = (arr4 - mean) / std
        return np.ascontiguousarray(arr4, dtype=np.float32)

    def _encode(self, X: Any) -> np.ndarray:
        if self.model_ is None or self.torch_ is None or self.device_ is None:
            self._load_model()
        assert self.model_ is not None
        assert self.torch_ is not None
        assert self.device_ is not None

        torch = self.torch_
        arr = self._prepare_images(X)
        if int(arr.shape[0]) == 0:
            return np.empty((0, _feature_dim(str(self.feature_layer))), dtype=np.float32)
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, int(arr.shape[0]), int(self.batch_size)):
                end = min(start + int(self.batch_size), int(arr.shape[0]))
                batch = torch.as_tensor(arr[start:end], dtype=torch.float32, device=self.device_)
                feat = self.model_.nin(batch, out_feat_keys=[str(self.feature_layer)])
                feat = feat.reshape(feat.shape[0], -1)
                outputs.append(feat.detach().cpu().numpy().astype(np.float32, copy=False))
        Z = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
        if self.unit_normalize:
            norms = np.linalg.norm(Z, axis=1, keepdims=True)
            norms = np.where(norms > 1.0e-12, norms, 1.0).astype(np.float32)
            Z = (Z / norms).astype(np.float32, copy=False)
        return Z

    def runtime_artifacts(
        self, *, produced: dict[str, Any] | None = None, split: str = ""
    ) -> dict[str, Any]:
        if self.precomputed_info_ is not None:
            files = self.precomputed_info_["files"]
            info = self._build_precomputed_info(
                features_path=Path(files["features_npz"]["path"]),
                labels_path=Path(files["labels_npz"]["path"]),
                extracted_npy_path=Path(files["features_npy_cache"]),
                features_sha256=str(files["features_npz"]["sha256"]),
                labels_sha256=str(files["labels_npz"]["sha256"]),
                row_offset=int(self.precomputed_info_["alignment"]["row_offset"]),
                produced=produced,
            )
        elif self.info_ is not None:
            info = self._build_info(
                checkpoint=self.info_["checkpoint"],
                device=self.device_,
                produced=produced,
            )
        elif produced is not None and isinstance(produced.get("features.aet.info"), dict):
            info = dict(produced["features.aet.info"])
            features = produced.get("features.aet")
            shape = getattr(features, "shape", None)
            dtype = getattr(features, "dtype", None)
            info["output"] = {
                "artifact": "features.aet",
                "shape": None if shape is None else [int(dim) for dim in shape],
                "dtype": None if dtype is None else str(dtype),
            }
        else:
            self._validate_params()
            checkpoint = self._checkpoint_info(self._resolve_checkpoint_path())
            info = self._build_info(checkpoint=checkpoint, device=self.device_, produced=produced)
        info["runtime"] = dict(info.get("runtime") or {})
        info["runtime"]["split"] = str(split)
        return {"features.aet.info": info}

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        del rng
        if self.source == "precomputed":
            Z = self._encode_precomputed(store)
        else:
            Z = self._encode(store.require("raw.X"))
        produced = {"features.aet": Z}
        produced.update(self.runtime_artifacts(produced=produced))
        return produced
