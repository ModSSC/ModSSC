from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from modssc.inductive.deep import TorchModelBundle, build_torch_bundle_from_classifier
from modssc.inductive.deep.match_primitives import FixedSSLBatchSampler
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.helpers import match_trainer
from modssc.inductive.methods.helpers.match_trainer import MatchStepResult, MatchTrainerConfig
from modssc.inductive.types import DeviceSpec, InductiveDataset
from modssc.runtime.continuation import PlannedContinuation

_FIX_PROFILE = "paper:sohn2020-cifar10-table2-250:diagnostic-dev"
_FLEX_PROFILE = "paper:zhang2021-cifar10-table1-250:diagnostic-dev"
_REAL_MATCH_CONTRACT_VALIDATOR = match_trainer._validate_match_bundle_contract


def _trainer_config(
    profile: str,
    *,
    allow_short_run: bool | None = None,
) -> MatchTrainerConfig:
    google = "sohn2020" in profile
    return MatchTrainerConfig(
        reference_implementation="google_fixmatch" if google else "torchssl",
        sampler_mode="shuffle_repeat" if google else "replacement",
        sampler_shuffle_buffer=8192,
        augmentation_profile="google_fixmatch_ra" if google else "torchssl_ra",
        interleave_bn=google,
        evaluation_interval_steps=1024 if google else 5000,
        checkpoint_interval_steps=1024 if google else 5000,
        reporting_policy=("median_last_checkpoints" if google else "best_historical_checkpoint"),
        reporting_window_checkpoints=20,
        allow_short_run=(
            profile.endswith(":diagnostic-dev") if allow_short_run is None else allow_short_run
        ),
    )


@pytest.fixture(autouse=True)
def _isolate_loop_tests_from_reference_bundle_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loop tests use tiny linear models; contract tests exercise real WRNs below."""

    monkeypatch.setattr(
        match_trainer,
        "_validate_match_bundle_contract",
        lambda **_kwargs: ({"schema_version": 1, "test_fixture": True}, "f" * 64),
    )


def _identity() -> dict[str, str]:
    return {
        "task_id": "match-task",
        "campaign_identity_sha256": "match-campaign",
    }


def _write_pointer(
    root: Path,
    *,
    generation: object = "generation",
    schema_version: object = 1,
    task_id: object = "match-task",
    identity_sha256: object = "match-campaign",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "CURRENT.json").write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "task_id": task_id,
                "identity_sha256": identity_sha256,
                "generation": generation,
            }
        ),
        encoding="utf-8",
    )


def _saved_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[match_trainer._CheckpointStore, Path]:
    root = tmp_path / "checkpoint"
    monkeypatch.setenv("MODSSC_CHECKPOINT_ROOT", str(root))
    store = match_trainer._CheckpointStore(identity=_identity())
    store.save({"identity": _identity(), "value": 7}, step=1, reason="periodic")
    generation = store._current_generation()
    assert generation is not None
    return store, generation


def _bundle(*, full: bool = False) -> TorchModelBundle:
    model = torch.nn.Linear(3, 2, dtype=torch.float64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    if not full:
        return TorchModelBundle(model=model, optimizer=optimizer)

    class Scaler:
        def __init__(self) -> None:
            self.value = 11

        def state_dict(self) -> dict[str, int]:
            return {"value": self.value}

        def load_state_dict(self, state: dict[str, int]) -> None:
            self.value = int(state["value"])

    return TorchModelBundle(
        model=model,
        optimizer=optimizer,
        ema_model=copy.deepcopy(model),
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
        scaler=Scaler(),
    )


def _reference_bundle(profile: str) -> TorchModelBundle:
    google = "sohn2020" in profile
    params: dict[str, Any] = {
        "depth": 28,
        "widen_factor": 2,
        "reference_implementation": "google_fixmatch" if google else "torchssl",
        "bn_momentum": 0.001,
        "bn_eps": 0.001,
        "optimizer": "sgd",
        "lr": 0.03,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 0.0005,
        "decay_bias_and_norm": False,
        "scheduler": "cosine",
        "max_steps": 1 << 20,
        "cosine_cycles": 7.0 / 16.0,
        "ema_decay": 0.999,
        "predict_with_ema": True,
    }
    if not google:
        params.update(
            input_mean=[0.4913725490196078, 0.4823529411764706, 0.44666666666666666],
            input_std=[0.24705882352941178, 0.24352941176470588, 0.2615686274509804],
        )
    return build_torch_bundle_from_classifier(
        classifier_id="wide_resnet_cifar",
        classifier_backend="torch",
        classifier_params=params,
        sample=torch.zeros((1, 3, 32, 32)),
        num_classes=10,
        seed=3,
        ema=True,
    )


@pytest.mark.parametrize(
    ("profile", "sampler_mode", "augmentation", "initialization"),
    [
        (
            _FIX_PROFILE,
            "shuffle_repeat",
            "google_fixmatch_ra",
            "google_fixmatch_variance_scaling",
        ),
        (
            _FLEX_PROFILE,
            "replacement",
            "torchssl_ra",
            "torchssl_kaiming_normal",
        ),
    ],
)
def test_match_reference_bundle_contract_is_verified_and_fingerprinted(
    profile: str,
    sampler_mode: str,
    augmentation: str,
    initialization: str,
) -> None:
    bundle = _reference_bundle(profile)
    sampler = FixedSSLBatchSampler(
        250,
        50_000,
        labeled_batch_size=64,
        unlabeled_batch_size=448,
        seed=7,
        mode=sampler_mode,
        shuffle_buffer=8192,
    )
    contract, fingerprint = _REAL_MATCH_CONTRACT_VALIDATOR(
        config=_trainer_config(profile),
        bundle=bundle,
        batch_size=64,
        mu=7,
        sampler_contract=sampler.contract(),
    )
    assert contract["augmentation"] == augmentation
    assert contract["initialization"] == initialization
    assert contract["architecture"]["depth"] == 28
    assert contract["batches"] == {"labeled": 64, "unlabeled": 448, "mu": 7}
    assert len(fingerprint) == 64
    assert fingerprint == match_trainer._canonical_sha256(contract)


def test_match_reference_bundle_contract_rejects_executable_mismatch() -> None:
    bundle = _reference_bundle(_FIX_PROFILE)
    bad_optimizer = torch.optim.Adam(bundle.model.parameters(), lr=0.03)
    mislabeled = TorchModelBundle(
        model=bundle.model,
        optimizer=bad_optimizer,
        ema_model=bundle.ema_model,
        scheduler=bundle.scheduler,
        meta=bundle.meta,
    )
    sampler = FixedSSLBatchSampler(
        250,
        50_000,
        labeled_batch_size=64,
        unlabeled_batch_size=448,
        seed=7,
        mode="shuffle_repeat",
        shuffle_buffer=8192,
    )
    with pytest.raises(InductiveValidationError, match="optimizer.type"):
        _REAL_MATCH_CONTRACT_VALIDATOR(
            config=_trainer_config(_FIX_PROFILE),
            bundle=mislabeled,
            batch_size=64,
            mu=7,
            sampler_contract=sampler.contract(),
        )


@pytest.mark.parametrize(
    ("case", "expected_field"),
    [
        ("batch_size", "batch_size"),
        ("mu", "mu"),
        ("sampler", "sampler.mode"),
        ("model_type", "model.type"),
        ("model_depth", "model.depth"),
        ("model_no_batch_norm", "model.batch_norm_count"),
        ("model_batch_norm_momentum", "model.bn_momentum"),
        ("model_batch_norm_eps", "model.bn_eps"),
        ("google_input_mean_only", "model.input_normalization"),
        ("google_input_std_only", "model.input_normalization"),
        ("torchssl_input_mean_missing", "model.input_mean"),
        ("torchssl_input_mean_wrong", "model.input_mean"),
        ("torchssl_input_std_missing", "model.input_std"),
        ("torchssl_input_std_wrong", "model.input_std"),
        ("optimizer_lr", "optimizer.lr"),
        ("optimizer_momentum", "optimizer.momentum"),
        ("optimizer_nesterov", "optimizer.nesterov"),
        ("optimizer_group_decay", "optimizer.parameter_group_weight_decay"),
        ("scheduler_missing", "scheduler.type"),
        ("scheduler_wrong_type", "scheduler.type"),
        ("ema_missing", "ema_model.type"),
        ("ema_wrong_type", "ema_model.type"),
        ("ema_state_keys", "ema_model.state_keys"),
        ("ema_requires_grad", "ema_model.requires_grad"),
        ("meta_missing", "bundle.meta"),
        ("meta_string_mismatch", "bundle.meta.classifier_id"),
        ("meta_float_wrong_type", "bundle.meta.lr"),
    ],
)
def test_match_reference_bundle_contract_rejects_each_frozen_invariant(
    case: str,
    expected_field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _FLEX_PROFILE if case.startswith("torchssl_") else _FIX_PROFILE
    bundle = _reference_bundle(profile)
    batch_size = 64
    mu = 7
    sampler_mode = "replacement" if profile == _FLEX_PROFILE else "shuffle_repeat"

    if case == "batch_size":
        batch_size = 32
    elif case == "mu":
        mu = 1
    elif case == "sampler":
        sampler_mode = "replacement"
    elif case == "model_type":
        bundle = TorchModelBundle(
            model=torch.nn.Linear(3, 2),
            optimizer=bundle.optimizer,
            ema_model=bundle.ema_model,
            scheduler=bundle.scheduler,
            meta=bundle.meta,
        )
    elif case == "model_depth":
        bundle.model.depth = 16
    elif case == "model_no_batch_norm":
        monkeypatch.setattr(bundle.model, "modules", lambda: iter(()))
    elif case == "model_batch_norm_momentum":
        next(
            module for module in bundle.model.modules() if isinstance(module, torch.nn.BatchNorm2d)
        ).momentum = 0.1
    elif case == "model_batch_norm_eps":
        next(
            module for module in bundle.model.modules() if isinstance(module, torch.nn.BatchNorm2d)
        ).eps = 0.1
    elif case == "google_input_mean_only":
        bundle.model.input_mean = torch.zeros((1, 3, 1, 1))
    elif case == "google_input_std_only":
        bundle.model.input_std = torch.ones((1, 3, 1, 1))
    elif case == "torchssl_input_mean_missing":
        bundle.model.input_mean = None
    elif case == "torchssl_input_mean_wrong":
        bundle.model.input_mean = torch.zeros((1, 3, 1, 1))
    elif case == "torchssl_input_std_missing":
        bundle.model.input_std = None
    elif case == "torchssl_input_std_wrong":
        bundle.model.input_std = torch.ones((1, 3, 1, 1))
    elif case == "optimizer_lr":
        bundle.optimizer.defaults["lr"] = 0.04
    elif case == "optimizer_momentum":
        bundle.optimizer.defaults["momentum"] = 0.8
    elif case == "optimizer_nesterov":
        bundle.optimizer.defaults["nesterov"] = False
    elif case == "optimizer_group_decay":
        for group in bundle.optimizer.param_groups:
            group["weight_decay"] = 0.0
    elif case == "scheduler_missing":
        bundle = TorchModelBundle(
            model=bundle.model,
            optimizer=bundle.optimizer,
            ema_model=bundle.ema_model,
            meta=bundle.meta,
        )
    elif case == "scheduler_wrong_type":
        bundle = TorchModelBundle(
            model=bundle.model,
            optimizer=bundle.optimizer,
            ema_model=bundle.ema_model,
            scheduler=torch.optim.lr_scheduler.StepLR(bundle.optimizer, step_size=1),
            meta=bundle.meta,
        )
    elif case == "ema_missing":
        bundle = TorchModelBundle(
            model=bundle.model,
            optimizer=bundle.optimizer,
            scheduler=bundle.scheduler,
            meta=bundle.meta,
        )
    elif case == "ema_wrong_type":
        bundle = TorchModelBundle(
            model=bundle.model,
            optimizer=bundle.optimizer,
            ema_model=torch.nn.Linear(1, 1),
            scheduler=bundle.scheduler,
            meta=bundle.meta,
        )
    elif case == "ema_state_keys":
        bundle.ema_model.register_buffer("coverage_extra", torch.tensor(1))
    elif case == "ema_requires_grad":
        next(bundle.ema_model.parameters()).requires_grad_(True)
    elif case == "meta_missing":
        bundle = TorchModelBundle(
            model=bundle.model,
            optimizer=bundle.optimizer,
            ema_model=bundle.ema_model,
            scheduler=bundle.scheduler,
        )
    elif case in {"meta_string_mismatch", "meta_float_wrong_type"}:
        meta = dict(bundle.meta)
        if case == "meta_string_mismatch":
            meta["classifier_id"] = "not-wide-resnet"
        else:
            meta["lr"] = "not-a-number"
        bundle = TorchModelBundle(
            model=bundle.model,
            optimizer=bundle.optimizer,
            ema_model=bundle.ema_model,
            scheduler=bundle.scheduler,
            meta=meta,
        )

    sampler = FixedSSLBatchSampler(
        250,
        50_000,
        labeled_batch_size=64,
        unlabeled_batch_size=448,
        seed=7,
        mode=sampler_mode,
        shuffle_buffer=8192,
    )
    with pytest.raises(InductiveValidationError, match=expected_field):
        _REAL_MATCH_CONTRACT_VALIDATOR(
            config=_trainer_config(profile),
            bundle=bundle,
            batch_size=batch_size,
            mu=mu,
            sampler_contract=sampler.contract(),
        )


def test_paper_runtime_feature_detects_legacy_torch_precision_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"deterministic": False, "warn_only": True}
    matmul = SimpleNamespace(allow_tf32=True)
    cudnn = SimpleNamespace(
        allow_tf32=True,
        deterministic=False,
        benchmark=True,
        conv=SimpleNamespace(),
    )
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul), cudnn=cudnn),
        are_deterministic_algorithms_enabled=lambda: state["deterministic"],
        is_deterministic_algorithms_warn_only_enabled=lambda: state["warn_only"],
        use_deterministic_algorithms=lambda enabled, warn_only=False: state.update(
            deterministic=enabled,
            warn_only=warn_only,
        ),
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with match_trainer._paper_deterministic_runtime(fake_torch) as contract:
        assert contract["float32_precision"] == {}
        assert contract["legacy_allow_tf32"] == {
            "cuda_matmul": False,
            "cudnn": False,
        }
        assert matmul.allow_tf32 is False
        assert cudnn.allow_tf32 is False
    assert matmul.allow_tf32 is True
    assert cudnn.allow_tf32 is True
    assert state == {"deterministic": False, "warn_only": True}


def _data(
    *,
    y_dtype: torch.dtype = torch.int64,
    n_l: int = 3,
    n_u_w: int = 4,
    n_u_s: int | None = None,
    evaluation: bool = False,
) -> InductiveDataset:
    n_u_s = n_u_w if n_u_s is None else n_u_s
    meta: dict[str, Any] = {
        "dataset_fingerprint": "dataset",
        "split_fingerprint": "split",
        "partition_sha256": "partition",
    }
    if evaluation:
        meta["paper_evaluation"] = {
            "X_test": torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=torch.float64,
            ),
            "y_test": np.asarray([0, 1], dtype=np.int64),
        }
    return InductiveDataset(
        X_l=torch.arange(n_l * 3, dtype=torch.float64).reshape(n_l, 3),
        y_l=torch.arange(n_l, dtype=y_dtype).remainder(2),
        X_u_w=torch.arange(n_u_w * 3, dtype=torch.float64).reshape(n_u_w, 3),
        X_u_s=torch.arange(n_u_s * 3, dtype=torch.float64).reshape(n_u_s, 3),
        meta=meta,
    )


def _owner(
    *,
    profile: str = _FIX_PROFILE,
    bundle: TorchModelBundle | None = None,
    batch_size: int = 64,
    mu: int = 7,
    max_steps: int | None = 1,
) -> SimpleNamespace:
    config = _trainer_config(profile)
    return SimpleNamespace(
        spec=SimpleNamespace(
            model_bundle=_bundle() if bundle is None else bundle,
            batch_size=batch_size,
            mu=mu,
            max_steps=max_steps,
            training_mode="fixed_steps",
            reference_implementation=config.reference_implementation,
            sampler_mode=config.sampler_mode,
            sampler_shuffle_buffer=config.sampler_shuffle_buffer,
            augmentation_profile=config.augmentation_profile,
            interleave_bn=config.interleave_bn,
            evaluation_interval_steps=config.evaluation_interval_steps,
            checkpoint_interval_steps=config.checkpoint_interval_steps,
            reporting_policy=config.reporting_policy,
            reporting_window_checkpoints=config.reporting_window_checkpoints,
            allow_short_run=config.allow_short_run,
        )
    )


def _valid_step(
    logits_l: torch.Tensor,
    logits_uw: torch.Tensor,
    logits_us: torch.Tensor,
    y_l: torch.Tensor,
    _idx_u: torch.Tensor,
) -> MatchStepResult:
    loss = F.cross_entropy(logits_l, y_l)
    loss = loss + 0.0 * (logits_uw.sum() + logits_us.sum())
    return MatchStepResult(loss=loss, accepted=3.0, unlabeled=448, diagnostics={})


def _run(
    owner: SimpleNamespace,
    data: Any,
    *,
    method_id: str = "fixmatch",
    step_fn: Any = _valid_step,
    trace_getter: Any = None,
) -> Any:
    return match_trainer.run_fixed_step_match(
        owner,
        data,
        device=DeviceSpec(device="cpu", dtype="float64"),
        seed=17,
        method_id=method_id,
        step_fn=step_fn,
        state_getter=lambda: {},
        state_loader=lambda _state: None,
        trace_getter=trace_getter,
    )


def test_atomic_json_removes_staging_file_after_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "payload.json"

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(match_trainer.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        match_trainer._atomic_json(destination, {"value": 1})
    assert list(tmp_path.iterdir()) == []


def test_checkpoint_store_disabled_properties_and_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODSSC_CHECKPOINT_ROOT", raising=False)
    store = match_trainer._CheckpointStore(identity=_identity())
    assert not store.enabled
    assert store._current_generation() is None
    store.save({"value": 1}, step=1, reason="periodic")
    for property_name in (
        "payload_path",
        "metadata_path",
        "pointer_path",
        "generations_root",
    ):
        with pytest.raises(InductiveValidationError):
            getattr(store, property_name)


@pytest.mark.parametrize(
    ("pointer", "message"),
    [
        ("not-json", "pointer is invalid"),
        (json.dumps(["not", "a", "mapping"]), "must be a mapping"),
    ],
)
def test_checkpoint_pointer_rejects_invalid_json_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pointer: str,
    message: str,
) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "CURRENT.json").write_text(pointer, encoding="utf-8")
    monkeypatch.setenv("MODSSC_CHECKPOINT_ROOT", str(root))
    with pytest.raises(InductiveValidationError, match=message):
        match_trainer._CheckpointStore(identity=_identity())._current_generation()


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 2},
        {"task_id": "other"},
        {"identity_sha256": "other"},
        {"generation": 4},
        {"generation": ""},
        {"generation": "../escape"},
    ],
)
def test_checkpoint_pointer_rejects_every_incompatible_identity_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
) -> None:
    root = tmp_path / "checkpoint"
    _write_pointer(root, **overrides)
    monkeypatch.setenv("MODSSC_CHECKPOINT_ROOT", str(root))
    with pytest.raises(InductiveValidationError, match="identity is incompatible"):
        match_trainer._CheckpointStore(identity=_identity())._current_generation()


def test_checkpoint_pointer_rejects_missing_symlinked_and_escaping_generations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "checkpoint"
    monkeypatch.setenv("MODSSC_CHECKPOINT_ROOT", str(root))
    store = match_trainer._CheckpointStore(identity=_identity())

    _write_pointer(root)
    with pytest.raises(InductiveValidationError, match="generation is missing"):
        store._current_generation()

    generations = root / "generations"
    generations.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (generations / "generation").symlink_to(outside, target_is_directory=True)
    with pytest.raises(InductiveValidationError, match="escapes its root"):
        store._current_generation()


def test_checkpoint_save_reuses_generation_cleans_old_directories_and_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, first_generation = _saved_store(tmp_path, monkeypatch)
    first_payload = torch.load(
        first_generation / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    store.save(first_payload, step=1, reason="periodic")
    assert store._current_generation() == first_generation

    ignored_symlink = store.generations_root / "external-link"
    external = tmp_path / "external"
    external.mkdir()
    ignored_symlink.symlink_to(external, target_is_directory=True)
    store.save({"identity": _identity(), "value": 8}, step=2, reason="complete")
    assert not first_generation.exists()
    assert ignored_symlink.is_symlink()

    real_save = torch.save

    def fail_save(_payload: object, path: object) -> None:
        Path(path).write_bytes(b"partial")
        raise OSError("save failed")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="save failed"):
        store.save({"identity": _identity()}, step=3, reason="periodic")
    monkeypatch.setattr(torch, "save", real_save)
    assert not tuple(store.generations_root.glob(".staging-*"))


def test_checkpoint_load_requires_resume_root_and_complete_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODSSC_CHECKPOINT_ROOT", raising=False)
    monkeypatch.setenv("MODSSC_CHECKPOINT_RESUME", "0")
    assert match_trainer._CheckpointStore(identity=_identity()).load() is None

    monkeypatch.setenv("MODSSC_CHECKPOINT_RESUME", "1")
    with pytest.raises(InductiveValidationError, match="requires MODSSC_CHECKPOINT_ROOT"):
        match_trainer._CheckpointStore(identity=_identity()).load()

    root = tmp_path / "checkpoint"
    monkeypatch.setenv("MODSSC_CHECKPOINT_ROOT", str(root))
    with pytest.raises(InductiveValidationError, match="checkpoint is incomplete"):
        match_trainer._CheckpointStore(identity=_identity()).load()

    _write_pointer(root)
    generation = root / "generations" / "generation"
    generation.mkdir(parents=True)
    with pytest.raises(InductiveValidationError, match="checkpoint is incomplete"):
        match_trainer._CheckpointStore(identity=_identity()).load()


def test_checkpoint_load_rejects_metadata_identity_digest_and_payload_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, generation = _saved_store(tmp_path, monkeypatch)
    monkeypatch.setenv("MODSSC_CHECKPOINT_RESUME", "1")
    metadata_path = generation / "checkpoint.json"
    payload_path = generation / "checkpoint.pt"
    valid_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    valid_payload = torch.load(payload_path, map_location="cpu", weights_only=False)

    metadata_path.write_text("{", encoding="utf-8")
    with pytest.raises(InductiveValidationError, match="metadata is invalid"):
        store.load()

    for mutate, message in (
        (lambda value: value.update(identity={"wrong": True}), "identity is incompatible"),
        (lambda value: value.update(task_id="wrong"), "campaign identity is incompatible"),
        (
            lambda value: value.update(identity_sha256="wrong"),
            "campaign identity is incompatible",
        ),
        (lambda value: value.update(checkpoint_sha256=None), "digest is invalid"),
        (lambda value: value.update(checkpoint_sha256="0" * 64), "digest is invalid"),
    ):
        metadata = copy.deepcopy(valid_metadata)
        mutate(metadata)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(InductiveValidationError, match=message):
            store.load()

    payload_path.write_bytes(b"not a torch checkpoint")
    corrupt_metadata = copy.deepcopy(valid_metadata)
    corrupt_metadata["checkpoint_sha256"] = match_trainer._sha256_file(payload_path)
    metadata_path.write_text(json.dumps(corrupt_metadata), encoding="utf-8")
    with pytest.raises(InductiveValidationError, match="payload is corrupt"):
        store.load()

    torch.save({"identity": {"wrong": True}}, payload_path)
    incompatible_metadata = copy.deepcopy(valid_metadata)
    incompatible_metadata["checkpoint_sha256"] = match_trainer._sha256_file(payload_path)
    incompatible_metadata["history"] = "not-a-list"
    metadata_path.write_text(json.dumps(incompatible_metadata), encoding="utf-8")
    with pytest.raises(InductiveValidationError, match="payload identity is incompatible"):
        store.load()

    torch.save(valid_payload, payload_path)
    no_history = copy.deepcopy(valid_metadata)
    no_history["checkpoint_sha256"] = match_trainer._sha256_file(payload_path)
    no_history["history"] = "not-a-list"
    metadata_path.write_text(json.dumps(no_history), encoding="utf-8")
    assert store.load() == valid_payload


def test_rng_state_roundtrip_cuda_branch_and_invalid_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_state = [torch.tensor([1], dtype=torch.uint8)]
    restored: list[Any] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: cuda_state)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda value: restored.append(value),
    )
    state = match_trainer._rng_state(torch)
    assert state["torch_cuda"] == cuda_state
    match_trainer._restore_rng_state(torch, state)
    assert restored == [cuda_state]

    with pytest.raises(InductiveValidationError, match="RNG state is invalid"):
        match_trainer._restore_rng_state(torch, {})


def test_bundle_state_roundtrip_and_component_compatibility_errors() -> None:
    full = _bundle(full=True)
    state = match_trainer._bundle_state(full)
    match_trainer._load_bundle_state(full, state)

    missing_cases = (
        ("ema_model", full),
        ("scheduler", full),
        ("scaler", full),
    )
    for name, bundle in missing_cases:
        broken = copy.deepcopy(state)
        broken[name] = None
        with pytest.raises(InductiveValidationError, match="model state is invalid"):
            match_trainer._load_bundle_state(bundle, broken)

    plain = _bundle()
    plain_state = match_trainer._bundle_state(plain)
    for name, value, message in (
        ("ema_model", state["ema_model"], "unexpected EMA model"),
        ("scheduler", state["scheduler"], "unexpected scheduler"),
        ("scaler", state["scaler"], "unexpected scaler"),
    ):
        broken = copy.deepcopy(plain_state)
        broken[name] = value
        with pytest.raises(InductiveValidationError, match=message):
            match_trainer._load_bundle_state(plain, broken)

    broken_model = copy.deepcopy(plain_state)
    broken_model["model"] = {"missing.weight": torch.ones(1)}
    with pytest.raises(InductiveValidationError, match="model state is invalid"):
        match_trainer._load_bundle_state(plain, broken_model)


def test_tensor_indices_accepts_tensor_and_numpy() -> None:
    tensor = torch.tensor([1.0, 2.0])
    converted_tensor = match_trainer._tensor_indices(tensor, device=torch.device("cpu"))
    converted_numpy = match_trainer._tensor_indices(
        np.asarray([3, 4], dtype=np.int32),
        device=torch.device("cpu"),
    )
    assert converted_tensor.dtype == converted_numpy.dtype == torch.int64
    assert converted_tensor.tolist() == [1, 2]
    assert converted_numpy.tolist() == [3, 4]


def test_reference_image_views_use_raw_pool_source_indices_and_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeAugmenter:
        def __init__(self, *, profile: str, seed: int) -> None:
            calls.append({"profile": profile, "seed": seed})

        def apply_batch(self, batch: Any, **kwargs: Any) -> Any:
            calls.append({"batch": batch.clone(), **kwargs})
            return batch + len(calls)

    import modssc.data_augmentation.cifar_reference as cifar_reference

    monkeypatch.setattr(cifar_reference, "CifarReferenceAugmentation", FakeAugmenter)
    X_l = torch.zeros((2, 3, 4, 4))
    X_u_w = torch.ones((3, 3, 4, 4))
    X_u_s = torch.full((3, 3, 4, 4), 2.0)
    raw_u = torch.full((3, 3, 4, 4), 4.0)
    configured = SimpleNamespace(seed=91)
    data = SimpleNamespace(
        X_u=raw_u,
        meta={
            "online_augmentation": configured,
            "source_idx_l": np.asarray([10, 11]),
            "source_idx_u": np.asarray([20, 21, 22]),
        },
    )
    views = match_trainer._reference_batch_views(
        augmentation_profile="torchssl_ra",
        data=data,
        X_l=X_l,
        X_u_w=X_u_w,
        X_u_s=X_u_s,
        idx_l=torch.tensor([1, 0]),
        idx_u=torch.tensor([2, 0]),
        step=7,
    )
    assert calls[0] == {"profile": "torchssl_ra", "seed": 91}
    assert [call["view"] for call in calls[1:]] == [
        "labeled_weak",
        "unlabeled_weak",
        "unlabeled_strong",
    ]
    assert calls[1]["sample_ids"].tolist() == [11, 10]
    assert calls[2]["sample_ids"].tolist() == [22, 20]
    torch.testing.assert_close(calls[2]["batch"], raw_u[[2, 0]])
    assert all(isinstance(view, torch.Tensor) for view in views)

    fallback_data = SimpleNamespace(
        X_u=None,
        meta={
            "augmentation_seed": 5,
            "idx_l": np.asarray([0, 1]),
            "idx_u": np.asarray([0, 1, 2]),
        },
    )
    match_trainer._reference_batch_views(
        augmentation_profile="google_fixmatch_ra",
        data=fallback_data,
        X_l=X_l,
        X_u_w=X_u_w,
        X_u_s=X_u_s,
        idx_l=torch.tensor([0]),
        idx_u=torch.tensor([1]),
        step=0,
    )
    assert calls[4] == {"profile": "google_fixmatch_ra", "seed": 5}
    torch.testing.assert_close(calls[6]["batch"], X_u_w[[1]])


def test_reference_image_views_require_source_indices() -> None:
    X = torch.zeros((2, 3, 4, 4))
    with pytest.raises(InductiveValidationError, match="source sample indices"):
        match_trainer._reference_batch_views(
            augmentation_profile="google_fixmatch_ra",
            data=SimpleNamespace(X_u=X, meta={}),
            X_l=X,
            X_u_w=X,
            X_u_s=X,
            idx_l=torch.tensor([0]),
            idx_u=torch.tensor([0]),
            step=0,
            augmenter=SimpleNamespace(apply_batch=lambda *args, **kwargs: args[0]),
        )


def test_forward_match_noninterleaved_and_all_shape_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(3, 2)
    x_lb = torch.zeros((2, 3))
    x_uw = torch.ones((3, 3))
    x_us = torch.full((3, 3), 2.0)
    outputs = match_trainer._forward_match(
        model,
        x_lb=x_lb,
        x_uw=x_uw,
        x_us=x_us,
        interleave_bn=False,
        mu=1,
    )
    assert [tuple(value.shape) for value in outputs] == [(2, 2), (3, 2), (3, 2)]

    monkeypatch.setattr(match_trainer, "extract_logits", lambda _output: torch.ones(8))
    with pytest.raises(InductiveValidationError, match="logits must be 2D"):
        match_trainer._forward_match(
            model,
            x_lb=x_lb,
            x_uw=x_uw,
            x_us=x_us,
            interleave_bn=False,
            mu=1,
        )

    monkeypatch.setattr(match_trainer, "extract_logits", lambda _output: torch.ones(7, 2))
    with pytest.raises(InductiveValidationError, match="preserve batch size"):
        match_trainer._forward_match(
            model,
            x_lb=x_lb,
            x_uw=x_uw,
            x_us=x_us,
            interleave_bn=False,
            mu=1,
        )

    class SlicedLogits:
        ndim = 2
        shape = (8, 2)

        def __getitem__(self, key: Any) -> torch.Tensor:
            if key == slice(None, 2, None):
                return torch.ones(2, 3)
            if key == slice(2, 5, None):
                return torch.ones(3, 2)
            return torch.ones(3, 2)

    monkeypatch.setattr(match_trainer, "extract_logits", lambda _output: SlicedLogits())
    with pytest.raises(InductiveValidationError, match="class dimension"):
        match_trainer._forward_match(
            model,
            x_lb=x_lb,
            x_uw=x_uw,
            x_us=x_us,
            interleave_bn=False,
            mu=1,
        )

    class UnequalSlices(SlicedLogits):
        def __getitem__(self, key: Any) -> torch.Tensor:
            if key == slice(None, 2, None):
                return torch.ones(2, 2)
            if key == slice(2, 5, None):
                return torch.ones(3, 2)
            return torch.ones(3, 3)

    monkeypatch.setattr(match_trainer, "extract_logits", lambda _output: UnequalSlices())
    with pytest.raises(InductiveValidationError, match="shape mismatch"):
        match_trainer._forward_match(
            model,
            x_lb=x_lb,
            x_uw=x_uw,
            x_us=x_us,
            interleave_bn=False,
            mu=1,
        )


def test_evaluation_payload_accuracy_and_metric_policies() -> None:
    assert match_trainer._evaluation_payload(SimpleNamespace(meta=None)) is None
    assert (
        match_trainer._evaluation_payload(
            SimpleNamespace(meta={"paper_evaluation": {"X_test": torch.ones(1, 3)}})
        )
        is None
    )
    X = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
    model = torch.nn.Linear(3, 2, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        model.bias.zero_()
    model.train()
    bundle = TorchModelBundle(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))
    assert match_trainer._accuracy(bundle, X=X, y=np.asarray([0, 1]), batch_size=1) == 1.0
    assert (
        match_trainer._accuracy(
            bundle,
            X=X,
            y=torch.tensor([0, 1], dtype=torch.int32),
            batch_size=2,
        )
        == 1.0
    )
    assert model.training
    with pytest.raises(InductiveValidationError, match="test split is empty"):
        match_trainer._accuracy(
            bundle,
            X=torch.empty((0, 3), dtype=torch.float64),
            y=torch.empty(0, dtype=torch.int64),
        )

    assert match_trainer._paper_metrics(
        reporting_policy="median_last_checkpoints",
        reporting_window_checkpoints=20,
        history=[],
    ) == {
        "historical_paper_metric": None,
        "fixed_terminal_metric": None,
        "selection_uses_test": False,
    }
    fixed = match_trainer._paper_metrics(
        reporting_policy="median_last_checkpoints",
        reporting_window_checkpoints=20,
        history=[
            {"test_accuracy": 0.7},
            {"test_accuracy": 0.9},
            {"test_accuracy": 0.8},
        ],
    )
    assert fixed["historical_paper_metric"]["test_accuracy"] == 0.8
    assert fixed["fixed_terminal_metric"]["test_accuracy"] == 0.8
    assert not fixed["selection_uses_test"]

    with pytest.raises(InductiveValidationError, match="historical checkpoint evaluation"):
        match_trainer._paper_metrics(
            reporting_policy="best_historical_checkpoint",
            reporting_window_checkpoints=20,
            history=[{"test_accuracy": 0.8, "historical_eligible": False}],
        )
    adaptive = match_trainer._paper_metrics(
        reporting_policy="best_historical_checkpoint",
        reporting_window_checkpoints=20,
        history=[
            {"test_accuracy": 0.7, "historical_eligible": True},
            {"test_accuracy": 0.9, "historical_eligible": True},
            {"test_accuracy": 0.8},
        ],
    )
    assert adaptive["historical_paper_metric"]["test_accuracy"] == 0.9
    assert adaptive["fixed_terminal_metric"]["test_accuracy"] == 0.8
    assert adaptive["selection_uses_test"]


def test_forced_continuation_validation_and_historical_evaluation_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODSSC_FORCE_CONTINUATION_STEP", raising=False)
    assert (
        match_trainer._forced_continuation_step(
            allow_short_run=True,
            target_steps=5,
        )
        is None
    )

    monkeypatch.setenv("MODSSC_FORCE_CONTINUATION_STEP", "2")
    with pytest.raises(InductiveValidationError, match="allow_short_run"):
        match_trainer._forced_continuation_step(
            allow_short_run=False,
            target_steps=5,
        )

    monkeypatch.setenv("MODSSC_FORCE_CONTINUATION_STEP", "invalid")
    with pytest.raises(InductiveValidationError, match="must be an integer"):
        match_trainer._forced_continuation_step(
            allow_short_run=True,
            target_steps=5,
        )

    for step in ("0", "5", "6"):
        monkeypatch.setenv("MODSSC_FORCE_CONTINUATION_STEP", step)
        with pytest.raises(InductiveValidationError, match="inside the configured run"):
            match_trainer._forced_continuation_step(
                allow_short_run=True,
                target_steps=5,
            )

    monkeypatch.setenv("MODSSC_FORCE_CONTINUATION_STEP", "2")
    assert (
        match_trainer._forced_continuation_step(
            allow_short_run=True,
            target_steps=5,
        )
        == 2
    )
    assert match_trainer._historical_evaluation(
        global_step=0,
        target_steps=10_000,
        interval=5000,
    )
    assert not match_trainer._historical_evaluation(
        global_step=1,
        target_steps=10_000,
        interval=5000,
    )
    assert match_trainer._historical_evaluation(
        global_step=5_000,
        target_steps=10_000,
        interval=5000,
    )
    assert not match_trainer._historical_evaluation(
        global_step=9_000,
        target_steps=10_000,
        interval=5000,
    )


def test_continuation_environment_and_public_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODSSC_CONTINUATION_REQUESTED", "1")
    assert match_trainer._continuation_requested()
    monkeypatch.setenv("MODSSC_CONTINUATION_REQUESTED", "0")
    monkeypatch.delenv("MODSSC_CONTINUATION_SIGNAL", raising=False)
    assert not match_trainer._continuation_requested()

    with pytest.raises(PlannedContinuation) as raised:
        match_trainer._raise_continuation()
    assert raised.value.signum == 0


@pytest.mark.parametrize(
    ("owner", "data", "message"),
    [
        (_owner(), None, "data must not be None"),
        (_owner(), SimpleNamespace(X_l=np.ones((2, 3))), "requires torch tensors"),
        (
            _owner(),
            InductiveDataset(
                X_l=torch.ones((2, 3)),
                y_l=torch.tensor([0, 1]),
                X_u_w=None,
                X_u_s=None,
            ),
            "requires weak and strong views",
        ),
        (_owner(), _data(n_l=0), "must be non-empty"),
        (_owner(), _data(n_u_w=0), "must be non-empty"),
        (_owner(), _data(n_u_w=3, n_u_s=2), "must have the same number of rows"),
        (_owner(), _data(y_dtype=torch.int32), "y_l must be int64"),
        (_owner(bundle=None), _data(), "model_bundle must be provided"),
        (_owner(batch_size=32), _data(), "requires batches 64/448"),
        (_owner(mu=1), _data(), "requires batches 64/448"),
        (_owner(max_steps=None), _data(), "inside \\(0, 2\\^20\\]"),
        (_owner(max_steps=(1 << 20) + 1), _data(), "inside \\(0, 2\\^20\\]"),
    ],
)
def test_run_paper_match_rejects_invalid_protocol_inputs(
    monkeypatch: pytest.MonkeyPatch,
    owner: SimpleNamespace,
    data: Any,
    message: str,
) -> None:
    monkeypatch.delenv("MODSSC_CHECKPOINT_ROOT", raising=False)
    if "model_bundle must be provided" in message:
        owner.spec.model_bundle = None
    with pytest.raises(InductiveValidationError, match=message):
        _run(owner, data)


def test_run_paper_match_own_pool_size_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data(n_u_w=3)
    converted = SimpleNamespace(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=None,
        X_u_w=data.X_u_w,
        X_u_s=torch.ones((2, 3), dtype=torch.float64),
        meta=data.meta,
    )
    monkeypatch.setattr(match_trainer, "ensure_torch_data", lambda *_args, **_kwargs: converted)
    with pytest.raises(InductiveValidationError, match="must have equal size"):
        _run(_owner(), data)


def test_full_contract_enforces_and_accepts_exact_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = "paper:sohn2020-cifar10-table2-250"
    with pytest.raises(InductiveValidationError, match="requires exactly 2\\^20"):
        _run(_owner(profile=canonical, max_steps=1), _data())

    monkeypatch.setattr(match_trainer, "MATCH_REFERENCE_TARGET_STEPS", 1)
    result = _run(_owner(profile=canonical, max_steps=1), _data())
    assert result.optimization_steps == 1


@pytest.mark.parametrize(
    ("step_fn", "message"),
    [
        (lambda *_args: object(), "invalid result"),
        (
            lambda logits_l, logits_uw, logits_us, y_l, _idx: MatchStepResult(
                loss=F.cross_entropy(logits_l, y_l) + 0.0 * (logits_uw.sum() + logits_us.sum()),
                accepted=0.0,
                unlabeled=447,
                diagnostics={},
            ),
            "exactly 448",
        ),
    ],
)
def test_run_paper_match_rejects_invalid_step_hook(
    monkeypatch: pytest.MonkeyPatch,
    step_fn: Any,
    message: str,
) -> None:
    monkeypatch.delenv("MODSSC_CHECKPOINT_ROOT", raising=False)
    with pytest.raises(InductiveValidationError, match=message):
        _run(_owner(), _data(), step_fn=step_fn)


def test_run_paper_match_rejects_labels_outside_model_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODSSC_CHECKPOINT_ROOT", raising=False)
    data = _data()
    data = InductiveDataset(
        X_l=data.X_l,
        y_l=torch.full((3,), 4, dtype=torch.int64),
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta=data.meta,
    )
    with pytest.raises(InductiveValidationError, match="within \\[0, n_classes"):
        _run(_owner(), data)


def test_run_adaptive_profile_without_evaluation_or_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODSSC_CHECKPOINT_ROOT", raising=False)
    owner = _owner(profile=_FLEX_PROFILE)
    result = _run(owner, _data(), method_id="flexmatch")
    assert result.optimization_steps == 1
    assert result.evaluation_history == ()
    assert result.paper_metrics["historical_paper_metric"] is None
    assert owner.diagnostics_["reference_stack"] == "torchssl"
    assert owner.diagnostics_["method_state"] == {}


@pytest.mark.parametrize(
    ("profile", "method_id", "expected"),
    [
        (
            _FIX_PROFILE,
            "fixmatch",
            {
                "mode": "shuffle_repeat",
                "reference_algorithm": ("tensorflow.data.Dataset.repeat().shuffle(buffer_size)"),
                "rng_backend": "numpy_pcg64",
                "seed_policy": ("explicit_independent_loader_seeds_from_numpy_seedsequence"),
                "shuffle_buffer": 8192,
                "historical_bitstream_claimed": False,
            },
        ),
        (
            _FLEX_PROFILE,
            "flexmatch",
            {
                "mode": "replacement",
                "reference_algorithm": ("torch.utils.data.RandomSampler(replacement=True)"),
                "rng_backend": "torch_cpu_generator",
                "seed_policy": ("explicit_independent_loader_seeds_from_torch_root_generator"),
                "historical_bitstream_claimed": False,
            },
        ),
    ],
)
def test_run_exposes_stable_sampler_contract(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    method_id: str,
    expected: dict[str, str | int | bool],
) -> None:
    monkeypatch.delenv("MODSSC_CHECKPOINT_ROOT", raising=False)
    owner = _owner(profile=profile)
    _run(owner, _data(), method_id=method_id)
    assert owner.diagnostics_["sampler_contract"] == expected
    runtime = owner.diagnostics_["numeric_runtime_contract"]
    assert runtime["float32_precision"] == {
        "global": "ieee",
        "cuda_matmul": "ieee",
        "cudnn": "ieee",
        "cudnn_conv": "ieee",
    }
    assert runtime["rng_initialization"].startswith("python_numpy_torch")
    probe = owner.diagnostics_["numeric_probe"]
    assert set(probe) >= {
        "initial_model_sha256",
        "batch_indices_sha256",
        "augmented_inputs_sha256",
        "logits_sha256",
        "loss",
        "accepted",
        "step_diagnostics",
    }
    assert ("pseudo_label_mask_sha256" in probe) is (method_id == "fixmatch")


def test_fresh_run_seeds_all_global_rngs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODSSC_CHECKPOINT_ROOT", raising=False)
    owner = _owner()
    data = _data()
    python_seeds: list[int] = []
    numpy_seeds: list[int] = []
    torch_seeds: list[int] = []
    real_python_seed = random.seed
    real_numpy_seed = np.random.seed
    real_torch_seed = torch.manual_seed

    def python_seed(seed: int) -> None:
        python_seeds.append(seed)
        real_python_seed(seed)

    def numpy_seed(seed: int) -> None:
        numpy_seeds.append(seed)
        real_numpy_seed(seed)

    def torch_seed(seed: int) -> torch.Generator:
        torch_seeds.append(seed)
        return real_torch_seed(seed)

    monkeypatch.setattr(random, "seed", python_seed)
    monkeypatch.setattr(np.random, "seed", numpy_seed)
    monkeypatch.setattr(torch, "manual_seed", torch_seed)

    _run(owner, data)

    assert python_seeds == [17]
    assert numpy_seeds == [17]
    assert torch_seeds == [17]


@pytest.mark.parametrize("with_trace", [False, True])
def test_run_adaptive_profile_records_historical_and_terminal_evaluations(
    monkeypatch: pytest.MonkeyPatch,
    with_trace: bool,
) -> None:
    monkeypatch.delenv("MODSSC_CHECKPOINT_ROOT", raising=False)
    owner = _owner(profile=_FLEX_PROFILE)
    trace_getter = (lambda: {"threshold": 0.75}) if with_trace else None
    result = _run(
        owner,
        _data(evaluation=True),
        method_id="flexmatch",
        trace_getter=trace_getter,
    )
    assert len(result.evaluation_history) == 2
    historical, terminal = result.evaluation_history
    assert historical["historical_eligible"] is True
    assert terminal["historical_eligible"] is False
    assert terminal["terminal_evaluation"] is True
    assert ("method_state" in historical) is with_trace
    assert ("method_state" in terminal) is with_trace
    assert result.paper_metrics["selection_uses_test"] is True


def test_run_image_profile_constructs_reference_augmenter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[str, int]] = []

    class FakeAugmenter:
        def __init__(self, *, profile: str, seed: int) -> None:
            created.append((profile, seed))

        def apply_batch(self, batch: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
            return batch

    import modssc.data_augmentation.cifar_reference as cifar_reference

    monkeypatch.setattr(cifar_reference, "CifarReferenceAugmentation", FakeAugmenter)
    monkeypatch.delenv("MODSSC_CHECKPOINT_ROOT", raising=False)
    n_l = 3
    n_u = 4
    data = InductiveDataset(
        X_l=torch.zeros((n_l, 3, 2, 2), dtype=torch.float64),
        y_l=torch.tensor([0, 1, 0], dtype=torch.int64),
        X_u=torch.ones((n_u, 3, 2, 2), dtype=torch.float64),
        X_u_w=torch.ones((n_u, 3, 2, 2), dtype=torch.float64),
        X_u_s=torch.ones((n_u, 3, 2, 2), dtype=torch.float64),
        meta={
            "augmentation_seed": 23,
            "source_idx_l": np.arange(n_l),
            "source_idx_u": np.arange(n_u),
        },
    )
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(12, 2, dtype=torch.float64),
    )
    bundle = TorchModelBundle(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
    )
    result = _run(_owner(bundle=bundle), data)
    assert result.optimization_steps == 1
    assert created == [("google_fixmatch_ra", 23)]


def test_forced_continuation_run_saves_before_interrupting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "forced"
    monkeypatch.setenv("MODSSC_CHECKPOINT_ROOT", str(checkpoint_root))
    monkeypatch.setenv("MODSSC_CHECKPOINT_RESUME", "0")
    monkeypatch.setenv("MODSSC_FORCE_CONTINUATION_STEP", "1")
    monkeypatch.setenv("MODSSC_CONTINUATION_REQUESTED", "0")
    with pytest.raises(PlannedContinuation):
        _run(_owner(max_steps=2), _data())
    pointer = json.loads((checkpoint_root / "CURRENT.json").read_text(encoding="utf-8"))
    assert pointer["step"] == 1


@pytest.mark.parametrize("existing_cublas", [None, ":16:8"])
def test_paper_runtime_restores_deterministic_state_after_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_cublas: str | None,
) -> None:
    state = {"enabled": False, "warn_only": True}
    calls: list[tuple[bool, bool]] = []
    precision_targets = (
        torch.backends,
        torch.backends.cuda.matmul,
        torch.backends.cudnn,
        torch.backends.cudnn.conv,
    )
    precision_state = tuple(owner.fp32_precision for owner in precision_targets)

    def use_deterministic_algorithms(
        enabled: bool,
        *,
        warn_only: bool = False,
    ) -> None:
        state["enabled"] = bool(enabled)
        state["warn_only"] = bool(warn_only)
        calls.append((bool(enabled), bool(warn_only)))

    monkeypatch.setattr(
        torch,
        "are_deterministic_algorithms_enabled",
        lambda: state["enabled"],
    )
    monkeypatch.setattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        lambda: state["warn_only"],
    )
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        use_deterministic_algorithms,
    )
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    if existing_cublas is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", existing_cublas)
    checkpoint_root = tmp_path / "deterministic-runtime"
    monkeypatch.setenv("MODSSC_CHECKPOINT_ROOT", str(checkpoint_root))
    monkeypatch.setenv("MODSSC_CHECKPOINT_RESUME", "0")
    monkeypatch.setenv("MODSSC_FORCE_CONTINUATION_STEP", "1")
    monkeypatch.setenv("MODSSC_CONTINUATION_REQUESTED", "0")

    def assert_runtime(
        logits_l: torch.Tensor,
        logits_uw: torch.Tensor,
        logits_us: torch.Tensor,
        y_l: torch.Tensor,
        idx_u: torch.Tensor,
    ) -> MatchStepResult:
        assert state == {"enabled": True, "warn_only": False}
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert all(owner.fp32_precision == "ieee" for owner in precision_targets)
        return _valid_step(logits_l, logits_uw, logits_us, y_l, idx_u)

    with pytest.raises(PlannedContinuation):
        _run(
            _owner(max_steps=2),
            _data(),
            step_fn=assert_runtime,
        )

    assert calls == [(True, False), (False, True)]
    assert state == {"enabled": False, "warn_only": True}
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == existing_cublas
    assert tuple(owner.fp32_precision for owner in precision_targets) == precision_state


@pytest.mark.parametrize(
    ("history", "next_step", "message"),
    [
        ("invalid", 0, "evaluation history is invalid"),
        ([], -1, "step is outside"),
        ([], 2, "step is outside"),
    ],
)
def test_run_paper_match_rejects_invalid_resumed_progress(
    monkeypatch: pytest.MonkeyPatch,
    history: object,
    next_step: int,
    message: str,
) -> None:
    monkeypatch.delenv("MODSSC_CHECKPOINT_ROOT", raising=False)
    owner = _owner()
    sampler = FixedSSLBatchSampler(
        3,
        4,
        labeled_batch_size=64,
        unlabeled_batch_size=448,
        seed=17,
        mode="shuffle_repeat",
        shuffle_buffer=8192,
    )
    payload = {
        "bundle": match_trainer._bundle_state(owner.spec.model_bundle),
        "sampler": sampler.state_dict(),
        "method_state": {},
        "rng": match_trainer._rng_state(torch),
        "next_step": next_step,
        "evaluation_history": history,
    }
    monkeypatch.setattr(match_trainer._CheckpointStore, "load", lambda _self: payload)
    with pytest.raises(InductiveValidationError, match=message):
        _run(owner, _data())
