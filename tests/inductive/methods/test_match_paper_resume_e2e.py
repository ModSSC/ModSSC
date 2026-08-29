from __future__ import annotations

import copy
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.helpers import match_trainer
from modssc.inductive.methods.helpers.match_trainer import MatchStepResult
from modssc.inductive.seed import seed_everything
from modssc.inductive.types import DeviceSpec, InductiveDataset
from modssc.runtime.continuation import PlannedContinuation
from modssc.runtime.execution import ExecutionContext, RunIdentity


def _assert_state_equal(actual: Any, expected: Any, *, path: str = "state") -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0, msg=path)
        return
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray), path
        np.testing.assert_array_equal(actual, expected, err_msg=path)
        return
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping), path
        assert tuple(actual) == tuple(expected), path
        for key in expected:
            _assert_state_equal(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        assert isinstance(actual, Sequence) and not isinstance(actual, (str, bytes)), path
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_state_equal(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
        return
    assert type(actual) is type(expected), path
    assert actual == expected, path


def _synthetic_data(
    execution_context: ExecutionContext | None = None,
) -> InductiveDataset:
    x_l = torch.linspace(-1.0, 1.0, 17 * 3, dtype=torch.float64).reshape(17, 3)
    y_l = torch.arange(17, dtype=torch.int64).remainder(2)
    x_u = torch.linspace(-0.8, 0.9, 29 * 3, dtype=torch.float64).reshape(29, 3)
    x_test = torch.tensor(
        [
            [-0.9, -0.4, 0.2],
            [-0.1, 0.5, 0.8],
            [0.7, -0.3, 0.4],
            [0.2, 0.1, -0.6],
        ],
        dtype=torch.float64,
    )
    return InductiveDataset(
        X_l=x_l,
        y_l=y_l,
        X_u=x_u,
        X_u_w=x_u + 0.015,
        X_u_s=x_u * 0.91 - 0.025,
        meta={
            "dataset_fingerprint": "synthetic-match-e2e",
            "split_fingerprint": "synthetic-split-e2e",
            "partition_sha256": "synthetic-partition-e2e",
            "evaluation_splits": {
                "test": {
                    "X": x_test,
                    "y": torch.tensor([0, 1, 0, 1], dtype=torch.int64),
                }
            },
        },
        execution_context=execution_context,
    )


def _bundle() -> TorchModelBundle:
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 7, dtype=torch.float64),
        torch.nn.Tanh(),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(7, 2, dtype=torch.float64),
    )
    ema_model = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    return TorchModelBundle(
        model=model,
        optimizer=optimizer,
        ema_model=ema_model,
        scheduler=scheduler,
        meta={
            "ema_decay": 0.8,
            "ema_strategy": "parameters_only_copy_buffers",
        },
    )


def _runner(*, seed: int) -> tuple[Any, dict[str, Any], Any]:
    bundle = _bundle()
    owner = SimpleNamespace(
        spec=SimpleNamespace(
            model_bundle=bundle,
            batch_size=64,
            mu=7,
            max_steps=5,
            training_mode="fixed_steps",
            reference_implementation="google_fixmatch",
            sampler_mode="shuffle_repeat",
            sampler_shuffle_buffer=8192,
            augmentation_profile="google_fixmatch_ra",
            interleave_bn=True,
            evaluation_interval_steps=1024,
            checkpoint_interval_steps=1024,
            reporting_policy="median_last_checkpoints",
            reporting_window_checkpoints=20,
            allow_short_run=True,
        ),
        diagnostics_=None,
    )
    method_state: dict[str, Any] = {
        "calls": 0,
        "random_trace": [],
        "logit_trace": [],
        "tensor_checksum": torch.zeros((), dtype=torch.float64),
    }

    def step_fn(
        logits_l: torch.Tensor,
        logits_uw: torch.Tensor,
        logits_us: torch.Tensor,
        y_l: torch.Tensor,
        _idx_u: torch.Tensor,
    ) -> MatchStepResult:
        python_draw = random.random()
        numpy_draw = float(np.random.random())
        torch_draw = torch.rand((), dtype=logits_l.dtype, device=logits_l.device)
        random_scale = python_draw + numpy_draw + float(torch_draw)
        loss = F.cross_entropy(logits_l, y_l)
        loss = loss + random_scale * 1e-3 * (logits_uw.square().mean() + logits_us.square().mean())
        with torch.no_grad():
            accepted = float((torch.softmax(logits_uw, dim=1).amax(dim=1) >= 0.5).sum().item())
            method_state["calls"] += 1
            method_state["random_trace"].append((python_draw, numpy_draw, float(torch_draw)))
            method_state["logit_trace"].append(
                (
                    float(logits_l.sum()),
                    float(logits_uw.sum()),
                    float(logits_us.sum()),
                )
            )
            method_state["tensor_checksum"] = (
                method_state["tensor_checksum"]
                + logits_uw.detach().sum().cpu()
                + logits_us.detach().sum().cpu()
            )
        return MatchStepResult(
            loss=loss,
            accepted=accepted,
            unlabeled=448,
            diagnostics={
                "method_calls": method_state["calls"],
                "random_scale": random_scale,
            },
        )

    def state_getter() -> Mapping[str, Any]:
        return {
            "calls": int(method_state["calls"]),
            "random_trace": tuple(method_state["random_trace"]),
            "logit_trace": tuple(method_state["logit_trace"]),
            "tensor_checksum": method_state["tensor_checksum"].clone(),
        }

    def state_loader(state: Mapping[str, Any]) -> None:
        method_state["calls"] = int(state["calls"])
        method_state["random_trace"] = list(state["random_trace"])
        method_state["logit_trace"] = list(state["logit_trace"])
        method_state["tensor_checksum"] = state["tensor_checksum"].clone()

    def trace_getter() -> Mapping[str, Any]:
        return {
            "calls": int(method_state["calls"]),
            "tensor_checksum": float(method_state["tensor_checksum"]),
        }

    def run(data: InductiveDataset) -> Any:
        return match_trainer.run_fixed_step_match(
            owner,
            data,
            device=DeviceSpec(device="cpu", dtype="float64"),
            seed=seed,
            method_id="fixmatch",
            step_fn=step_fn,
            state_getter=state_getter,
            state_loader=state_loader,
            trace_getter=trace_getter,
            _enforce_reference_contract=False,
        )

    return owner, method_state, run


def _checkpoint(context: ExecutionContext) -> tuple[dict[str, Any], dict[str, Any]]:
    root = context.checkpoint_dir
    pointer = json.loads((root / "CURRENT.json").read_text(encoding="utf-8"))
    generation = root / "generations" / pointer["generation"]
    metadata = json.loads((generation / "checkpoint.json").read_text(encoding="utf-8"))
    payload = torch.load(
        generation / "payload.bin",
        map_location="cpu",
        weights_only=False,
    )
    return payload, metadata


def test_full_match_contract_requires_exact_step_count() -> None:
    data = _synthetic_data()
    owner, _, run = _runner(seed=7)
    owner.spec.allow_short_run = False

    with pytest.raises(InductiveValidationError, match=r"exactly 2\^20"):
        run(data)

    owner.spec.allow_short_run = True
    owner.spec.max_steps = (1 << 20) + 1
    with pytest.raises(InductiveValidationError, match=r"inside \(0, 2\^20\]"):
        run(data)


def test_reference_contract_bypass_is_restricted_to_short_runs() -> None:
    data = _synthetic_data()
    owner, _, run = _runner(seed=7)
    owner.spec.allow_short_run = False
    owner.spec.max_steps = 1 << 20

    with pytest.raises(InductiveValidationError, match="explicit short run"):
        run(data)


def test_paper_match_interrupted_resume_is_bit_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODSSC_CONTINUATION_REQUESTED", "0")

    continuous_context = ExecutionContext(
        RunIdentity("a" * 64, 2718),
        tmp_path / "continuous-run",
        resume_policy="auto",
        checkpoint_root=tmp_path / "continuous",
    )
    continuous_data = _synthetic_data(continuous_context)
    seed_everything(314159, deterministic=True)
    _, continuous_method_state, continuous_run = _runner(seed=2718)
    monkeypatch.setattr(match_trainer, "_continuation_requested", lambda: False)
    continuous_result = continuous_run(continuous_data)
    continuous_checkpoint, continuous_metadata = _checkpoint(continuous_context)

    resumed_context = ExecutionContext(
        RunIdentity("a" * 64, 2718),
        tmp_path / "resumed-run",
        resume_policy="auto",
        checkpoint_root=tmp_path / "resumed",
    )
    resumed_data = _synthetic_data(resumed_context)
    seed_everything(314159, deterministic=True)
    _, interrupted_method_state, interrupted_run = _runner(seed=2718)
    continuation_checks = 0

    def interrupt_between_periodic_checkpoints() -> bool:
        nonlocal continuation_checks
        continuation_checks += 1
        return continuation_checks == 3

    monkeypatch.setattr(
        match_trainer,
        "_continuation_requested",
        interrupt_between_periodic_checkpoints,
    )
    with pytest.raises(PlannedContinuation):
        interrupted_run(resumed_data)
    interrupted_checkpoint, interrupted_metadata = _checkpoint(resumed_context)
    assert interrupted_checkpoint["next_step"] == 3
    assert interrupted_metadata["reason"] == "planned_continuation"
    assert interrupted_method_state["calls"] == 3

    # Deliberately perturb all global RNGs and create a fresh bundle/method.
    # A correct resume must ignore these values and restore the checkpointed state.
    seed_everything(8675309, deterministic=True)
    _, resumed_method_state, resumed_run = _runner(seed=2718)
    monkeypatch.setattr(match_trainer, "_continuation_requested", lambda: False)
    resumed_result = resumed_run(resumed_data)
    resumed_checkpoint, resumed_metadata = _checkpoint(resumed_context)

    assert resumed_result.resumed_from_step == 3
    assert continuous_result.resumed_from_step == 0
    assert resumed_result.optimization_steps == continuous_result.optimization_steps == 5
    assert resumed_result.target_steps == continuous_result.target_steps == 5
    assert resumed_result.accepted == continuous_result.accepted
    assert resumed_result.unlabeled == continuous_result.unlabeled
    _assert_state_equal(
        resumed_result.evaluation_history,
        continuous_result.evaluation_history,
        path="evaluation_history",
    )
    _assert_state_equal(
        resumed_result.paper_metrics,
        continuous_result.paper_metrics,
        path="paper_metrics",
    )

    for state_name in (
        "model",
        "optimizer",
        "scheduler",
        "ema_model",
    ):
        assert continuous_checkpoint["bundle"][state_name] is not None
        _assert_state_equal(
            resumed_checkpoint["bundle"][state_name],
            continuous_checkpoint["bundle"][state_name],
            path=f"bundle.{state_name}",
        )
    for state_name in (
        "sampler",
        "rng",
        "method_state",
        "evaluation_history",
    ):
        _assert_state_equal(
            resumed_checkpoint[state_name],
            continuous_checkpoint[state_name],
            path=state_name,
        )
    _assert_state_equal(
        resumed_method_state,
        continuous_method_state,
        path="live_method_state",
    )

    assert continuous_metadata["reason"] == resumed_metadata["reason"] == "complete"
    assert [
        *(event["reason"] for event in continuous_checkpoint["checkpoint_history"]),
        continuous_metadata["reason"],
    ] == ["periodic", "periodic", "complete"]
    assert [
        *(event["reason"] for event in resumed_checkpoint["checkpoint_history"]),
        resumed_metadata["reason"],
    ] == ["periodic", "planned_continuation", "periodic", "complete"]
