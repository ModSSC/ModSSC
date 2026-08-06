from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import bench.main as bench_main
from bench import reproduce
from bench.orchestrators import reporting as report_orch
from bench.schema import ExperimentConfig
from bench.utils.io import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DCL_LOCK_DIR = REPO_ROOT / "bench/campaigns/locks/dcl-vote-zhou-goldman-2004-v1"
DCL_CARD_PATH = REPO_ROOT / "bench/configs/reproductions/democratic_co_learning/vote.yaml"


def _dcl_card() -> reproduce.ReproductionCard:
    return next(
        card for card in reproduce.discover_cards() if card.config_path == DCL_CARD_PATH.resolve()
    )


def test_packaged_dcl_vote_lock_authenticates_all_twenty_replays() -> None:
    raw = load_yaml(DCL_CARD_PATH)

    lock, evidence = reproduce._authenticate_dcl_vote_replays(raw)

    assert tuple(evidence) == tuple(range(1, 21))
    assert tuple(entry.seed for entry in lock.selected) == tuple(range(1, 21))
    for seed, replay in evidence.items():
        assert Path(replay["selection_path"]) == DCL_LOCK_DIR / "selected-partitions.json"
        assert Path(replay["replay_path"]) == DCL_LOCK_DIR / "splits" / f"seed-{seed:03d}"
        assert replay["selection_rank"] == seed
        assert replay["split_fingerprint"] == lock.by_seed()[seed].split_fingerprint


def test_static_verification_authenticates_packaged_dcl_vote_replays() -> None:
    report = reproduce.verify_card(_dcl_card())

    assert report.execution_ready is True
    assert not [issue for issue in report.issues if issue.code == "E_REPRO_DCL_PARTITIONS"]


def test_dcl_vote_dry_run_reports_authenticated_replays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(reproduce, "_require_static_verification", lambda _card: None)
    result = reproduce.prepare_card(_dcl_card(), cache_dir=tmp_path, dry_run=True)

    assert result.dry_run is True
    assert "dcl-vote-partitions:20/20-authenticated" in result.protocol_checks


def test_dcl_vote_run_card_injects_the_frozen_replay_for_selected_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    card = _dcl_card()
    effective_runs: list[dict[str, object]] = []
    monkeypatch.setattr(reproduce, "_require_static_verification", lambda _card: None)
    monkeypatch.setattr(reproduce, "prepare_card", lambda *_args, **_kwargs: None)

    def fake_run_experiment_single(
        config_path: Path,
        *,
        raw: dict[str, object],
        cfg: ExperimentConfig,
    ) -> SimpleNamespace:
        assert config_path == DCL_CARD_PATH.resolve()
        assert cfg.run.seed == 3
        effective_runs.append(raw)
        return SimpleNamespace(code=0, run_json_path=tmp_path / "run.json")

    monkeypatch.setattr(bench_main, "run_experiment_single", fake_run_experiment_single)

    assert reproduce.run_card(card, cache_dir=tmp_path / "datasets", seed=3) == 0
    assert len(effective_runs) == 1
    replay = effective_runs[0]["sampling"]["replay"]  # type: ignore[index]
    assert replay["selection_rank"] == 3
    assert Path(replay["replay_path"]) == DCL_LOCK_DIR / "splits/seed-003"


def test_dcl_vote_num_runs_uses_the_locked_seed_prefix_and_writes_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    card = _dcl_card()
    seen_seeds: list[int] = []
    summaries: list[dict[str, object]] = []
    monkeypatch.setattr(reproduce, "_require_static_verification", lambda _card: None)
    monkeypatch.setattr(reproduce, "prepare_card", lambda *_args, **_kwargs: None)

    def fake_run_experiment_single(
        _config_path: Path,
        *,
        raw: dict[str, object],
        cfg: ExperimentConfig,
    ) -> SimpleNamespace:
        seed = int(cfg.run.seed)
        seen_seeds.append(seed)
        replay = raw["sampling"]["replay"]  # type: ignore[index]
        assert replay["selection_rank"] == seed
        return SimpleNamespace(code=0, run_json_path=tmp_path / f"run-{seed}.json")

    monkeypatch.setattr(bench_main, "run_experiment_single", fake_run_experiment_single)
    monkeypatch.setattr(
        report_orch,
        "write_seed_sweep_summary",
        lambda **kwargs: summaries.append(kwargs) or tmp_path / "summary.json",
    )

    assert reproduce.run_card(card, cache_dir=tmp_path / "datasets", num_runs=2) == 0
    assert seen_seeds == [1, 2]
    assert len(summaries) == 1
    assert summaries[0]["requested_seeds"] == [1, 2]


@pytest.mark.parametrize(
    ("seed", "num_runs", "message"),
    [
        (0, None, "frozen DCL Vote seeds"),
        (None, 21, "cannot exceed"),
        (1, 1, "mutually exclusive"),
    ],
)
def test_dcl_vote_rejects_unregistered_run_overrides(
    seed: int | None,
    num_runs: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reproduce._select_dcl_vote_seeds(
            load_yaml(DCL_CARD_PATH),
            seed=seed,
            num_runs=num_runs,
        )


def test_dcl_vote_run_card_rejects_unregistered_seed_before_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reproduce,
        "prepare_card",
        lambda *_args, **_kwargs: pytest.fail("invalid seed prepared the dataset"),
    )

    with pytest.raises(ValueError, match="frozen DCL Vote seeds"):
        reproduce.run_card(_dcl_card(), seed=21)


def test_dcl_vote_authentication_fails_closed_on_tampered_packaged_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    copied_root = tmp_path / "checkout"
    copied_lock = copied_root / reproduce._DCL_VOTE_LOCK_RELATIVE.parent
    shutil.copytree(DCL_LOCK_DIR, copied_lock)
    (copied_lock / "splits/seed-020/arrays.npz").write_bytes(b"tampered")
    monkeypatch.setattr(reproduce, "REPO_ROOT", copied_root)

    with pytest.raises(
        reproduce.ReproductionRegistryError,
        match="frozen-partition authentication failed",
    ):
        reproduce._authenticate_dcl_vote_replays(load_yaml(DCL_CARD_PATH))
