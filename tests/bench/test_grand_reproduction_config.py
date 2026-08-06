from __future__ import annotations

from pathlib import Path

import yaml

from bench.schema import ExperimentConfig

REPRODUCTION_PATH = (
    Path(__file__).resolve().parents[2]
    / "bench"
    / "configs"
    / "reproductions"
    / "grand"
    / "cora.yaml"
)
ACCEPTANCE_PATH = (
    Path(__file__).resolve().parents[2] / "bench" / "campaigns" / "article10-paper-acceptance.yaml"
)


def test_grand_cora_card_pins_the_public_planetoid_protocol() -> None:
    raw = yaml.safe_load(REPRODUCTION_PATH.read_text(encoding="utf-8"))
    cfg = ExperimentConfig.from_dict(raw)

    assert cfg.run.seeds == list(range(100))
    assert cfg.dataset.id == "cora"
    assert cfg.dataset.options == {"split": "public"}
    assert cfg.dataset.download is False
    assert cfg.sampling.plan["policy"] == {
        "respect_official_test": True,
        "use_official_graph_masks": True,
        "allow_override_official": False,
    }
    assert cfg.method.method_id == "grand"
    assert cfg.method.profile == "paper:feng2020-cora-table1"
    assert cfg.method.params == {
        "training_mode": "random_propagation_consistency",
        "hidden_dim": 32,
        "mlp_dropout": 0.5,
        "input_dropout": 0.5,
        "hidden_dropout": 0.5,
        "use_batch_norm": False,
        "prop_steps": 8,
        "dropnode": 0.5,
        "num_samples": 4,
        "temperature": 0.5,
        "lambda_consistency": 1.0,
        "consistency_rampup_epochs": 0,
        "lr": 0.01,
        "weight_decay": 0.0005,
        "max_epochs": 5000,
        "patience": 200,
        "add_self_loops": True,
    }


def test_grand_acceptance_uses_the_official_population_standard_deviation() -> None:
    acceptance = yaml.safe_load(ACCEPTANCE_PATH.read_text(encoding="utf-8"))
    card = acceptance["protocols"]["feng-2020-cora-table1-planetoid"]

    assert card["repetitions"] == 100
    assert card["target"] == {
        "split": "test",
        "metric": "accuracy",
        "transform": "identity",
        "published_mean": 0.854,
        "published_std": 0.004,
        "published_std_ddof": 0,
        "margin_absolute": 0.01,
    }
    assert card["known_deviations"] == []
    assert card["critical_unknowns"] == []
    assert card["environment_differences"] == [
        "The pinned paper used PyTorch 1.2.0 on RTX 2080 Ti; "
        "runtime hardware is recorded separately from this scientific protocol."
    ]
