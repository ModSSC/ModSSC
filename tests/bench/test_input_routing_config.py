from __future__ import annotations

from pathlib import Path

import yaml

from bench.schema import ExperimentConfig

_ROOT = Path(__file__).resolve().parents[2]


def _minimal_config() -> dict[str, object]:
    return {
        "run": {"name": "routing", "seed": 1, "output_dir": "runs"},
        "dataset": {"id": "toy"},
        "sampling": {
            "seed": 1,
            "plan": {"split": {"kind": "holdout"}},
        },
        "preprocess": {"seed": 1, "cache": False, "plan": {"steps": []}},
        "method": {
            "kind": "inductive",
            "id": "pseudo_label",
            "device": {"device": "cpu", "dtype": "float32"},
        },
        "evaluation": {"report_splits": ["test"], "metrics": ["accuracy"]},
    }


def test_sampling_routing_policy_defaults_to_reject() -> None:
    config = ExperimentConfig.from_dict(_minimal_config())

    assert config.sampling.inductive_graph_policy == "reject"


def test_sampling_routing_policy_is_passed_through_for_native_validation() -> None:
    raw = _minimal_config()
    raw["sampling"]["inductive_graph_policy"] = "masks_to_indices"  # type: ignore[index]

    config = ExperimentConfig.from_dict(raw)

    assert config.sampling.inductive_graph_policy == "masks_to_indices"


def test_all_inductive_graph_cards_declare_mask_conversion_policy() -> None:
    cards = sorted((_ROOT / "bench" / "configs").glob("**/inductive/*/graph/*.yaml"))
    assert cards

    missing: list[str] = []
    for card in cards:
        raw = yaml.safe_load(card.read_text(encoding="utf-8"))
        if raw["sampling"].get("inductive_graph_policy") != "masks_to_indices":
            missing.append(str(card.relative_to(_ROOT)))

    assert missing == []
