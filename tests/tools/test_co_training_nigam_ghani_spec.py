from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bench.campaign import generate as generate_module
from bench.campaign.generate import generate_campaign
from bench.campaign.manifest import load_manifest
from bench.campaign.scientific_gates import evaluate_gate, load_gate_registry
from bench.schema import ExperimentConfig
from bench.utils.io import load_yaml
from modssc.inductive.methods.co_training import CoTrainingSpec, _validate_protocol
from modssc.sampling.plan import SamplingPlan

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = (
    REPO_ROOT / "tools/hpc/specs/historical-paper-co-training-nigam-ghani2000-v2.example.yaml"
)
CONFIG_PATH = (
    REPO_ROOT / "bench/configs/reproductions/co_training/webkb_course_nigam_ghani_2000.yaml"
)
GATE_PATH = REPO_ROOT / "bench/campaigns/scientific-gates.yaml"


def _clean_placeholder_runtime(**_kwargs: object) -> dict[str, object]:
    return {
        "git_sha": "REPLACE_WITH_CLEAN_COMMIT",
        "git_dirty": False,
        "git_diff_sha256": "0" * 64,
    }


def test_nigam_ghani_card_freezes_one_profile_and_exact_sampling() -> None:
    raw = load_yaml(CONFIG_PATH)
    config = ExperimentConfig.from_dict(raw)
    sampling = SamplingPlan.from_dict(config.sampling.plan)

    assert config.run.seed == 21
    assert config.run.seeds == list(range(21, 31))
    assert config.dataset.id == "webkb_course_cotraining"
    assert config.method.profile == "paper:nigam-ghani2000-webkb-table2"
    assert config.method.params["protocol"] == "shared_pool_exhaustive_multiset"
    assert sampling.labeling.class_counts == {"0": 9, "1": 3}
    assert sampling.split.as_dict() == {
        "kind": "holdout",
        "test_fraction": 263 / 1051,
        "val_fraction": 0.0,
        "stratify": False,
        "shuffle": True,
    }
    _validate_protocol(CoTrainingSpec(**config.method.params))


def test_nigam_ghani_campaign_is_one_unprotected_local_cpu_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    assert raw["campaign_id"] == "historical-paper-co-training-nigam-ghani2000-v2"
    assert raw["expect"] == {
        "config_count": 1,
        "task_count": 10,
        "tasks_per_method": {"co_training": 10},
        "tasks_by_profile": {"cpu_text": 10},
        "tasks_by_site": {"local-cpu": 10},
    }
    assert len(raw["cells"]) == 1

    monkeypatch.setattr(generate_module, "collect_runtime_versions", _clean_placeholder_runtime)
    output = tmp_path / "campaign"
    generated = generate_campaign(
        SPEC_PATH,
        repo_root=REPO_ROOT,
        output_dir=output,
        _allow_template_placeholders=True,
    )
    meta, tasks = load_manifest(Path(generated.manifest_path))

    assert generated.task_count == 10
    assert meta["counts_by_method"] == {"co_training": 10}
    assert {task.seed for task in tasks} == set(range(21, 31))
    assert {task.method_profile for task in tasks} == {"paper:nigam-ghani2000-webkb-table2"}
    assert {task.resource_profile for task in tasks} == {"cpu_text"}
    assert {task.assigned_site for task in tasks} == {"local-cpu"}
    assert not (output / "submit").exists()
    assert not (output / "profiles/resources.json").exists()

    decision = evaluate_gate(
        load_gate_registry(GATE_PATH),
        campaign_id=generated.campaign_id,
        track="paper",
        method_id="co_training",
        claim_scope_id=tasks[0].claim_scope_id,
        campaign_stage=tasks[0].campaign_stage,
        claim_eligible=tasks[0].claim_eligible,
    )
    assert decision.allowed is True
    assert decision.blockers == ()
