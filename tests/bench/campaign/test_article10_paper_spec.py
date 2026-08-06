from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from bench.campaign import generate as generate_module
from bench.campaign.generate import generate_campaign
from bench.campaign.manifest import load_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO_ROOT / "tools" / "hpc" / "specs" / "article10-paper.example.yaml"

EXPECTED_COUNTS = {
    "democratic_co_learning": 20,
    "fixmatch": 5,
    "flexmatch": 3,
    "free_match": 3,
    "grand": 100,
    "laplace_learning": 500,
    "poisson_learning": 500,
    "pseudo_label": 10,
    "softmatch": 3,
    "tri_training": 3,
}

EXPECTED_SINGLE_PROFILES = {
    "democratic_co_learning": "paper:zhou-goldman-2004-adult-table3",
    "fixmatch": "paper:sohn2020-cifar10-table2-250",
    "flexmatch": "paper:zhang2021-cifar10-table1-250",
    "free_match": "paper:wang2023-cifar10-table1-40",
    "grand": "paper:feng2020-cora-table1",
    "pseudo_label": "paper:lee2013-mnist-table2-600",
    "softmatch": "paper:chen2023-cifar10-table2-250",
    "tri_training": "paper:zhou-li-2005-wdbc-table3-j48",
}


def test_article10_paper_manifest_is_complete_pinned_and_honest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        lambda **kwargs: {
            "git_sha": "REPLACE_WITH_CLEAN_COMMIT",
            "git_dirty": False,
            "git_diff_sha256": "0" * 64,
        },
    )

    generated = generate_campaign(
        SPEC_PATH,
        repo_root=REPO_ROOT,
        output_dir=tmp_path / "article10-paper",
        _allow_template_placeholders=True,
    )
    meta, tasks = load_manifest(Path(generated.manifest_path))

    assert generated.task_count == 1147
    assert len({task.config_path for task in tasks}) == 18
    assert meta["counts_by_method"] == EXPECTED_COUNTS
    assert meta["counts_by_profile"] == {
        "a100_gpu": 14,
        "cpu_graph": 1000,
        "cpu_tabular": 33,
        "v100_gpu": 100,
    }
    assert meta["counts_by_site"] == {"slurm-gpu": 114, "regional": 1033}

    profiles_by_method: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        profiles_by_method[task.method_id].add(task.method_profile)
    for method_id, profile in EXPECTED_SINGLE_PROFILES.items():
        assert profiles_by_method[method_id] == {profile}
    assert {task.dataset_id for task in tasks if task.method_id == "tri_training"} == {"wdbc"}
    for method_id in ("laplace_learning", "poisson_learning"):
        assert profiles_by_method[method_id] == {
            f"paper:calder2020-mnist-table1-{method_id.removesuffix('_learning')}-{budget}-label-per-class"
            for budget in range(1, 6)
        }

    assert all(task.method_profile.startswith("paper:") for task in tasks)
    assert all(
        task.expected_dataset_fingerprint is not None
        and task.expected_dataset_fingerprint.startswith("REPLACE_WITH_")
        for task in tasks
    )
    assert all(
        task.expected_split_fingerprint is not None and len(task.expected_split_fingerprint) == 64
        for task in tasks
    )
    assert len({(task.config_path, task.seed) for task in tasks}) == len(tasks)
    assert all(
        task.required_seed_count == EXPECTED_COUNTS[task.method_id]
        for task in tasks
        if task.method_id not in {"laplace_learning", "poisson_learning"}
    )
    assert {
        task.required_seed_count
        for task in tasks
        if task.method_id in {"laplace_learning", "poisson_learning"}
    } == {100}
    grand_tasks = [task for task in tasks if task.method_id == "grand"]
    assert len(grand_tasks) == 100
    assert len({task.expected_split_fingerprint for task in grand_tasks}) == 1
    assert {task.model_seed for task in grand_tasks} == set(range(100))
    assert all(task.model_seed == task.seed for task in grand_tasks)
    assert {task.fidelity_status for task in grand_tasks} == {"paper_matched"}
    assert {task.fidelity_status for task in tasks} == {
        "paper_matched",
        "paper_approx",
        "not_claimable",
    }
    assert {task.method_id for task in tasks if task.fidelity_status == "not_claimable"} == {
        "democratic_co_learning"
    }
