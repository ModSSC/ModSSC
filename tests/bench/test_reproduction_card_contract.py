from __future__ import annotations

import tomllib
from pathlib import Path

from bench.schema import ExperimentConfig
from bench.utils.io import load_yaml
from modssc.data_loader import dataset_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[2]
CARDS_ROOT = REPO_ROOT / "bench" / "configs" / "reproductions"

EXPECTED_ACCEPTANCE_BY_CARD = {
    "democratic_co_learning/adult.yaml": "zhou-goldman-2004-adult-table3",
    "democratic_co_learning/vote.yaml": "zhou-goldman-2004-vote-table3",
    "fixmatch/cifar10-250.yaml": "sohn-2020-cifar10-table2-250",
    "flexmatch/cifar10-250.yaml": "zhang-2021-cifar10-table1-250",
    "free_match/cifar10-40.yaml": "wang-2023-cifar10-table1-40",
    "grand/cora.yaml": "feng-2020-cora-table1-planetoid",
    "laplace_learning/mnist-table1-1-label-per-class.yaml": (
        "calder-2020-mnist-table1-laplace-1-label-per-class"
    ),
    "laplace_learning/mnist-table1-2-label-per-class.yaml": (
        "calder-2020-mnist-table1-laplace-2-label-per-class"
    ),
    "laplace_learning/mnist-table1-3-label-per-class.yaml": (
        "calder-2020-mnist-table1-laplace-3-label-per-class"
    ),
    "laplace_learning/mnist-table1-4-label-per-class.yaml": (
        "calder-2020-mnist-table1-laplace-4-label-per-class"
    ),
    "laplace_learning/mnist-table1-5-label-per-class.yaml": (
        "calder-2020-mnist-table1-laplace-5-label-per-class"
    ),
    "poisson_learning/mnist-table1-1-label-per-class.yaml": (
        "calder-2020-mnist-table1-poisson-1-label-per-class"
    ),
    "poisson_learning/mnist-table1-2-label-per-class.yaml": (
        "calder-2020-mnist-table1-poisson-2-label-per-class"
    ),
    "poisson_learning/mnist-table1-3-label-per-class.yaml": (
        "calder-2020-mnist-table1-poisson-3-label-per-class"
    ),
    "poisson_learning/mnist-table1-4-label-per-class.yaml": (
        "calder-2020-mnist-table1-poisson-4-label-per-class"
    ),
    "poisson_learning/mnist-table1-5-label-per-class.yaml": (
        "calder-2020-mnist-table1-poisson-5-label-per-class"
    ),
    "pseudo_label/mnist.yaml": "lee-2013-mnist-table2-600",
    "softmatch/cifar10-250.yaml": "chen-2023-cifar10-table2-250",
    "tri_training/vote_table3_j48.yaml": ("zhou-li-2005-vote-table3-j48-80pct-unlabeled"),
    "tri_training/wdbc_table3_j48.yaml": ("zhou-li-2005-wdbc-table3-j48-80pct-unlabeled"),
}

PAPER_MATCHED_PROTOCOLS = {
    "sohn-2020-cifar10-table2-250",
    "zhang-2021-cifar10-table1-250",
    "wang-2023-cifar10-table1-40",
    "chen-2023-cifar10-table2-250",
    "feng-2020-cora-table1-planetoid",
}

REMOVED_EVIDENCE_PREFIXES = (
    "bench/assets/",
    "bench/campaign/",
    "bench/campaigns/",
    "provenance/",
    "tools/",
)


def test_all_reproduction_cards_are_portable_declarative_runner_inputs() -> None:
    cards = sorted(CARDS_ROOT.glob("*/*.yaml"))
    assert len(cards) == 20

    observed_acceptance: dict[str, str] = {}
    observed_methods: set[str] = set()
    requested_runs = 0

    for card in cards:
        raw = load_yaml(card)
        config = ExperimentConfig.from_dict(raw)
        observed_methods.add(config.method.method_id)
        requested_runs += len(config.run.seeds or ())
        assert config.dataset.integrity is not None, f"{card}: dataset identity is not pinned"
        assert (
            dataset_fingerprint(
                config.dataset.id,
                options=config.dataset.options,
            )
            == config.dataset.integrity.fingerprint
        ), f"{card}: dataset fingerprint is stale"

        relative = card.relative_to(CARDS_ROOT).as_posix()
        acceptance = config.acceptance
        if acceptance is not None:
            observed_acceptance[relative] = acceptance.protocol_id
            assert config.run.benchmark_mode is True, card
            assert acceptance.method_id == config.method.method_id, card
            assert acceptance.repetitions == len(config.run.seeds or ()), card
            assert acceptance.protocol_id == EXPECTED_ACCEPTANCE_BY_CARD[relative], card
            if acceptance.protocol_id in PAPER_MATCHED_PROTOCOLS:
                expected_fidelity = "paper_matched"
            elif acceptance.protocol_id == "zhou-goldman-2004-adult-table3":
                expected_fidelity = "not_claimable"
            else:
                expected_fidelity = "paper_approx"
            assert acceptance.fidelity_ceiling == expected_fidelity, card
            assert acceptance.conformity.status == "passed", card
            assert len(acceptance.conformity.evidence) == 1, card
            for evidence in acceptance.conformity.evidence:
                assert evidence.startswith("docs/replications/protocols/"), (card, evidence)
                assert not evidence.startswith(REMOVED_EVIDENCE_PREFIXES), (card, evidence)
                assert (REPO_ROOT / evidence).is_file(), (card, evidence)

        plan = config.sampling.plan
        partition = plan.get("partition")
        if isinstance(partition, dict):
            assert "ordered_indices_artifact" not in partition, card

        labeling = plan.get("labeling")
        if isinstance(labeling, dict):
            assert "fixed_indices_artifact" not in labeling, card

        if config.graph is not None and config.graph.enabled:
            assert "precomputed_path" not in config.graph.spec, card
            assert "precomputed_sha256" not in config.graph.spec, card

    assert len(observed_acceptance) == 20
    assert observed_acceptance == EXPECTED_ACCEPTANCE_BY_CARD
    assert observed_methods == {
        "democratic_co_learning",
        "fixmatch",
        "flexmatch",
        "free_match",
        "grand",
        "laplace_learning",
        "poisson_learning",
        "pseudo_label",
        "softmatch",
        "tri_training",
    }
    assert requested_runs == 1170
    assert {
        protocol_id
        for relative, protocol_id in observed_acceptance.items()
        if ExperimentConfig.from_dict(load_yaml(CARDS_ROOT / relative)).acceptance.fidelity_ceiling
        == "paper_matched"
    } == PAPER_MATCHED_PROTOCOLS

    adult = ExperimentConfig.from_dict(
        load_yaml(CARDS_ROOT / "democratic_co_learning" / "adult.yaml")
    ).acceptance
    vote = ExperimentConfig.from_dict(
        load_yaml(CARDS_ROOT / "democratic_co_learning" / "vote.yaml")
    ).acceptance
    assert adult is not None and adult.fidelity_ceiling == "not_claimable"
    assert adult.unknowns
    assert vote is not None and vote.fidelity_ceiling == "paper_approx"
    assert vote.conformity.status == "passed"

    assert not (CARDS_ROOT / "resources").exists()


def test_only_the_five_bounded_paper_canaries_remain() -> None:
    diagnostics_root = REPO_ROOT / "bench" / "configs" / "diagnostics"
    cards = {
        path.relative_to(diagnostics_root).as_posix() for path in diagnostics_root.rglob("*.yaml")
    }
    assert cards == {
        "paper_canaries/fixmatch/cifar10-250-dev.yaml",
        "paper_canaries/flexmatch/cifar10-250-dev.yaml",
        "paper_canaries/free_match/cifar10-40-dev.yaml",
        "paper_canaries/grand/cora-dev.yaml",
        "paper_canaries/softmatch/cifar10-250-dev.yaml",
    }


def test_distribution_exposes_only_the_generic_benchmark_entrypoint() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = project["project"]["scripts"]
    assert scripts["modssc-bench"] == "bench.main:main"
    assert "modssc-reproduce" not in scripts
