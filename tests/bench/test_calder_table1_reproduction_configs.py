from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from bench.campaign.protocols.calder.official import (
    OFFICIAL_KNN_SHA256,
    OFFICIAL_PERMUTATIONS_SHA256,
    PERMUTATIONS_ARTIFACT_SHA256,
)
from bench.orchestrators.method_transductive import _build_spec
from bench.orchestrators.preprocess import _plan_from_dict as preprocess_plan_from_dict
from bench.schema import ExperimentConfig
from bench.seed_sweep import apply_global_seed
from modssc.graph.specs import GraphBuilderSpec
from modssc.sampling.plan import SamplingPlan
from modssc.transductive.registry import get_method_class

REPO_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTIONS_ROOT = REPO_ROOT / "bench" / "configs" / "reproductions"
PREPARED_PERMUTATIONS = (
    REPO_ROOT
    / "bench"
    / "assets"
    / "calder2020"
    / "protocol_inputs"
    / "splits"
    / "mnist-table1-permutations.ragged-int64-v1.npz"
)
PREPARED_PERMUTATIONS_SHA256 = PERMUTATIONS_ARTIFACT_SHA256

TABLE1_TARGETS = {
    "laplace_learning": {
        1: (16.1, 6.2),
        2: (28.2, 10.3),
        3: (42.0, 12.4),
        4: (57.8, 12.3),
        5: (69.5, 12.2),
    },
    "poisson_learning": {
        1: (90.2, 4.0),
        2: (93.6, 1.6),
        3: (94.5, 1.1),
        4: (94.9, 0.8),
        5: (95.3, 0.7),
    },
}
CASES = tuple(
    (method_id, budget, target_mean, target_std)
    for method_id, targets in TABLE1_TARGETS.items()
    for budget, (target_mean, target_std) in targets.items()
)


def _relative_path(method_id: str, budget: int) -> str:
    return f"{method_id}/mnist-table1-{budget}-label-per-class.yaml"


def _load(method_id: str, budget: int) -> tuple[Path, dict[str, Any], ExperimentConfig]:
    path = REPRODUCTIONS_ROOT / _relative_path(method_id, budget)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return path, raw, ExperimentConfig.from_dict(raw)


@pytest.mark.parametrize(("method_id", "budget", "target_mean", "target_std"), CASES)
def test_calder_table1_card_pins_the_published_protocol(
    method_id: str,
    budget: int,
    target_mean: float,
    target_std: float,
) -> None:
    path, _, cfg = _load(method_id, budget)
    text = path.read_text(encoding="utf-8")

    assert "Fidelity status: paper_matched in the frozen v4 campaign" in text
    assert "prepared directly through ModSSC without upstream source code" in text
    assert (
        f"Published Table 1 target: {target_mean:.1f} +/- {target_std:.1f}% unlabeled accuracy"
    ) in text

    assert cfg.run.seed == 0
    assert cfg.run.seeds == list(range(100))
    assert cfg.run.seeded_sections == ["sampling"]
    assert cfg.run.benchmark_mode is False
    assert cfg.run.fail_fast is True

    assert cfg.dataset.id == "mnist"
    assert cfg.dataset.download is False
    assert cfg.dataset.cache_dir == "${MODSSC_DATASET_CACHE_DIR}"

    sampling = SamplingPlan.from_dict(cfg.sampling.plan)
    assert sampling.split.kind == "holdout"
    assert sampling.split.test_fraction == pytest.approx(0.0)
    assert sampling.split.val_fraction == pytest.approx(0.0)
    assert sampling.split.stratify is True
    assert sampling.split.shuffle is True
    assert sampling.labeling.mode == "per_class"
    assert sampling.labeling.value == budget
    assert sampling.labeling.strategy == "balanced"
    assert sampling.labeling.min_per_class == budget
    assert sampling.labeling.per_class is True
    assert sampling.labeling.fixed_indices is None
    artifact = sampling.labeling.fixed_indices_artifact
    assert artifact is not None
    assert artifact.path.endswith(
        "protocol_inputs/splits/mnist-table1-permutations.ragged-int64-v1.npz"
    )
    assert artifact.sha256 == PREPARED_PERMUTATIONS_SHA256
    assert artifact.source_sha256 == OFFICIAL_PERMUTATIONS_SHA256
    assert artifact.key == "perm"
    assert artifact.index_stride == 5
    assert artifact.index_offset == budget - 1
    assert artifact.expected_size == budget * 10
    assert artifact.expected_per_class == budget
    assert sampling.imbalance.kind == "none"
    assert sampling.policy.merge_official_splits is True
    assert sampling.policy.respect_official_test is False
    assert sampling.policy.use_official_graph_masks is True
    assert sampling.policy.allow_override_official is False

    pre_plan = preprocess_plan_from_dict(cfg.preprocess.plan)
    assert cfg.preprocess.seed == 1
    assert cfg.preprocess.fit_on == "train"
    assert cfg.preprocess.cache is True
    assert pre_plan.output_key == "features.X"
    assert [step.step_id for step in pre_plan.steps] == [
        "labels.encode",
        "vision.ensure_num_channels",
        "vision.resize",
        "core.ensure_2d",
    ]

    assert cfg.graph is not None
    assert cfg.graph.enabled is True
    assert cfg.graph.seed == 1
    assert cfg.graph.cache is True
    assert cfg.graph.require_cache_hit is True
    assert cfg.graph.expected_fingerprint == (
        "209e8c9a6427fcd1403d76f1111654fc202e92d18d771ab37a5da92e14de693c"
    )
    assert (
        cfg.graph.expected_preprocess_fingerprint
        == "preprocess:7d44ae1b3a7f09a1c241a9b5e16ec7ff4502e3b4ef7c8aeadb4a6561caa25f20"
    )
    assert cfg.graph.spec == {
        "scheme": "knn",
        "metric": "euclidean",
        "k": 10,
        "symmetrize": "mean",
        "weights": {"kind": "knn_gaussian"},
        "normalize": "none",
        "self_loops": True,
        "include_self_in_knn": True,
        "edge_weight_dtype": "float64",
        "backend": "precomputed",
        "chunk_size": 1024,
        "precomputed_path": (
            "${MODSSC_ROOT}/bench/assets/calder2020/protocol_inputs/graph/mnist-vae-knn30.npz"
        ),
        "precomputed_sha256": OFFICIAL_KNN_SHA256,
        "feature_field": "features.X",
    }
    graph_spec = GraphBuilderSpec.from_dict(cfg.graph.spec)
    graph_spec.validate()
    assert graph_spec.scheme == "knn"
    assert graph_spec.metric == "euclidean"
    assert graph_spec.k == 10
    assert graph_spec.symmetrize == "mean"
    assert graph_spec.weights.kind == "knn_gaussian"
    assert graph_spec.normalize == "none"
    assert graph_spec.self_loops is True
    assert graph_spec.include_self_in_knn is True
    assert graph_spec.edge_weight_dtype == "float64"
    assert graph_spec.backend == "precomputed"
    assert graph_spec.chunk_size == 1024
    assert graph_spec.precomputed_sha256 == OFFICIAL_KNN_SHA256
    assert graph_spec.feature_field == "features.X"

    short_method = "poisson" if method_id == "poisson_learning" else "laplace"
    assert cfg.method.kind == "transductive"
    assert cfg.method.method_id == method_id
    assert cfg.method.profile == (
        f"paper:calder2020-mnist-table1-{short_method}-{budget}-label-per-class"
    )
    assert cfg.method.device.device == "cpu"
    assert cfg.method.device.dtype == "float32"

    method_spec = _build_spec(
        get_method_class(method_id),
        dict(cfg.method.params),
        strict=False,
    )
    if method_id == "poisson_learning":
        assert method_spec.backend == "numpy"
        assert method_spec.solver == "paper_iteration"
        assert method_spec.center_sources is True
        assert method_spec.balance_scores is True
        assert method_spec.min_iter == 50
        assert method_spec.max_iter == 1000
    else:
        assert method_spec.backend == "numpy"
        assert method_spec.solver == "calder2020_conjugate_gradient"
        assert method_spec.cg_tol == pytest.approx(1.0e-5)
        assert method_spec.cg_max_iter == 100_000

    assert cfg.evaluation.split_for_model_selection is None
    assert cfg.evaluation.report_splits == ["unlabeled"]
    assert cfg.evaluation.metrics == ["accuracy"]


def test_calder_table1_cards_share_one_preprocessing_and_graph_identity() -> None:
    loaded = {(method_id, budget): _load(method_id, budget)[1] for method_id, budget, _, _ in CASES}
    reference = loaded[("laplace_learning", 1)]

    for raw in loaded.values():
        assert raw["preprocess"] == reference["preprocess"]
        assert raw["graph"] == reference["graph"]

    assert {raw["preprocess"]["cache_dir"] for raw in loaded.values()} == {
        "${MODSSC_PREPROCESS_CACHE_DIR}/calder2020-mnist-table1"
    }
    assert {raw["graph"]["cache_dir"] for raw in loaded.values()} == {
        "${MODSSC_GRAPH_CACHE_DIR}/calder2020-mnist-table1"
    }

    for budget in range(1, 6):
        laplace = loaded[("laplace_learning", budget)]
        poisson = loaded[("poisson_learning", budget)]
        assert laplace["sampling"] == poisson["sampling"]
        assert laplace["run"]["seeds"] == poisson["run"]["seeds"] == list(range(100))


def test_calder_table1_laplace_and_poisson_use_exact_identical_permutations() -> None:
    with np.load(PREPARED_PERMUTATIONS, allow_pickle=False) as archive:
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        values = np.asarray(archive["values"], dtype=np.int64)

    specs = {
        (method_id, budget): SamplingPlan.from_dict(
            _load(method_id, budget)[2].sampling.plan
        ).labeling
        for method_id in TABLE1_TARGETS
        for budget in range(1, 6)
    }

    for seed in range(100):
        for budget in range(1, 6):
            row_index = seed * 5 + budget - 1
            row = values[offsets[row_index] : offsets[row_index + 1]]
            for method_id in TABLE1_TARGETS:
                artifact = specs[(method_id, budget)].fixed_indices_artifact
                assert artifact is not None
                assert artifact.sha256 == PREPARED_PERMUTATIONS_SHA256
                assert artifact.source_sha256 == OFFICIAL_PERMUTATIONS_SHA256
                assert artifact.index_stride == 5
                assert artifact.index_offset == budget - 1
                assert artifact.expected_size == row.size == budget * 10


def test_calder_table1_sweep_changes_only_the_sampling_seed() -> None:
    _, raw, _ = _load("poisson_learning", 1)

    for seed in (0, 37, 99):
        seeded = apply_global_seed(
            raw,
            seed=seed,
            seeded_sections=raw["run"]["seeded_sections"],
        )

        assert seeded["run"]["seed"] == seed
        assert seeded["sampling"]["seed"] == seed
        assert seeded["preprocess"]["seed"] == 1
        assert seeded["graph"]["seed"] == 1


def test_calder_acceptance_cards_use_population_std_and_exact_diagnostics() -> None:
    acceptance = yaml.safe_load(
        (REPO_ROOT / "bench/campaigns/article10-paper-acceptance.yaml").read_text(encoding="utf-8")
    )["protocols"]

    for method_id in TABLE1_TARGETS:
        short_name = method_id.removesuffix("_learning")
        for budget in range(1, 6):
            card = acceptance[f"calder-2020-mnist-table1-{short_name}-{budget}-label-per-class"]
            assert card["target"]["published_std_ddof"] == 0
            assert card["known_deviations"] == []
            diagnostics = {item["path"]: item for item in card["required_diagnostics"]}
            assert diagnostics["artifacts.method.diagnostics.converged"] == {
                "path": "artifacts.method.diagnostics.converged",
                "op": "eq",
                "value": True,
            }
            if method_id == "laplace_learning":
                assert diagnostics["artifacts.method.diagnostics.solver"]["value"] == (
                    "calder2020_conjugate_gradient"
                )
                assert "artifacts.method.diagnostics.absolute_residual" in diagnostics
            else:
                assert diagnostics["artifacts.method.diagnostics.solver"]["value"] == (
                    "paper_iteration"
                )
                assert diagnostics["artifacts.method.diagnostics.decision_rule"]["value"] == (
                    "paper_class_prior_correction"
                )
                assert "artifacts.method.diagnostics.mixing_residual" in diagnostics
