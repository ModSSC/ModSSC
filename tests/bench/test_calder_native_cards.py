from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from bench import main as bench_main
from bench.orchestrators import dataset as dataset_orch
from bench.orchestrators import sampling as sampling_orch
from bench.schema import BenchConfigError, ExperimentConfig
from bench.seed_sweep import apply_global_seed
from modssc.data_loader.types import LoadedDataset, Split
from modssc.graph.specs import GraphBuilderSpec
from modssc.preprocess import PreprocessPlan
from modssc.runtime.method_spec import build_method_spec
from modssc.sampling.labeling import select_labeled
from modssc.sampling.plan import SamplingPlan
from modssc.transductive.registry import get_method_class

REPO_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTIONS_ROOT = REPO_ROOT / "bench" / "configs" / "reproductions"

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
def test_calder_table1_card_declares_native_statistical_protocol(
    method_id: str,
    budget: int,
    target_mean: float,
    target_std: float,
) -> None:
    path, _, cfg = _load(method_id, budget)
    text = path.read_text(encoding="utf-8")

    assert "article does not publish generated arrays or label draws" in text
    assert "frozen v4 artifact-backed results" not in text
    assert "load ModSSC's SHA-pinned VAE kNN graph" not in text
    assert (
        f"Published Table 1 target: {target_mean:.1f} +/- {target_std:.1f}% unlabeled accuracy"
    ) in text

    assert cfg.run.seed == 0
    assert cfg.run.seeds == list(range(100))
    assert cfg.run.seeded_sections == ["sampling"]
    assert cfg.run.benchmark_mode is True
    assert cfg.acceptance is not None
    assert cfg.acceptance.protocol_id == (
        f"calder-2020-mnist-table1-{method_id.removesuffix('_learning')}-{budget}-label-per-class"
    )
    assert cfg.acceptance.method_id == method_id
    assert cfg.acceptance.repetitions == 100
    assert cfg.acceptance.fidelity_ceiling == "paper_approx"
    assert len(cfg.acceptance.deviations) == 3
    assert any("VAE2 was introduced after" in item for item in cfg.acceptance.deviations)
    assert any("Annoy version and seed" in item for item in cfg.acceptance.deviations)
    assert any("label draws" in item for item in cfg.acceptance.deviations)
    assert cfg.acceptance.unknowns == ()
    expected_protocol = (
        "docs/replications/protocols/poisson-learning-calder-2020.md"
        if method_id == "poisson_learning"
        else "docs/replications/protocols/laplace-learning-calder-2020.md"
    )
    assert cfg.acceptance.conformity.evidence == (expected_protocol,)
    assert cfg.run.fail_fast is True

    assert cfg.dataset.id == "mnist"
    assert cfg.dataset.download is False
    assert cfg.dataset.cache_dir == "${MODSSC_DATASET_CACHE_DIR}"
    assert cfg.dataset.integrity is not None
    assert cfg.dataset.integrity.fingerprint == (
        "a509362fa8ce20622694ff1f4d85b7a5a2009a412d9e6ab54d1c35a2f8a6ab01"
    )
    assert cfg.dataset.integrity.content_sha256 == (
        "a918159751de8d67828d471643f73e70781ae5986e63d7d544be33e04139a89d"
    )

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
    assert sampling.labeling.fixed_indices_artifact is None
    assert sampling.labeling.selection_order == "choice"
    assert sampling.imbalance.kind == "none"
    assert sampling.policy.merge_official_splits is True
    assert sampling.policy.respect_official_test is False
    assert sampling.policy.use_official_graph_masks is True
    assert sampling.policy.allow_override_official is False

    pre_plan = PreprocessPlan.from_dict(cfg.preprocess.plan)
    assert cfg.preprocess.seed == 1
    assert cfg.preprocess.fit_on == "train"
    assert cfg.preprocess.cache is True
    assert pre_plan.output_key == "features.vae"
    assert [step.step_id for step in pre_plan.steps] == [
        "labels.encode",
        "vision.ensure_num_channels",
        "vision.resize",
        "core.ensure_2d",
        "core.vae",
    ]
    assert dict(pre_plan.steps[-1].params) == {
        "preset": "graphlearning_mnist_vae2",
        "cache_key": "calder2020-mnist-vae2-seed1",
        "model_seed": 1,
        "device": "cuda",
        "fit_scope": "all",
    }

    assert cfg.graph is not None
    assert cfg.graph.enabled is True
    assert cfg.graph.seed == 1
    assert cfg.graph.cache is True
    assert cfg.graph.require_cache_hit is False
    assert cfg.graph.expected_fingerprint is None
    assert cfg.graph.expected_preprocess_fingerprint is None
    assert "precomputed_path" not in cfg.graph.spec
    assert "precomputed_sha256" not in cfg.graph.spec

    graph_spec = GraphBuilderSpec.from_dict(cfg.graph.spec)
    graph_spec.validate()
    assert graph_spec.scheme == "knn"
    assert graph_spec.metric == "euclidean"
    assert graph_spec.k == 10
    assert graph_spec.symmetrize == "mean"
    assert graph_spec.weights.kind == "knn_gaussian"
    assert graph_spec.normalize == "none"
    assert graph_spec.include_self_in_knn is True
    assert graph_spec.edge_weight_dtype == "float64"
    assert graph_spec.backend == "annoy"
    assert graph_spec.faiss_exact is False
    assert graph_spec.annoy_n_trees == 10
    assert graph_spec.annoy_query_k == 30
    assert graph_spec.annoy_search_k == -1
    assert graph_spec.annoy_rerank is False
    assert graph_spec.chunk_size == 1024
    assert graph_spec.feature_field == "features.vae"
    assert graph_spec.self_loops is True
    assert graph_spec.diagonal_policy == "preserve"

    short_method = "poisson" if method_id == "poisson_learning" else "laplace"
    assert cfg.method.kind == "transductive"
    assert cfg.method.method_id == method_id
    assert cfg.method.profile == (
        f"paper:calder2020-mnist-table1-{short_method}-{budget}-label-per-class"
    )
    assert cfg.method.device.device == "cpu"
    assert cfg.method.device.dtype == "float32"

    method_spec = build_method_spec(
        get_method_class(method_id),
        dict(cfg.method.params),
        require_spec=True,
        strict=False,
    )
    assert method_spec.require_convergence is True
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


def test_calder_table1_cards_share_native_vae_and_directed_knn_contract() -> None:
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
        assert laplace["graph"]["spec"]["self_loops"] is True
        assert laplace["graph"]["spec"]["diagonal_policy"] == "preserve"
        assert poisson["graph"]["spec"]["self_loops"] is True
        assert poisson["graph"]["spec"]["diagonal_policy"] == "preserve"


def test_calder_native_balanced_draws_are_deterministic_and_shared() -> None:
    y = np.repeat(np.arange(10, dtype=np.int64), 20)
    train_idx = np.arange(y.size, dtype=np.int64)

    for budget in range(1, 6):
        selected_by_seed: dict[int, np.ndarray] = {}
        for seed in (0, 37, 99):
            selections: list[np.ndarray] = []
            for method_id in TABLE1_TARGETS:
                sampling_plan = SamplingPlan.from_dict(_load(method_id, budget)[2].sampling.plan)
                selected = select_labeled(
                    train_idx=train_idx,
                    y=y,
                    spec=sampling_plan.labeling,
                    rng=np.random.default_rng(
                        sampling_plan.component_seeds.resolve(seed)["labeling"]
                    ),
                )
                selections.append(selected)
                counts = np.bincount(y[selected], minlength=10)
                np.testing.assert_array_equal(counts, np.full(10, budget, dtype=np.int64))

            np.testing.assert_array_equal(selections[0], selections[1])
            selected_by_seed[seed] = selections[0]

        assert not np.array_equal(selected_by_seed[0], selected_by_seed[37])


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
        vae_params = seeded["preprocess"]["plan"]["steps"][-1]["params"]
        assert vae_params["model_seed"] == 1
        assert vae_params["fit_scope"] == "all"


def test_calder_provider_integrity_is_checked_before_native_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the lightweight load -> integrity -> merge Calder boundary."""

    _, _, cfg = _load("poisson_learning", 1)
    assert cfg.dataset.integrity is not None
    source_fingerprint = cfg.dataset.integrity.fingerprint
    source_content_sha256 = cfg.dataset.integrity.content_sha256
    assert source_fingerprint is not None
    assert source_content_sha256 is not None

    source = LoadedDataset(
        train=Split(X=np.array([[0.0], [1.0]]), y=np.array([0, 1])),
        test=Split(X=np.array([[2.0]]), y=np.array([1])),
        meta={
            "dataset_fingerprint": source_fingerprint,
            "dataset_content_sha256": source_content_sha256,
            "modality": "vision",
        },
    )
    monkeypatch.setattr(dataset_orch, "load_dataset", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        dataset_orch,
        "verify_dataset_content",
        lambda *args, **kwargs: {
            "cache_fingerprint": source_fingerprint,
            "content_sha256": source_content_sha256,
            "content_manifest_sha256": cfg.dataset.integrity.content_manifest_sha256,
            "cache_state_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(
        dataset_orch,
        "dataset_info",
        lambda _dataset_id: SimpleNamespace(
            as_dict=lambda: {"provider": "test", "modality": "vision"}
        ),
    )

    loaded, _ = dataset_orch.load(cfg.dataset)
    dataset_orch.verify_integrity(loaded, cfg.dataset)
    merged = sampling_orch.prepare_dataset(loaded, plan_dict=cfg.sampling.plan)

    assert merged.test is None
    assert merged.meta["dataset_fingerprint_source"] == source_fingerprint
    assert merged.meta["dataset_fingerprint"] != source_fingerprint
    np.testing.assert_array_equal(merged.train.y, np.array([0, 1, 1]))

    with pytest.raises(BenchConfigError, match="integrity.fingerprint"):
        dataset_orch.verify_integrity(merged, cfg.dataset)

    runner_source = inspect.getsource(bench_main._run_experiment_single)
    assert runner_source.index("ds_orch.verify_integrity") < runner_source.index(
        "sampling_orch.prepare_dataset"
    )
