from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.data_loader.types import LoadedDataset, Split
from modssc.preprocess.api import (
    _build_purge_keep_sets,
    _dataset_fingerprint,
    _final_keep_keys,
    _initial_store,
    _maybe_warn_nonfinite,
    _purge_store,
    _shape_of,
    fit_transform,
    preprocess,
    resolve_plan,
)
from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.plan import PreprocessPlan, StepConfig
from modssc.preprocess.registry import StepRegistry, StepSpec
from modssc.preprocess.types import ResolvedStep


def test_dataset_fingerprint_fallback():
    train = Split(X=np.array([[1]]), y=np.array([0]))
    ds = LoadedDataset(train=train, test=None, meta={"modality": "tabular"})

    if "dataset_fingerprint" in ds.meta:
        del ds.meta["dataset_fingerprint"]

    fp = _dataset_fingerprint(ds)
    assert fp.startswith("dataset:")
    assert "dataset_fingerprint" not in ds.meta


def test_dataset_fingerprint_fallback_with_graph():
    train = Split(
        X=np.array([[1]]),
        y=np.array([0]),
        edges=np.array([[0], [0]]),
        masks={"train": np.array([True])},
    )
    ds = LoadedDataset(train=train, test=None, meta={})
    fp = _dataset_fingerprint(ds)
    assert fp.startswith("dataset:")


def test_initial_store_with_masks():
    split = Split(
        X=np.array([]), y=np.array([]), masks={"train": np.array([1]), "val": np.array([0])}
    )
    store = _initial_store(split)
    assert store.has("graph.mask.train")
    assert store.has("graph.mask.val")


def test_shape_of_handles_invalid_shape() -> None:
    class BadShape:
        shape = ("bad",)

    assert _shape_of(BadShape()) is None


def test_maybe_warn_nonfinite_paths(caplog) -> None:
    _maybe_warn_nonfinite("not-array", [1, 2, 3])

    big = np.zeros((1_000_001,), dtype=np.float32)
    _maybe_warn_nonfinite("big-array", big)

    with caplog.at_level("WARNING"):
        _maybe_warn_nonfinite("has-nan", np.array([1.0, np.nan], dtype=np.float32))


def test_resolve_plan_coverage():
    train = Split(X=np.array([]), y=np.array([]))
    test = Split(X=np.array([]), y=np.array([]))
    ds = LoadedDataset(train=train, test=test, meta={"modality": "tabular"})

    registry = MagicMock(spec=StepRegistry)

    spec_valid = StepSpec(
        step_id="valid",
        import_path="modssc.preprocess.steps.base:BaseStep",
        kind="transform",
        consumes={"raw.X"},
        produces={"out"},
        modalities={"tabular"},
    )

    spec_missing = StepSpec(
        step_id="missing_req",
        import_path="modssc.preprocess.steps.base:BaseStep",
        kind="transform",
        consumes={"non_existent"},
        produces={"out"},
    )

    registry.spec.side_effect = lambda sid: {
        "valid": spec_valid,
        "missing_req": spec_missing,
        "disabled": spec_valid,
    }[sid]

    plan = PreprocessPlan(
        steps=[
            StepConfig(step_id="disabled", enabled=False),
            StepConfig(step_id="missing_req", enabled=True, requires_fields=("non_existent",)),
            StepConfig(step_id="valid", enabled=True),
        ]
    )
    resolved = resolve_plan(ds, plan, registry=registry)

    assert len(resolved.steps) == 1
    assert resolved.steps[0].step_id == "valid"

    assert len(resolved.skipped) == 2
    reasons = [s.reason for s in resolved.skipped]
    assert "disabled" in reasons[0]
    assert "missing required fields" in reasons[1]


def test_preprocess_cache_dir_override(tmp_path):
    ds = LoadedDataset(
        train=Split(X=np.array([]), y=np.array([])), meta={"dataset_fingerprint": "fp"}
    )
    plan = PreprocessPlan(steps=[])

    with patch("modssc.preprocess.api.CacheManager") as mock_cm_cls:
        mock_cm = MagicMock()
        mock_cm_cls.for_dataset.return_value = mock_cm

        preprocess(ds, plan, cache=True, cache_dir=str(tmp_path))

        assert mock_cm.root == tmp_path.resolve()


def test_preprocess_fittable_missing_fit():
    ds = LoadedDataset(
        train=Split(X=np.array([]), y=np.array([])), meta={"dataset_fingerprint": "fp"}
    )
    plan = PreprocessPlan(steps=[StepConfig(step_id="bad_fit")])

    registry = MagicMock(spec=StepRegistry)
    registry.spec.return_value = StepSpec(
        step_id="bad_fit",
        import_path="modssc.preprocess.steps.base:BaseStep",
        kind="fittable",
        consumes={},
        produces={},
    )

    registry.instantiate.return_value = MagicMock(spec=[])

    with pytest.raises(PreprocessValidationError, match="declared fittable but has no fit"):
        preprocess(ds, plan, registry=registry, fit_indices=np.array([0]))


def test_preprocess_invalid_return_type():
    ds = LoadedDataset(
        train=Split(X=np.array([]), y=np.array([])), meta={"dataset_fingerprint": "fp"}
    )
    plan = PreprocessPlan(steps=[StepConfig(step_id="bad_ret")])

    registry = MagicMock(spec=StepRegistry)
    registry.spec.return_value = StepSpec(
        step_id="bad_ret",
        import_path="modssc.preprocess.steps.base:BaseStep",
        kind="transform",
        consumes={},
        produces={},
    )
    step_instance = MagicMock()
    step_instance.transform.return_value = "not a dict"
    registry.instantiate.return_value = step_instance

    with pytest.raises(PreprocessValidationError, match="must return a dict"):
        preprocess(ds, plan, registry=registry)


def test_preprocess_full_flow_with_test_split(tmp_path):
    train = Split(X=np.array([[1]]), y=np.array([0]), edges=np.array([[0], [0]]))
    test = Split(X=np.array([[2]]), y=np.array([1]), edges=np.array([[1], [1]]))
    ds = LoadedDataset(train=train, test=test, meta={"dataset_fingerprint": "fp"})

    plan = PreprocessPlan(steps=[StepConfig(step_id="step1")])

    registry = MagicMock(spec=StepRegistry)
    registry.spec.return_value = StepSpec(
        step_id="step1",
        import_path="modssc.preprocess.steps.base:BaseStep",
        kind="transform",
        consumes={"raw.X"},
        produces={"out"},
    )

    step_instance = MagicMock()

    step_instance.transform.side_effect = [
        {"out": np.array([[10]]), "graph.edge_weight": np.array([0.5])},
        {"out": np.array([[20]]), "graph.edge_weight": np.array([0.8])},
    ]
    registry.instantiate.return_value = step_instance

    with patch("modssc.preprocess.api.CacheManager") as mock_cm_cls:
        mock_cm = MagicMock()
        mock_cm.has_step_outputs.return_value = False
        mock_cm_cls.for_dataset.return_value = mock_cm

        res = preprocess(ds, plan, registry=registry, cache=True)

        assert res.test_artifacts is not None
        assert res.test_artifacts.has("out")
        assert res.dataset.test is not None

        assert isinstance(res.dataset.test.edges, dict)
        assert "edge_weight" in res.dataset.test.edges

        assert mock_cm.save_step_outputs.call_count == 2
        call_args = mock_cm.save_step_outputs.call_args_list
        assert call_args[1][1]["split"] == "test"


def test_preprocess_test_split_cache_hit():
    train = Split(X=np.array([[1]]), y=np.array([0]))
    test = Split(X=np.array([[2]]), y=np.array([1]))
    ds = LoadedDataset(train=train, test=test, meta={"dataset_fingerprint": "fp"})

    plan = PreprocessPlan(steps=[StepConfig(step_id="step1")])

    registry = MagicMock(spec=StepRegistry)
    registry.spec.return_value = StepSpec(
        step_id="step1",
        import_path="modssc.preprocess.steps.base:BaseStep",
        kind="transform",
        consumes={},
        produces={},
    )
    registry.instantiate.return_value = MagicMock()
    registry.instantiate.return_value.transform.return_value = {}

    with patch("modssc.preprocess.api.CacheManager") as mock_cm_cls:
        mock_cm = MagicMock()

        def has_outputs(fp, split):
            return split == "test"

        mock_cm.has_step_outputs.side_effect = has_outputs

        mock_cm.load_step_outputs.return_value = {"out": 1}
        mock_cm_cls.for_dataset.return_value = mock_cm

        preprocess(ds, plan, registry=registry, cache=True)

        mock_cm.load_step_outputs.assert_called_once()
        assert mock_cm.load_step_outputs.call_args[1]["split"] == "test"


def test_preprocess_test_split_invalid_return():
    train = Split(X=np.array([[1]]), y=np.array([0]))
    test = Split(X=np.array([[2]]), y=np.array([1]))
    ds = LoadedDataset(train=train, test=test, meta={"dataset_fingerprint": "fp"})

    plan = PreprocessPlan(steps=[StepConfig(step_id="step1")])

    registry = MagicMock(spec=StepRegistry)
    registry.spec.return_value = StepSpec(
        step_id="step1",
        import_path="modssc.preprocess.steps.base:BaseStep",
        kind="transform",
        consumes={},
        produces={},
    )

    step_instance = MagicMock()
    step_instance.transform.side_effect = [
        {},
        "not a dict",
    ]
    registry.instantiate.return_value = step_instance

    with patch("modssc.preprocess.api.CacheManager") as mock_cm_cls:
        mock_cm = MagicMock()
        mock_cm.has_step_outputs.return_value = False
        mock_cm_cls.for_dataset.return_value = mock_cm

        with pytest.raises(PreprocessValidationError, match="must return a dict"):
            preprocess(ds, plan, registry=registry, cache=True)


def test_fit_transform_alias():
    with patch("modssc.preprocess.api.preprocess") as mock_preprocess:
        fit_transform("arg", kw="arg")
        mock_preprocess.assert_called_once_with("arg", kw="arg")


def test_preprocess_fittable_success():
    train = Split(X=np.array([[1]]), y=np.array([0]))
    ds = LoadedDataset(train=train, meta={"dataset_fingerprint": "fp"})

    plan = PreprocessPlan(steps=[StepConfig(step_id="fittable")])

    registry = MagicMock(spec=StepRegistry)
    registry.spec.return_value = StepSpec(
        step_id="fittable",
        import_path="modssc.preprocess.steps.base:BaseStep",
        kind="fittable",
        consumes={},
        produces={},
    )

    step_instance = MagicMock()
    step_instance.transform.return_value = {}
    registry.instantiate.return_value = step_instance

    preprocess(ds, plan, registry=registry, fit_indices=np.array([0]), cache=False)

    step_instance.fit.assert_called_once()


def test_preprocess_no_cache():
    train = Split(X=np.array([[1]]), y=np.array([0]))
    ds = LoadedDataset(train=train, meta={"dataset_fingerprint": "fp"})
    plan = PreprocessPlan(steps=[])

    with patch("modssc.preprocess.api.CacheManager") as mock_cm_cls:
        preprocess(ds, plan, cache=False)
        mock_cm_cls.for_dataset.assert_not_called()


def test_preprocess_test_split_no_cache():
    train = Split(X=np.array([[1]]), y=np.array([0]))
    test = Split(X=np.array([[2]]), y=np.array([1]))
    ds = LoadedDataset(train=train, test=test, meta={"dataset_fingerprint": "fp"})

    plan = PreprocessPlan(steps=[StepConfig(step_id="step1")])

    registry = MagicMock(spec=StepRegistry)
    registry.spec.return_value = StepSpec(
        step_id="step1",
        import_path="modssc.preprocess.steps.base:BaseStep",
        kind="transform",
        consumes={},
        produces={},
    )
    registry.instantiate.return_value = MagicMock()
    registry.instantiate.return_value.transform.return_value = {}

    with patch("modssc.preprocess.api.CacheManager") as mock_cm_cls:
        preprocess(ds, plan, registry=registry, cache=False)
        mock_cm_cls.for_dataset.assert_not_called()


def test_dataset_fingerprint_direct():
    train = Split(X=np.array([[1]]), y=np.array([0]))
    ds = LoadedDataset(train=train, meta={})
    fp = _dataset_fingerprint(ds)
    assert fp.startswith("dataset:")


def test_final_keep_keys_with_labels_and_graph():
    steps = (
        ResolvedStep(
            step_id="labels",
            params={},
            index=0,
            spec=StepSpec(
                step_id="labels",
                import_path="x",
                kind="transform",
                consumes=(),
                produces=("labels.y",),
            ),
        ),
        ResolvedStep(
            step_id="graph",
            params={},
            index=1,
            spec=StepSpec(
                step_id="graph",
                import_path="x",
                kind="transform",
                consumes=(),
                produces=("graph.edge_weight", "graph.edge_index"),
            ),
        ),
    )
    keep = _final_keep_keys(steps, output_key="raw.X", initial_keys={"raw.X", "raw.y"})
    assert "raw.X" in keep
    assert "labels.y" in keep
    assert "graph.edge_weight" in keep
    assert "graph.edge_index" in keep


def test_final_keep_keys_missing_output_and_edge_index():
    steps = (
        ResolvedStep(
            step_id="graph",
            params={},
            index=0,
            spec=StepSpec(
                step_id="graph",
                import_path="x",
                kind="transform",
                consumes=(),
                produces=("graph.edge_index",),
            ),
        ),
    )
    keep = _final_keep_keys(steps, output_key="features.X", initial_keys={"raw.X", "raw.y"})
    assert "features.X" in keep
    assert "raw.X" in keep
    assert "raw.y" in keep
    assert "graph.edge_index" in keep


def test_build_purge_keep_sets_includes_implicit_consumes():
    steps = (
        ResolvedStep(
            step_id="step0",
            params={},
            index=0,
            spec=StepSpec(
                step_id="step0",
                import_path="x",
                kind="transform",
                consumes=(),
                produces=("features.X",),
            ),
        ),
        ResolvedStep(
            step_id="graph.node2vec",
            params={},
            index=1,
            spec=StepSpec(
                step_id="graph.node2vec",
                import_path="x",
                kind="transform",
                consumes=(),
                produces=("features.X",),
            ),
        ),
    )
    keep_sets = _build_purge_keep_sets(
        steps, output_key="features.X", initial_keys={"raw.X", "raw.y"}
    )
    assert "raw.X" in keep_sets[0]
    assert "raw.y" in keep_sets[0]


def test_purge_store_filters_keys():
    store = _initial_store(Split(X=np.array([[1]]), y=np.array([0])))
    store.set("features.X", np.array([[1.0]]))
    _purge_store(store, keep={"features.X"})
    assert store.keys() == ["features.X"]
    _purge_store(store, keep=set())
    assert store.keys() == []


def test_preprocess_incomplete_cache_falls_back():
    train = Split(X=np.array([[1]]), y=np.array([0]))
    test = Split(X=np.array([[2]]), y=np.array([1]))
    ds = LoadedDataset(train=train, test=test, meta={"dataset_fingerprint": "fp"})
    plan = PreprocessPlan(steps=[StepConfig(step_id="step1")])

    registry = MagicMock(spec=StepRegistry)
    registry.spec.return_value = StepSpec(
        step_id="step1",
        import_path="modssc.preprocess.steps.base:BaseStep",
        kind="transform",
        consumes={},
        produces={"out"},
    )

    step_instance = MagicMock()
    step_instance.transform.side_effect = [{"out": np.array([[3]])}, {"out": np.array([[4]])}]
    registry.instantiate.return_value = step_instance

    with patch("modssc.preprocess.api.CacheManager") as mock_cm_cls:
        mock_cm = MagicMock()
        mock_cm.has_step_outputs.return_value = True
        mock_cm.load_step_outputs.return_value = {}
        mock_cm_cls.for_dataset.return_value = mock_cm

        preprocess(ds, plan, registry=registry, cache=True)
    assert step_instance.transform.call_count == 2


def test_preprocess_purge_unused_artifacts(caplog):
    train = Split(X=np.array([[1]]), y=np.array([0]))
    test = Split(X=np.array([[2]]), y=np.array([1]))
    ds = LoadedDataset(train=train, test=test, meta={"dataset_fingerprint": "fp"})
    plan = PreprocessPlan(steps=[StepConfig(step_id="step1")], output_key="raw.X")

    registry = MagicMock(spec=StepRegistry)
    registry.spec.return_value = StepSpec(
        step_id="step1",
        import_path="modssc.preprocess.steps.base:BaseStep",
        kind="transform",
        consumes=(),
        produces={"features.X"},
    )

    step_instance = MagicMock()
    step_instance.transform.side_effect = [
        {"features.X": np.array([[10]])},
        {"features.X": np.array([[20]])},
    ]
    registry.instantiate.return_value = step_instance

    with caplog.at_level("DEBUG", logger="modssc.preprocess.api"):
        res = preprocess(ds, plan, registry=registry, cache=False, purge_unused_artifacts=True)
    assert res.train_artifacts.keys() == ["raw.X", "raw.y"]
    assert "features.X" not in res.train_artifacts
    assert res.test_artifacts is not None
    assert res.test_artifacts.keys() == ["raw.X", "raw.y"]
    assert "features.X" not in res.test_artifacts


def test_preprocess_purge_unused_artifacts_train_only():
    train = Split(X=np.array([[1]]), y=np.array([0]))
    ds = LoadedDataset(train=train, test=None, meta={"dataset_fingerprint": "fp"})
    plan = PreprocessPlan(steps=[StepConfig(step_id="step1")], output_key="raw.X")

    registry = MagicMock(spec=StepRegistry)
    registry.spec.return_value = StepSpec(
        step_id="step1",
        import_path="modssc.preprocess.steps.base:BaseStep",
        kind="transform",
        consumes=(),
        produces={"features.X"},
    )
    step_instance = MagicMock()
    step_instance.transform.return_value = {"features.X": np.array([[10]])}
    registry.instantiate.return_value = step_instance

    res = preprocess(ds, plan, registry=registry, cache=False, purge_unused_artifacts=True)
    assert res.test_artifacts is None
    assert res.train_artifacts.keys() == ["raw.X", "raw.y"]
    assert "features.X" not in res.train_artifacts
