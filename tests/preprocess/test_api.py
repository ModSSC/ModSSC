from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.data_loader.types import LoadedDataset, Split
from modssc.preprocess.api import (
    _build_purge_keep_sets,
    _dataset_fingerprint,
    _device_hint,
    _estimate_bytes,
    _estimate_collection_bytes,
    _final_keep_keys,
    _format_bytes,
    _format_size_estimate,
    _get_torch,
    _gpu_model_for_device,
    _initial_store,
    _maybe_log_gpu_info,
    _maybe_warn_nonfinite,
    _normalize_device_name,
    _purge_store,
    _shape_of,
    _split_data_size,
    fit_transform,
    preprocess,
    resolve_plan,
)
from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.plan import PreprocessPlan, StepConfig
from modssc.preprocess.registry import StepRegistry, StepSpec
from modssc.preprocess.store import ArtifactStore
from modssc.preprocess.types import ResolvedStep


class DummyTensor:
    def __init__(self, device: str = "cuda:0", *, element_size: int = 2, nelement: int = 3) -> None:
        self.device = device
        self._element_size = element_size
        self._nelement = nelement

    def element_size(self) -> int:
        return self._element_size

    def nelement(self) -> int:
        return self._nelement


class DummyCuda:
    def __init__(
        self,
        *,
        available: bool = True,
        name: str = "Mock GPU",
        current_index: int = 0,
        raise_name: bool = False,
    ) -> None:
        self._available = available
        self._name = name
        self._current_index = current_index
        self._raise_name = raise_name

    def is_available(self) -> bool:
        return self._available

    def current_device(self) -> int:
        return self._current_index

    def get_device_name(self, index: int) -> str:
        if self._raise_name:
            raise RuntimeError("name error")
        return self._name


class DummyTorchDevice:
    def __init__(
        self,
        *,
        device_type: str = "cuda",
        index: int | None = None,
        cuda: DummyCuda | None = None,
        device_raises: bool = False,
    ) -> None:
        self._device_type = device_type
        self._index = index
        self._device_raises = device_raises
        self.cuda = cuda or DummyCuda()

    def device(self, device_str: str) -> SimpleNamespace:
        if self._device_raises:
            raise RuntimeError("device error")
        return SimpleNamespace(type=self._device_type, index=self._index)


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


def test_get_torch_import_error(monkeypatch) -> None:
    def raise_import(name: str) -> None:
        raise ImportError()

    _get_torch.cache_clear()
    monkeypatch.setattr("importlib.import_module", raise_import)
    assert _get_torch() is None
    _get_torch.cache_clear()


def test_estimate_collection_bytes_scales_known_items() -> None:
    size, unknown, approx = _estimate_collection_bytes(
        [b"abcd"], max_items=10, depth=0, total_len=3
    )
    assert size == 12
    assert unknown == 0
    assert approx is True


def test_estimate_collection_bytes_accumulates_unknown() -> None:
    size, unknown, approx = _estimate_collection_bytes(
        [object()], max_items=10, depth=0, total_len=3
    )
    assert size == 0
    assert unknown == 3
    assert approx is True


def test_estimate_collection_bytes_max_items_break() -> None:
    size, unknown, approx = _estimate_collection_bytes(
        [b"a", b"b"], max_items=1, depth=0, total_len=None
    )
    assert size == 1
    assert unknown == 0
    assert approx is True


def test_estimate_bytes_none_and_bytes() -> None:
    assert _estimate_bytes(None) == (0, 0, False)
    assert _estimate_bytes(b"abc") == (3, 0, False)


def test_estimate_bytes_invalid_nbytes_falls_back() -> None:
    class BadNbytes:
        nbytes = "bad"

    assert _estimate_bytes(BadNbytes()) == (0, 1, False)


def test_estimate_bytes_sparse_like_success() -> None:
    class SparseLike:
        def __init__(self) -> None:
            self.data = np.zeros(4, dtype=np.int8)
            self.indices = np.zeros(2, dtype=np.int8)
            self.indptr = np.zeros(3, dtype=np.int8)

    size, unknown, approx = _estimate_bytes(SparseLike())
    assert size == 9
    assert unknown == 0
    assert approx is False


def test_estimate_bytes_sparse_like_exception() -> None:
    class BadNbytes:
        @property
        def nbytes(self) -> int:
            raise RuntimeError("boom")

    class SparseLike:
        def __init__(self) -> None:
            self.data = BadNbytes()
            self.indices = np.zeros(1, dtype=np.int8)
            self.indptr = np.zeros(1, dtype=np.int8)

    assert _estimate_bytes(SparseLike()) == (0, 1, False)


def test_estimate_bytes_depth_limit() -> None:
    assert _estimate_bytes(object(), depth=2) == (0, 1, False)


def test_estimate_bytes_mapping() -> None:
    size, unknown, approx = _estimate_bytes({"a": b"abc"})
    assert size == 3
    assert unknown == 0
    assert approx is False


def test_estimate_bytes_torch_tensor() -> None:
    dummy_torch = SimpleNamespace(Tensor=DummyTensor)
    with patch("modssc.preprocess.api._get_torch", return_value=dummy_torch):
        size, unknown, approx = _estimate_bytes(DummyTensor(element_size=4, nelement=5))
    assert size == 20
    assert unknown == 0
    assert approx is False


def test_format_bytes_kb() -> None:
    assert _format_bytes(1024) == "1.0 KB"


def test_format_bytes_pb() -> None:
    assert _format_bytes(1024**5) == "1.0 PB"


def test_format_size_estimate_unknown() -> None:
    assert _format_size_estimate(0, unknown=1, approx=False) == "unknown"


def test_format_size_estimate_with_extras() -> None:
    formatted = _format_size_estimate(1024, unknown=2, approx=True)
    assert formatted.startswith("1.0 KB")
    assert "approx" in formatted
    assert "unknown=2" in formatted


def test_split_data_size_none_store() -> None:
    assert _split_data_size(None) == (0, 0, False)


def test_split_data_size_missing_x_key_and_masks() -> None:
    store = ArtifactStore()
    store.set("raw.y", np.array([1], dtype=np.int8))
    store.set("graph.mask.train", np.array([True]))
    size, unknown, approx = _split_data_size(store)
    assert size == store.get("raw.y").nbytes + store.get("graph.mask.train").nbytes
    assert unknown == 0
    assert approx is False


def test_split_data_size_output_key_same_as_y_key() -> None:
    store = ArtifactStore()
    store.set("raw.y", np.array([1], dtype=np.int8))
    size, unknown, approx = _split_data_size(store, output_key="raw.y")
    assert size == store.get("raw.y").nbytes
    assert unknown == 0
    assert approx is False


def test_device_hint_prefers_params_and_falls_back() -> None:
    class StepObj:
        device = "mps:0"

    assert _device_hint({"device": "cuda:1"}, StepObj()) == "cuda:1"
    assert _device_hint("not-a-dict", StepObj()) == "mps:0"


def test_normalize_device_name_auto() -> None:
    with patch("modssc.preprocess.api.resolve_device_name", return_value="cuda:0") as mock_resolve:
        assert _normalize_device_name("auto", torch="torch") == "cuda:0"
    mock_resolve.assert_called_once_with("auto", torch="torch")


def test_gpu_model_for_device_cuda_not_available() -> None:
    torch = DummyTorchDevice(cuda=DummyCuda(available=False))
    assert _gpu_model_for_device(torch, "cuda:0") is None


def test_gpu_model_for_device_cuda_current_device_name() -> None:
    torch = DummyTorchDevice(
        device_type="cuda",
        index=None,
        cuda=DummyCuda(available=True, name="Mock", current_index=2),
    )
    assert _gpu_model_for_device(torch, "cuda") == "Mock"


def test_gpu_model_for_device_cuda_name_failure() -> None:
    torch = DummyTorchDevice(device_type="cuda", index=0, cuda=DummyCuda(raise_name=True))
    assert _gpu_model_for_device(torch, "cuda:0") is None


def test_gpu_model_for_device_cpu() -> None:
    torch = DummyTorchDevice(device_type="cpu")
    assert _gpu_model_for_device(torch, "cpu") is None


def test_gpu_model_for_device_mps_device() -> None:
    torch = DummyTorchDevice(device_type="mps")
    assert _gpu_model_for_device(torch, "mps") == "Apple MPS"


def test_gpu_model_for_device_mps_string_when_device_raises() -> None:
    torch = DummyTorchDevice(device_raises=True)
    assert _gpu_model_for_device(torch, "mps:0") == "Apple MPS"


def test_maybe_log_gpu_info_without_torch() -> None:
    with patch("modssc.preprocess.api._get_torch", return_value=None):
        assert (
            _maybe_log_gpu_info(
                {}, object(), produced_train=None, produced_test=None, use_device_hint=False
            )
            is False
        )


def test_maybe_log_gpu_info_logs_with_model_and_without() -> None:
    dummy_torch = SimpleNamespace(Tensor=DummyTensor)
    outputs = {"x": DummyTensor(device="cuda:0")}
    with (
        patch("modssc.preprocess.api._get_torch", return_value=dummy_torch),
        patch(
            "modssc.preprocess.api._gpu_model_for_device",
            side_effect=["Mock GPU", None],
        ),
        patch("modssc.preprocess.api.logger.info") as mock_info,
    ):
        assert (
            _maybe_log_gpu_info(
                {}, object(), produced_train=outputs, produced_test=None, use_device_hint=False
            )
            is True
        )
        assert (
            _maybe_log_gpu_info(
                {}, object(), produced_train=outputs, produced_test=None, use_device_hint=False
            )
            is True
        )
    assert mock_info.call_count == 2
    first_args = mock_info.call_args_list[0].args
    second_args = mock_info.call_args_list[1].args
    assert first_args == ("Preprocess GPU device: device=%s model=%s", "cuda:0", "Mock GPU")
    assert second_args == ("Preprocess GPU device: device=%s", "cuda:0")


def test_preprocess_skips_gpu_log_after_first_step() -> None:
    train = Split(X=np.array([[1]]), y=np.array([0]))
    ds = LoadedDataset(train=train, meta={"dataset_fingerprint": "fp"})
    plan = PreprocessPlan(steps=[StepConfig(step_id="step1"), StepConfig(step_id="step2")])

    registry = MagicMock(spec=StepRegistry)
    registry.spec.side_effect = lambda sid: StepSpec(
        step_id=sid,
        import_path="modssc.preprocess.steps.base:BaseStep",
        kind="transform",
        consumes=(),
        produces=(),
    )
    step1 = MagicMock()
    step1.transform.return_value = {}
    step2 = MagicMock()
    step2.transform.return_value = {}
    registry.instantiate.side_effect = [step1, step2]

    with patch("modssc.preprocess.api._maybe_log_gpu_info", return_value=True) as mock_gpu:
        preprocess(ds, plan, registry=registry, cache=False)
    assert mock_gpu.call_count == 1


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


@pytest.mark.parametrize(
    "step_id, expected",
    [
        ("graph.node2vec", {"raw.X", "raw.y"}),
        ("graph.dgi", {"raw.X"}),
    ],
)
def test_build_purge_keep_sets_includes_implicit_consumes(step_id, expected):
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
            step_id=step_id,
            params={},
            index=1,
            spec=StepSpec(
                step_id=step_id,
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
    for key in expected:
        assert key in keep_sets[0]


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
