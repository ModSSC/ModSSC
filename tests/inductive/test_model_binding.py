from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import modssc.inductive.model_binding as model_binding
from modssc.inductive.model_binding import (
    ModelBindingError,
    ModelBindingSpec,
    ModelBuildConfig,
    bind_model_to_spec,
)
from modssc.inductive.registry import available_methods, get_method_class, get_method_info
from modssc.runtime.method_spec import build_method_spec

EXPECTED_BINDINGS = {
    "adamatch": ("single", ("model_bundle",)),
    "adsh": ("single", ("model_bundle",)),
    "comatch": ("single", ("model_bundle",)),
    "daso": ("single", ("model_bundle",)),
    "deep_co_training": ("pair", ("model_bundle_1", "model_bundle_2")),
    "defixmatch": ("single", ("model_bundle",)),
    "fixmatch": ("single", ("model_bundle",)),
    "flexmatch": ("single", ("model_bundle",)),
    "free_match": ("single", ("model_bundle",)),
    "mean_teacher": ("single", ("model_bundle",)),
    "meta_pseudo_labels": ("teacher_student", ("student_bundle", "teacher_bundle")),
    "mixmatch": ("single", ("model_bundle",)),
    "noisy_student": ("single", ("model_bundle",)),
    "pi_model": ("single", ("model_bundle",)),
    "simclr_v2": ("pretrain_finetune", ("pretrain_bundle", "finetune_bundle")),
    "softmatch": ("single", ("model_bundle",)),
    "temporal_ensembling": ("single", ("model_bundle",)),
    "uda": ("single", ("model_bundle",)),
    "vat": ("single", ("model_bundle",)),
}
MODEL_BUNDLE_FIELDS = {
    "model_bundle",
    "teacher_bundle",
    "student_bundle",
    "model_bundle_1",
    "model_bundle_2",
    "pretrain_bundle",
    "finetune_bundle",
    "shared_bundle",
    "head_bundles",
}


def test_every_method_with_bundle_fields_declares_a_native_binding() -> None:
    missing: list[str] = []
    invalid: list[str] = []
    for method_id in available_methods():
        method_cls = get_method_class(method_id)
        spec = getattr(method_cls(), "spec", None)
        if spec is None or not is_dataclass(spec):
            continue
        spec_fields = {item.name for item in fields(spec)}
        if not spec_fields.intersection(MODEL_BUNDLE_FIELDS):
            continue
        binding = get_method_info(method_id).model_binding
        if binding.kind == "none":
            missing.append(method_id)
            continue
        declared = set(binding.bundle_fields)
        declared.update(
            name
            for name in (binding.shared_bundle_field, binding.head_bundles_field)
            if name is not None
        )
        if not declared.issubset(spec_fields):
            invalid.append(method_id)

    assert not missing, f"methods missing model bindings: {missing}"
    assert not invalid, f"methods declare absent model fields: {invalid}"


@pytest.mark.parametrize("method_id", sorted(EXPECTED_BINDINGS))
def test_deep_methods_declare_model_binding(method_id: str) -> None:
    expected_kind, expected_fields = EXPECTED_BINDINGS[method_id]
    binding = get_method_info(method_id).model_binding

    assert binding.kind == expected_kind
    assert binding.bundle_fields == expected_fields


def test_trinet_owns_three_head_classifier_policy() -> None:
    binding = get_method_info("trinet").model_binding

    assert binding.kind == "shared_heads"
    assert binding.shared_bundle_field == "shared_bundle"
    assert binding.head_bundles_field == "head_bundles"
    assert binding.head_count == 3
    assert binding.head_classifier_ids == ("mlp", "logreg")
    assert binding.head_classifier_fallback == "logreg"


def test_build_method_spec_applies_params_in_src() -> None:
    method_cls = get_method_class("fixmatch")

    spec = build_method_spec(method_cls, {"batch_size": 7})

    assert spec is not None
    assert spec.batch_size == 7
    assert spec.model_bundle is None


@dataclass(frozen=True)
class _ManyBundleSpec:
    first: Any | None = None
    second: Any | None = None


@pytest.mark.parametrize(
    ("binding", "fields"),
    [
        (ModelBindingSpec.single("first"), ("first",)),
        (
            ModelBindingSpec.teacher_student(
                student_field="first",
                teacher_field="second",
            ),
            ("first", "second"),
        ),
        (
            ModelBindingSpec.pair(first_field="first", second_field="second"),
            ("first", "second"),
        ),
        (
            ModelBindingSpec.pretrain_finetune(
                pretrain_field="first",
                finetune_field="second",
            ),
            ("first", "second"),
        ),
    ],
)
def test_native_binding_constructs_declared_independent_bundles(
    monkeypatch: pytest.MonkeyPatch,
    binding: ModelBindingSpec,
    fields: tuple[str, ...],
) -> None:
    torch = pytest.importorskip("torch")
    calls: list[dict[str, Any]] = []

    def fake_builder(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(model=object(), seed=kwargs["seed"])

    monkeypatch.setattr(model_binding, "build_torch_bundle_from_classifier", fake_builder)
    bound = bind_model_to_spec(
        _ManyBundleSpec(),
        ModelBuildConfig(classifier_id="mlp", classifier_backend="torch"),
        binding=binding,
        X_l=torch.ones((4, 2), dtype=torch.float32),
        y_l=torch.tensor([0, 1, 0, 1]),
        default_ema=False,
        seed=17,
        strict=True,
    )

    assert [getattr(bound, field_name).seed for field_name in fields] == [
        17 + offset for offset in range(len(fields))
    ]
    assert [call["seed"] for call in calls] == [17 + offset for offset in range(len(fields))]


def _importable_test_factory(*, marker: str) -> Any:
    return SimpleNamespace(marker=marker)


def test_native_binding_preserves_import_string_factories() -> None:
    bound = bind_model_to_spec(
        _ManyBundleSpec(),
        ModelBuildConfig(
            factory=f"{__name__}:_importable_test_factory",
            params={"marker": "loaded"},
        ),
        binding=ModelBindingSpec.pair(first_field="first", second_field="second"),
        X_l=np.ones((2, 2), dtype=np.float32),
        y_l=np.array([0, 1]),
        default_ema=False,
        seed=1,
    )

    assert bound.first.marker == "loaded"
    assert bound.second.marker == "loaded"
    assert bound.first is not bound.second


def test_single_binding_builds_a_real_torch_bundle() -> None:
    torch = pytest.importorskip("torch")
    method_cls = get_method_class("fixmatch")
    spec = build_method_spec(method_cls, require_spec=True)

    bound = bind_model_to_spec(
        spec,
        ModelBuildConfig(
            classifier_id="mlp",
            classifier_backend="torch",
            classifier_params={"hidden_sizes": [4]},
        ),
        binding=get_method_info("fixmatch").model_binding,
        X_l=torch.ones((4, 3), dtype=torch.float32),
        y_l=torch.tensor([0, 1, 0, 1]),
        default_ema=False,
        seed=9,
        strict=True,
    )

    assert bound.model_bundle is not None
    assert bound.model_bundle.model(torch.ones((2, 3))).shape == (2, 2)


@dataclass(frozen=True)
class _SharedHeadSpec:
    shared: Any | None = None
    heads: tuple[Any, ...] | None = None


def test_shared_head_binding_uses_method_owned_count_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    calls: list[dict[str, Any]] = []

    class SharedModel:
        def __call__(self, sample: Any) -> Any:
            batch_size = int(sample.shape[0])
            return {"feat": torch.ones((batch_size, 5), dtype=torch.float32)}

    def fake_builder(**kwargs: Any) -> Any:
        calls.append(kwargs)
        model = SharedModel() if len(calls) == 1 else object()
        return SimpleNamespace(model=model, seed=kwargs["seed"])

    monkeypatch.setattr(model_binding, "build_torch_bundle_from_classifier", fake_builder)
    binding = ModelBindingSpec.shared_heads(
        shared_field="shared",
        heads_field="heads",
        head_count=3,
        head_classifier_ids=("mlp", "logreg"),
        head_classifier_fallback="logreg",
    )
    bound = bind_model_to_spec(
        _SharedHeadSpec(),
        ModelBuildConfig(classifier_id="image_cnn", classifier_backend="torch", ema=True),
        binding=binding,
        X_l=torch.ones((4, 2), dtype=torch.float32),
        y_l=torch.tensor([0, 1, 0, 1]),
        default_ema=False,
        seed=23,
        strict=True,
    )

    assert bound.shared.seed == 23
    assert [head.seed for head in bound.heads] == [24, 25, 26]
    assert [call["classifier_id"] for call in calls] == [
        "image_cnn",
        "logreg",
        "logreg",
        "logreg",
    ]
    assert [call["ema"] for call in calls] == [True, False, False, False]


def test_shared_head_binding_probes_the_runtime_feature_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    head_samples: list[Any] = []

    class SharedModel:
        def __call__(self, sample: Any) -> Any:
            return torch.zeros((int(sample.shape[0]), 2), dtype=torch.float32)

        def forward_features(self, sample: Any) -> Any:
            return torch.ones((int(sample.shape[0]), 7), dtype=torch.float32)

    shared_model = SharedModel()
    calls = 0

    def fake_builder(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(
                model=shared_model,
                meta={"forward_features": shared_model.forward_features},
            )
        head_samples.append(kwargs["sample"])
        return SimpleNamespace(model=object(), meta=None)

    monkeypatch.setattr(model_binding, "build_torch_bundle_from_classifier", fake_builder)
    binding = ModelBindingSpec.shared_heads(
        shared_field="shared",
        heads_field="heads",
        head_count=3,
        head_classifier_ids=("mlp", "logreg"),
        head_classifier_fallback="logreg",
    )

    bind_model_to_spec(
        _SharedHeadSpec(),
        ModelBuildConfig(classifier_id="lstm_scratch", classifier_backend="torch"),
        binding=binding,
        X_l=torch.ones((4, 5), dtype=torch.float32),
        y_l=torch.tensor([0, 1, 0, 1]),
        default_ema=False,
        seed=23,
        strict=True,
    )

    assert len(head_samples) == 3
    assert all(tuple(sample.shape) == (1, 7) for sample in head_samples)


def test_trinet_lstm_heads_match_runtime_feature_width() -> None:
    torch = pytest.importorskip("torch")
    method_cls = get_method_class("trinet")
    binding = get_method_info("trinet").model_binding
    spec = build_method_spec(method_cls, require_spec=True)
    X_l = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0], [2.0, 1.0, 3.0]])

    bound = bind_model_to_spec(
        spec,
        ModelBuildConfig(
            classifier_id="lstm_scratch",
            classifier_backend="torch",
            classifier_params={
                "vocab_size": 8,
                "embed_dim": 4,
                "hidden_dim": 5,
                "bidirectional": False,
                "dropout": 0.0,
            },
            ema=False,
        ),
        binding=binding,
        X_l=X_l,
        y_l=torch.tensor([0, 1, 0, 1]),
        default_ema=False,
        seed=7,
        strict=True,
    )

    features = bound.shared_bundle.meta["forward_features"](X_l[:2])
    assert features.shape == (2, 5)
    assert all(head.model(features).shape == (2, 2) for head in bound.head_bundles)


def test_binding_rejects_undeclared_spec_shape() -> None:
    with pytest.raises(ModelBindingError, match="absent spec fields"):
        bind_model_to_spec(
            _ManyBundleSpec(),
            ModelBuildConfig(factory=_importable_test_factory, params={"marker": "x"}),
            binding=ModelBindingSpec.single("model_bundle"),
            X_l=np.ones((2, 2), dtype=np.float32),
            y_l=np.array([0, 1]),
            default_ema=False,
            seed=0,
        )


def test_binding_requires_a_prebound_dataclass_when_model_config_is_omitted() -> None:
    with pytest.raises(ModelBindingError, match="requires a native model bundle"):
        bind_model_to_spec(
            None,
            None,
            binding=ModelBindingSpec.single("first"),
            X_l=np.ones((2, 2), dtype=np.float32),
            y_l=np.array([0, 1]),
            default_ema=False,
            seed=0,
        )


def test_shared_head_binding_accepts_complete_prebound_bundles() -> None:
    spec = _SharedHeadSpec(shared=object(), heads=(object(), object()))
    binding = ModelBindingSpec.shared_heads(
        shared_field="shared",
        heads_field="heads",
        head_count=2,
        head_classifier_ids=("mlp",),
        head_classifier_fallback="mlp",
    )

    bound = bind_model_to_spec(
        spec,
        None,
        binding=binding,
        X_l=np.ones((2, 2), dtype=np.float32),
        y_l=np.array([0, 1]),
        default_ema=False,
        seed=0,
    )

    assert bound is spec
