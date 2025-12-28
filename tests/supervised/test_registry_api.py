from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from modssc.supervised.api import available_classifiers, classifier_info, create_classifier
from modssc.supervised.errors import (
    OptionalDependencyError,
    UnknownBackendError,
    UnknownClassifierError,
)
from modssc.supervised.registry import (
    _REGISTRY,
    BackendSpec,
    ClassifierSpec,
    get_backend_spec,
    get_spec,
    iter_specs,
    list_classifiers,
    register_backend,
    register_classifier,
)
from modssc.supervised.types import ClassifierRuntime


def test_available_classifiers_contains_expected() -> None:
    keys = {d["key"] for d in available_classifiers()}
    assert {"knn", "svm_rbf", "logreg"}.issubset(keys)


def test_list_classifiers_sorted() -> None:
    keys = list_classifiers()
    assert keys == sorted(keys)
    assert "knn" in keys


def test_classifier_info_smoke() -> None:
    info = classifier_info("knn")
    assert info["key"] == "knn"
    assert "backends" in info
    assert "numpy" in info["backends"]


def test_unknown_classifier_raises() -> None:
    with pytest.raises(UnknownClassifierError):
        classifier_info("does_not_exist")


def test_unknown_backend_raises() -> None:
    with pytest.raises(UnknownBackendError):
        create_classifier("knn", backend="does_not_exist")


def test_auto_backend_selection_can_fallback_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("modssc.supervised.api.has_module", lambda _: False)
    clf = create_classifier("knn", backend="auto", params={"k": 3})
    assert clf.backend == "numpy"


def test_auto_backend_selection_raises_when_no_backend_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("modssc.supervised.api.has_module", lambda _: False)
    with pytest.raises(OptionalDependencyError):
        create_classifier("svm_rbf", backend="auto")


def test_load_object_path_validation() -> None:
    api = importlib.import_module("modssc.supervised.api")
    with pytest.raises(ValueError):
        api._load_object("invalid_path_without_colon")  # type: ignore[attr-defined]


def test_auto_backend_selection_prefers_sklearn_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("modssc.supervised.api.has_module", lambda m: m == "sklearn")
    clf = create_classifier("knn", backend="auto")
    assert clf.backend == "sklearn"


def test_auto_backend_selection_handles_torch_extra() -> None:
    mock_spec = ClassifierSpec(
        key="mock_torch",
        description="Mock",
        backends={
            "torch": BackendSpec(
                backend="torch",
                factory="foo:bar",
                required_extra="supervised-torch",
            )
        },
        preferred_backends=("torch",),
    )

    with (
        patch("modssc.supervised.api.get_spec", return_value=mock_spec),
        patch(
            "modssc.supervised.api.get_backend_spec",
            return_value=mock_spec.backends["torch"],
        ),
        patch("modssc.supervised.api.has_module", return_value=True) as mock_has,
        patch("modssc.supervised.api._load_object") as mock_load,
    ):
        mock_load.return_value = MagicMock()
        create_classifier("mock_torch", backend="auto")
        mock_has.assert_called_with("torch")
        mock_load.assert_called_with("foo:bar")


def test_available_classifiers_filtering(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("modssc.supervised.api.has_module", lambda _: False)
    classifiers = available_classifiers(available_only=True)
    knn_entry = next((c for c in classifiers if c["key"] == "knn"), None)
    assert knn_entry is not None
    assert "sklearn" not in knn_entry["backends"]
    assert "numpy" in knn_entry["backends"]


def test_create_classifier_runtime_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockClassifier:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    spec = ClassifierSpec(
        key="mock_clf",
        description="Mock",
        backends={"mock": BackendSpec(backend="mock", factory="mod:MockClassifier")},
        preferred_backends=("mock",),
    )

    monkeypatch.setattr("modssc.supervised.api.get_spec", lambda _: spec)
    monkeypatch.setattr("modssc.supervised.api.get_backend_spec", lambda _c, _b: spec.backends[_b])
    monkeypatch.setattr("modssc.supervised.api._load_object", lambda _: MockClassifier)

    runtime = ClassifierRuntime(seed=123, n_jobs=4)
    clf = create_classifier("mock_clf", backend="auto", runtime=runtime)

    assert clf.kwargs["seed"] == 123
    assert clf.kwargs["n_jobs"] == 4

    clf2 = create_classifier("mock_clf", backend="auto", params={"seed": 999}, runtime=runtime)
    assert clf2.kwargs["seed"] == 999


def test_create_classifier_skips_missing_preferred_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ClassifierSpec(
        key="mock_clf",
        description="Mock",
        backends={},
        preferred_backends=("missing_backend",),
    )
    monkeypatch.setattr("modssc.supervised.api.get_spec", lambda _: spec)

    with pytest.raises(UnknownBackendError):
        create_classifier("mock_clf", backend="auto")


def test_registry_conflicts() -> None:
    key = "temp_clf"
    register_classifier(key=key, description="desc1")
    register_classifier(key=key, description="desc1")

    with pytest.raises(ValueError, match="different metadata"):
        register_classifier(key=key, description="desc2")

    if key in _REGISTRY:
        del _REGISTRY[key]


def test_register_backend_errors() -> None:
    with pytest.raises(UnknownClassifierError):
        register_backend(classifier_id="unknown_clf", backend="b", factory="f")

    key = "temp_clf_2"
    register_classifier(key=key, description="desc")
    register_backend(classifier_id=key, backend="b1", factory="f1")

    with pytest.raises(ValueError, match="Backend already registered"):
        register_backend(classifier_id=key, backend="b1", factory="f2")

    if key in _REGISTRY:
        del _REGISTRY[key]


def test_get_spec_errors() -> None:
    with pytest.raises(UnknownClassifierError):
        get_spec("unknown_clf_xyz")


def test_get_backend_spec_errors() -> None:
    key = "temp_clf_3"
    register_classifier(key=key, description="desc")

    with pytest.raises(UnknownBackendError):
        get_backend_spec(key, "unknown_backend")

    if key in _REGISTRY:
        del _REGISTRY[key]


def test_iter_specs() -> None:
    specs = list(iter_specs())
    assert len(specs) > 0


def test_get_backend_spec_success() -> None:
    spec = get_backend_spec("knn", "numpy")
    assert spec.backend == "numpy"


def test_create_classifier_explicit_backend() -> None:
    clf = create_classifier("knn", backend="numpy")
    assert clf.backend == "numpy"


def test_available_classifiers_custom_extra() -> None:
    mock_spec = ClassifierSpec(
        key="mock_clf",
        description="Mock",
        backends={
            "custom": BackendSpec(backend="custom", factory="foo:bar", required_extra="custom_pkg")
        },
        preferred_backends=("custom",),
    )

    with (
        patch("modssc.supervised.api.iter_specs", return_value=[mock_spec]),
        patch("modssc.supervised.api.has_module", return_value=True) as mock_has_module,
    ):
        clfs = available_classifiers(available_only=True)
        assert len(clfs) == 1
        assert "custom" in clfs[0]["backends"]
        mock_has_module.assert_called_with("custom_pkg")


def test_create_classifier_auto_missing_preferred() -> None:
    with patch("modssc.supervised.api._load_object") as mock_load:
        mock_load.return_value = MagicMock()

        mock_spec = ClassifierSpec(
            key="mock_clf",
            description="Mock",
            backends={
                "other": BackendSpec(backend="other", factory="foo:bar", required_extra=None)
            },
            preferred_backends=("missing", "other"),
        )

        with (
            patch("modssc.supervised.api.get_spec", return_value=mock_spec),
            patch(
                "modssc.supervised.api.get_backend_spec",
                side_effect=lambda cid, b: mock_spec.backends[b],
            ),
        ):
            create_classifier("mock_clf", backend="auto")
            mock_load.assert_called_with("foo:bar")


def test_available_classifiers_all() -> None:
    clfs = available_classifiers(available_only=False)
    assert len(clfs) > 0
