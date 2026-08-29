from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.preprocess.errors import OptionalDependencyError, PreprocessValidationError
from modssc.preprocess.steps.text.count_vectorizer import (
    CountVectorizerStep,
    _texts,
    _visible_text,
)
from modssc.preprocess.store import ArtifactStore


def test_count_vectorizer_fits_selected_rows_and_returns_dense_counts() -> None:
    module = MagicMock()
    vectorizer = MagicMock()
    sparse = MagicMock()
    sparse.toarray.return_value = np.asarray([[1, 0], [0, 2]], dtype=np.int64)
    module.CountVectorizer.return_value = vectorizer
    vectorizer.fit.return_value = vectorizer
    vectorizer.transform.return_value = sparse
    store = ArtifactStore()
    store.set("raw.X", np.asarray([["alpha"], ["beta"], ["gamma"]], dtype=object))

    with patch("modssc.preprocess.steps.text.count_vectorizer.require", return_value=module):
        step = CountVectorizerStep(dense=True, lowercase=False, binary=True)
        step.fit(store, fit_indices=np.asarray([0, 2]), rng=np.random.default_rng(0))
        result = step.transform(store, rng=np.random.default_rng(0))

    vectorizer.fit.assert_called_once_with(["alpha", "gamma"])
    assert result["features.X"].dtype == np.float32
    np.testing.assert_array_equal(result["features.X"], np.asarray([[1, 0], [0, 2]]))
    kwargs = module.CountVectorizer.call_args.kwargs
    assert kwargs["lowercase"] is False
    assert kwargs["binary"] is True
    assert kwargs["dtype"] is np.float32


def test_count_vectorizer_sparse_and_validation_paths() -> None:
    module = MagicMock()
    vectorizer = MagicMock()
    sparse = object()
    module.CountVectorizer.return_value = vectorizer
    vectorizer.fit.return_value = vectorizer
    vectorizer.transform.return_value = sparse
    store = ArtifactStore()
    store.set("raw.X", ["one", "two"])

    with patch("modssc.preprocess.steps.text.count_vectorizer.require", return_value=module):
        step = CountVectorizerStep(dense=False)
        with pytest.raises(PreprocessValidationError, match="before fit"):
            step.transform(store, rng=np.random.default_rng(0))
        with pytest.raises(PreprocessValidationError, match="must be 1D"):
            step.fit(store, fit_indices=np.asarray([[0]]), rng=np.random.default_rng(0))
        with pytest.raises(PreprocessValidationError, match="outside raw.X"):
            step.fit(store, fit_indices=np.asarray([2]), rng=np.random.default_rng(0))
        step.fit(store, fit_indices=np.asarray([0]), rng=np.random.default_rng(0))
        assert step.transform(store, rng=np.random.default_rng(0))["features.X"] is sparse

    with (
        patch(
            "modssc.preprocess.steps.text.count_vectorizer.require",
            side_effect=OptionalDependencyError("missing"),
        ),
        pytest.raises(OptionalDependencyError),
    ):
        CountVectorizerStep().fit(store, fit_indices=np.asarray([0]), rng=np.random.default_rng(0))


def test_count_vectorizer_text_normalization() -> None:
    assert _texts(np.asarray(["a", "b"], dtype=object)) == ["a", "b"]
    assert _texts(("c", "d")) == ["c", "d"]
    assert _texts(value for value in (1, 2)) == ["1", "2"]


def test_visible_text_removes_mime_markup_and_non_visible_content() -> None:
    raw = (
        "Content-Type: text/html\r\nMIME-Version: 1.0\r\n\r\n"
        "<html><head><style>.hidden { color: red; }</style><title>A &amp; B</title></head>"
        '<body class="secret"><script>ignored()</script><a href="hidden-url">Course</a>'
        " page</body></html>"
    )

    assert _visible_text(raw) == "A & B Course  page"
    assert _texts(np.asarray([[raw]], dtype=object), strip_html=True) == ["A & B Course  page"]


def test_visible_text_keeps_plain_text_and_non_mime_prefix() -> None:
    assert _visible_text("plain words") == "plain words"
    assert _visible_text("Title\n\n<body>Body</body>") == "Title\n\n Body"


def test_count_vectorizer_can_use_visible_html_text() -> None:
    sklearn_text = pytest.importorskip("sklearn.feature_extraction.text")
    assert sklearn_text is not None
    store = ArtifactStore()
    store.set(
        "raw.X",
        np.asarray(
            [
                '<html><body class="metadata">visible alpha</body></html>',
                "<style>invisible</style><p>visible beta</p>",
            ],
            dtype=object,
        ),
    )
    step = CountVectorizerStep(dense=True, strip_html=True)
    step.fit(store, fit_indices=np.asarray([0, 1]), rng=np.random.default_rng(0))

    transformed = step.transform(store, rng=np.random.default_rng(0))

    assert transformed["features.X"].shape == (2, 3)
    assert set(step._vec.vocabulary_) == {"alpha", "beta", "visible"}
