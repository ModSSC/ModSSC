from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.preprocess.errors import OptionalDependencyError, PreprocessValidationError
from modssc.preprocess.steps.text.tfidf import _as_text_array
from modssc.preprocess.store import ArtifactStore


def test_ensure_strings_step():
    from modssc.preprocess.steps.text.ensure_strings import EnsureStringsStep

    step = EnsureStringsStep()
    store = ArtifactStore()
    rng = np.random.default_rng(42)

    store.set("raw.X", np.array([1, 2, 3]))
    res = step.transform(store, rng=rng)
    assert res["raw.X"] == ["1", "2", "3"]

    store.set("raw.X", [10, 20])
    res = step.transform(store, rng=rng)
    assert res["raw.X"] == ["10", "20"]


def test_hash_tokenizer_step():
    from modssc.preprocess.steps.text.hash_tokenizer import HashTokenizerStep

    step = HashTokenizerStep(vocab_size=100, max_length=5)
    store = ArtifactStore()
    rng = np.random.default_rng(42)

    store.set("raw.X", ["hello world", "test"])
    res = step.transform(store, rng=rng)

    ids = res["tokens.input_ids"]
    mask = res["tokens.attention_mask"]

    assert ids.shape == (2, 5)
    assert mask.shape == (2, 5)
    assert mask[0, 0] == 1
    assert mask[0, 1] == 1
    assert mask[0, 2] == 0

    store.set("raw.X", ["", "   "])
    res = step.transform(store, rng=rng)
    assert np.all(res["tokens.attention_mask"] == 0)

    step_no_lower = HashTokenizerStep(vocab_size=100, max_length=5, lowercase=False)
    store.set("raw.X", ["Hello"])
    res = step_no_lower.transform(store, rng=rng)

    assert res["tokens.input_ids"].shape == (1, 5)

    store.set("raw.X", np.array("Hello world"))
    res = step.transform(store, rng=rng)
    assert res["tokens.input_ids"].shape == (1, 5)

    store.set("raw.X", np.array(["a b", "c"]))
    res = step.transform(store, rng=rng)
    assert res["tokens.input_ids"].shape == (2, 5)

    store.set("raw.X", (t for t in ["a b", "c"]))
    res = step.transform(store, rng=rng)
    assert res["tokens.input_ids"].shape == (2, 5)


def test_sentence_transformer_step():
    with patch("modssc.preprocess.steps.text.sentence_transformer.load_encoder") as mock_load:
        from modssc.preprocess.steps.text.sentence_transformer import SentenceTransformerStep

        mock_encoder = MagicMock()
        mock_load.return_value = mock_encoder
        mock_encoder.encode.return_value = np.zeros((2, 10), dtype=np.float32)

        step = SentenceTransformerStep(device="cpu")
        store = ArtifactStore()
        rng = np.random.default_rng(42)
        store.set("raw.X", ["a", "b"])

        res = step.transform(store, rng=rng)
        assert res["features.X"].shape == (2, 10)
        step.transform(store, rng=rng)
        assert mock_load.call_count == 1
        mock_load.assert_called_with("st:all-MiniLM-L6-v2", device="cpu")

        step = SentenceTransformerStep(device=None)
        step.transform(store, rng=rng)
        mock_load.assert_called_with("st:all-MiniLM-L6-v2")


def test_tfidf_step():
    mock_sklearn = MagicMock()
    mock_vec = MagicMock()
    mock_sklearn.TfidfVectorizer.return_value = mock_vec

    with (
        patch.dict("sys.modules", {"sklearn.feature_extraction.text": mock_sklearn}),
        patch("modssc.preprocess.steps.text.tfidf.require") as mock_require,
    ):
        mock_require.return_value = mock_sklearn

        from modssc.preprocess.steps.text.tfidf import TfidfStep

        step = TfidfStep()
        store = ArtifactStore()
        rng = np.random.default_rng(42)
        store.set("raw.X", ["doc1", "doc2", "doc3"])

        mock_vec.fit.return_value = mock_vec
        step.fit(store, fit_indices=np.array([0, 2]), rng=rng)
        mock_vec.fit.assert_called()

        args, _ = mock_vec.fit.call_args
        assert args[0] == ["doc1", "doc3"]

        mock_vec.transform.return_value = "sparse_matrix"
        res = step.transform(store, rng=rng)
        assert res["features.X"] == "sparse_matrix"

        step_unfit = TfidfStep()
        with pytest.raises(PreprocessValidationError, match="called before fit"):
            step_unfit.transform(store, rng=rng)

        with pytest.raises(PreprocessValidationError, match="must be 1D"):
            step.fit(store, fit_indices=np.array([[0]]), rng=rng)

        mock_require.side_effect = OptionalDependencyError("missing")
        with pytest.raises(OptionalDependencyError):
            step.fit(store, fit_indices=np.array([0]), rng=rng)


def test_tfidf_as_text_array_variants():
    arr = _as_text_array(np.array(["a", "b"]))
    assert arr.tolist() == ["a", "b"]

    arr = _as_text_array(t for t in ["x", "y"])
    assert arr.tolist() == ["x", "y"]
