from unittest.mock import MagicMock, patch

import numpy as np

from modssc.data_loader.providers.hf import (
    HuggingFaceDatasetsProvider,
    _apply_class_filter,
    _apply_limits,
    _extract_official_test,
    _limit_samples,
    _normalize_filter,
    _split_ref,
)
from modssc.data_loader.types import DatasetIdentity
from modssc.data_loader.uri import ParsedURI


def test_split_ref_with_slash():
    name, cfg = _split_ref("dataset/config")
    assert name == "dataset"
    assert cfg == "config"


def test_split_ref_with_slash_empty_config():
    name, cfg = _split_ref("dataset/   ")
    assert name == "dataset"
    assert cfg is None


def test_split_ref_no_slash():
    name, cfg = _split_ref("dataset")
    assert name == "dataset"
    assert cfg is None


def test_normalize_filter_variants():
    assert _normalize_filter(None) is None
    assert set(_normalize_filter({1, 2})) == {1, 2}
    assert _normalize_filter("x") == ["x"]


def test_apply_class_filter_and_limit_samples():
    X = np.array(["a", "b", "c"])
    y = np.array([0, 1, 0])
    X_f, y_f = _apply_class_filter(X, y, class_filter=[0])
    assert X_f.tolist() == ["a", "c"]
    assert y_f.tolist() == [0, 0]

    X_empty, y_empty = _limit_samples(X, y, max_samples=0, seed=None)
    assert X_empty.size == 0
    assert y_empty.size == 0

    X_take_no_seed, y_take_no_seed = _limit_samples(X, y, max_samples=2, seed=None)
    assert X_take_no_seed.shape == (2,)
    assert y_take_no_seed.shape == (2,)

    X_take, y_take = _limit_samples(X, y, max_samples=2, seed=123)
    assert X_take.shape == (2,)
    assert y_take.shape == (2,)


def test_apply_limits_none_split():
    assert _apply_limits(None, class_filter=None, max_samples=None, seed=None) is None


def test_resolve_with_config_and_defaults():
    provider = HuggingFaceDatasetsProvider()
    parsed = ParsedURI(provider="hf", reference="ag_news/plain_text")

    identity = provider.resolve(parsed, options={})

    assert identity.canonical_uri == "hf:ag_news/plain_text"
    assert identity.dataset_id == "ag_news"
    assert identity.resolved_kwargs["config"] == "plain_text"
    assert identity.resolved_kwargs["text_column"] == "text"
    assert identity.resolved_kwargs["label_column"] == "label"
    assert identity.resolved_kwargs["prefer_test_split"] is True


def test_resolve_with_option_overrides():
    provider = HuggingFaceDatasetsProvider()
    parsed = ParsedURI(provider="hf", reference="imdb")

    identity = provider.resolve(
        parsed,
        options={
            "config": "tiny",
            "text_column": "content",
            "label_column": "target",
            "prefer_test_split": False,
            "modality": "textual",
            "task": "sentiment",
        },
    )

    assert identity.canonical_uri == "hf:imdb/tiny"
    assert identity.modality == "textual"
    assert identity.task == "sentiment"
    assert identity.resolved_kwargs["config"] == "tiny"
    assert identity.resolved_kwargs["text_column"] == "content"
    assert identity.resolved_kwargs["label_column"] == "target"
    assert identity.resolved_kwargs["prefer_test_split"] is False


def test_extract_official_test_not_dictlike():
    ds = 123
    result = _extract_official_test(ds, "text", "label", prefer_test=True)
    assert result is None


def test_extract_official_test_no_candidates():
    ds = {"train": {"text": [], "label": []}}
    result = _extract_official_test(ds, "text", "label", prefer_test=True)
    assert result is None


def test_extract_official_test_prefer_test():
    ds = {"test": {"text": ["t1"], "label": [0]}, "validation": {"text": ["val_a"], "label": [1]}}

    result = _extract_official_test(ds, "text", "label", prefer_test=True)
    assert result.X[0] == "t1"


def test_extract_official_test_prefer_validation():
    ds = {"test": {"text": ["t1"], "label": [0]}, "validation": {"text": ["val_a"], "label": [1]}}

    result = _extract_official_test(ds, "text", "label", prefer_test=False)
    assert result.X[0] == "val_a"


def test_load_canonical_with_config(tmp_path):
    provider = HuggingFaceDatasetsProvider()
    identity = DatasetIdentity(
        provider="hf",
        canonical_uri="hf:dataset/config",
        dataset_id="dataset",
        version=None,
        modality="text",
        task="classification",
        resolved_kwargs={
            "name": "dataset",
            "config": "my_config",
            "text_column": "text",
            "label_column": "label",
            "prefer_test_split": True,
        },
    )

    with patch("modssc.data_loader.providers.hf.optional_import") as mock_import:
        mock_datasets = MagicMock()
        mock_import.return_value = mock_datasets

        mock_ds = {
            "train": {"text": ["train_text"], "label": [0]},
            "test": {"text": ["test_text"], "label": [1]},
        }
        mock_datasets.load_dataset.return_value = mock_ds

        ds = provider.load_canonical(identity, raw_dir=tmp_path)

        mock_datasets.load_dataset.assert_called_with(
            "dataset", "my_config", cache_dir=str(tmp_path)
        )

        assert ds.train.X[0] == "train_text"
        assert ds.test.X[0] == "test_text"
        assert ds.meta["config"] == "my_config"


def test_load_canonical_without_config(tmp_path):
    provider = HuggingFaceDatasetsProvider()
    identity = DatasetIdentity(
        provider="hf",
        canonical_uri="hf:dataset",
        dataset_id="dataset",
        version=None,
        modality="text",
        task="classification",
        resolved_kwargs={
            "name": "dataset",
            "config": None,
            "text_column": "text",
            "label_column": "label",
            "prefer_test_split": True,
        },
    )

    with patch("modssc.data_loader.providers.hf.optional_import") as mock_import:
        mock_datasets = MagicMock()
        mock_import.return_value = mock_datasets

        mock_ds = {
            "train": {"text": ["train_text"], "label": [0]},
            "validation": {"text": ["val_text"], "label": [1]},
        }
        mock_datasets.load_dataset.return_value = mock_ds

        ds = provider.load_canonical(identity, raw_dir=tmp_path)

        mock_datasets.load_dataset.assert_called_with("dataset", cache_dir=str(tmp_path))
        assert ds.test is not None
        assert ds.test.X[0] == "val_text"
        assert ds.meta["config"] is None
