import pytest

from modssc.data_loader.catalog import _merge
from modssc.data_loader.types import DatasetSpec


def test_merge_duplicate_keys_raises():
    c1 = {
        "a": DatasetSpec(
            key="a", modality="tabular", provider="p", uri="u", task="t", description="d"
        )
    }
    c2 = {
        "a": DatasetSpec(
            key="a", modality="vision", provider="p", uri="u", task="t", description="d"
        )
    }

    with pytest.raises(ValueError) as exc:
        _merge(c1, c2)
    assert "Duplicate dataset keys" in str(exc.value)
    assert "a" in str(exc.value)


def test_merge_success():
    c1 = {
        "a": DatasetSpec(
            key="a", modality="tabular", provider="p", uri="u", task="t", description="d"
        )
    }
    c2 = {
        "b": DatasetSpec(
            key="b", modality="vision", provider="p", uri="u", task="t", description="d"
        )
    }

    merged = _merge(c1, c2)
    assert len(merged) == 2
    assert "a" in merged
    assert "b" in merged
