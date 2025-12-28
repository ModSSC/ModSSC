import pytest

from modssc.preprocess.store import ArtifactStore


def test_artifact_store_coverage():
    store = ArtifactStore()
    store.set("a", 1)

    with pytest.raises(KeyError, match="Missing required artifact"):
        store.require("missing_key")

    store_copy = store.copy()
    assert store_copy.get("a") == 1
    assert store_copy is not store
    assert store_copy.data is not store.data

    assert store["a"] == 1
    with pytest.raises(KeyError):
        _ = store["missing"]

    assert "a" in store
    assert "missing" not in store

    assert list(store) == ["a"]

    assert len(store) == 1

    assert store.keys() == ["a"]

    assert store.has("a")
    assert not store.has("b")

    assert store.get("b", 2) == 2
