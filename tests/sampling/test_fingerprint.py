from __future__ import annotations

from modssc.sampling.fingerprint import derive_seed, stable_hash


def test_stable_hash_order_independent() -> None:
    a = stable_hash({"b": 1, "a": 2})
    b = stable_hash({"a": 2, "b": 1})
    assert a == b


def test_derive_seed_is_stable() -> None:
    assert derive_seed(0, "split") == derive_seed(0, "split")
    assert derive_seed(0, "split") != derive_seed(0, "labeling")
