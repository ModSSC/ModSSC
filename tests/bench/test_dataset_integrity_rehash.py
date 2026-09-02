from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench.orchestrators import dataset as dataset_orch
from bench.schema import BenchConfigError, DatasetConfig, DatasetIntegrityConfig
from modssc.data_loader.errors import ManifestError
from modssc.data_loader.types import LoadedDataset, Split


def _config(tmp_path: Path) -> DatasetConfig:
    return DatasetConfig(
        id="authenticated",
        options={"fold": 1},
        download=False,
        cache_dir=str(tmp_path),
        integrity=DatasetIntegrityConfig(
            fingerprint="a" * 64,
            content_sha256="b" * 64,
            content_manifest_sha256="c" * 64,
        ),
    )


def _dataset() -> LoadedDataset:
    return LoadedDataset(
        train=Split(
            X=np.arange(8, dtype=np.float32).reshape(4, 2),
            y=np.array([0, 1, 0, 1], dtype=np.int64),
        ),
        meta={
            "dataset_fingerprint": "stale",
            "dataset_content_sha256": "stale",
        },
    )


def test_declared_dataset_integrity_rehashes_cached_bytes_before_use(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(dataset_orch, "load_dataset", lambda *args, **kwargs: _dataset())
    monkeypatch.setattr(
        dataset_orch,
        "dataset_info",
        lambda _dataset_id: SimpleNamespace(as_dict=lambda: {"provider": "test"}),
    )

    def verify(dataset_id: str, **kwargs: object) -> dict[str, str]:
        calls.append({"dataset_id": dataset_id, **kwargs})
        return {
            "cache_fingerprint": "a" * 64,
            "content_sha256": "b" * 64,
            "content_manifest_sha256": "c" * 64,
            "cache_state_sha256": "d" * 64,
        }

    monkeypatch.setattr(dataset_orch, "verify_dataset_content", verify)

    loaded, _info = dataset_orch.load(cfg)

    expected_call = {
        "dataset_id": "authenticated",
        "cache_dir": tmp_path.resolve(),
        "options": {"fold": 1},
        "rehash": True,
    }
    assert calls == [expected_call]
    assert loaded.meta == {
        "dataset_fingerprint": "a" * 64,
        "dataset_cache_fingerprint": "a" * 64,
        "dataset_content_sha256": "b" * 64,
        "dataset_content_manifest_sha256": "c" * 64,
        "dataset_content_state_sha256": "d" * 64,
        "dataset_content_rehashed": True,
    }
    dataset_orch.verify_integrity(loaded, cfg)

    assert dataset_orch.revalidate_integrity(loaded, cfg) == {
        "cache_fingerprint": "a" * 64,
        "cache_state_sha256": "d" * 64,
        "content_manifest_sha256": "c" * 64,
        "content_sha256": "b" * 64,
    }
    assert calls == [expected_call, expected_call]


def test_dataset_rehash_rejects_incomplete_native_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    monkeypatch.setattr(dataset_orch, "load_dataset", lambda *args, **kwargs: _dataset())
    monkeypatch.setattr(
        dataset_orch,
        "verify_dataset_content",
        lambda *args, **kwargs: {"content_sha256": "b" * 64},
    )

    with pytest.raises(BenchConfigError) as raised:
        dataset_orch.load(cfg)

    assert raised.value.code == "E_BENCH_DATASET_INTEGRITY"
    assert "incomplete evidence" in str(raised.value)


def test_dataset_rehash_failure_is_a_typed_benchmark_integrity_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    monkeypatch.setattr(dataset_orch, "load_dataset", lambda *args, **kwargs: _dataset())
    monkeypatch.setattr(
        dataset_orch,
        "verify_dataset_content",
        lambda *args, **kwargs: (_ for _ in ()).throw(ManifestError("digest differs")),
    )

    with pytest.raises(BenchConfigError) as raised:
        dataset_orch.load(cfg)

    assert raised.value.code == "E_BENCH_DATASET_INTEGRITY"
    assert "digest differs" in str(raised.value)


def test_final_dataset_rehash_detects_a_mid_run_content_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    monkeypatch.setattr(dataset_orch, "load_dataset", lambda *args, **kwargs: _dataset())
    monkeypatch.setattr(
        dataset_orch,
        "dataset_info",
        lambda _dataset_id: SimpleNamespace(as_dict=lambda: {"provider": "test"}),
    )
    evidence = {
        "cache_fingerprint": "a" * 64,
        "content_sha256": "b" * 64,
        "content_manifest_sha256": "c" * 64,
        "cache_state_sha256": "d" * 64,
    }
    calls = 0

    def verify(*args: object, **kwargs: object) -> dict[str, str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return evidence
        raise ManifestError("content changed during execution")

    monkeypatch.setattr(dataset_orch, "verify_dataset_content", verify)
    loaded, _info = dataset_orch.load(cfg)

    with pytest.raises(BenchConfigError, match="content changed during execution"):
        dataset_orch.revalidate_integrity(loaded, cfg)


def test_dataset_without_integrity_contract_keeps_the_fast_load_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = DatasetConfig(id="toy", integrity=None)
    expected = _dataset()
    monkeypatch.setattr(dataset_orch, "load_dataset", lambda *args, **kwargs: expected)
    monkeypatch.setattr(
        dataset_orch,
        "dataset_info",
        lambda _dataset_id: SimpleNamespace(as_dict=lambda: {"provider": "test"}),
    )
    monkeypatch.setattr(
        dataset_orch,
        "verify_dataset_content",
        lambda *args, **kwargs: pytest.fail("undeclared integrity must not trigger rehash"),
    )

    loaded, _info = dataset_orch.load(cfg)

    assert loaded is expected
