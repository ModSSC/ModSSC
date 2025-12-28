import logging
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from modssc.cli import datasets as datasets_mod
from modssc.cli.datasets import app
from modssc.data_loader.errors import DataLoaderError

runner = CliRunner()


def test_list_datasets_with_modalities():
    with patch("modssc.cli.datasets.api") as mock_api:
        mock_api.available_datasets.return_value = ["d1", "d2"]
        mock_api.dataset_info.side_effect = lambda k: MagicMock(
            modality="graph" if k == "d1" else "text"
        )

        result = runner.invoke(app, ["list", "--modalities", "graph"])
        assert result.exit_code == 0
        assert "d1" in result.stdout
        assert "d2" not in result.stdout


def test_download_all_missing_extras():
    with patch("modssc.cli.datasets.api") as mock_api:
        report = MagicMock()
        report.summary.return_value = "Summary"
        report.missing_extras = {"extra1": ["d1"]}
        report.failed = {}
        mock_api.download_all_datasets.return_value = report

        result = runner.invoke(app, ["download", "--all"])
        assert result.exit_code == 0
        assert "Missing extras:" in result.stdout
        assert "extra1: d1" in result.stdout


def test_download_all_failed():
    with patch("modssc.cli.datasets.api") as mock_api:
        report = MagicMock()
        report.summary.return_value = "Summary"
        report.missing_extras = {}
        report.failed = {"d1": "Error"}
        mock_api.download_all_datasets.return_value = report

        result = runner.invoke(app, ["download", "--all"])
        assert result.exit_code == 2
        assert "Failed:" in result.stdout
        assert "d1: Error" in result.stdout


def test_providers_command():
    result = runner.invoke(app, ["providers"])
    assert result.exit_code == 0
    assert "toy" in result.stdout.lower()


def test_download_all_prints_sections(tmp_path):
    from modssc.data_loader.types import DownloadReport

    report = DownloadReport(
        downloaded=["toy"],
        skipped_already_cached=["iris"],
        skipped_missing_extras=["cifar10"],
        missing_extras={"vision": ["cifar10"]},
        failed={"ag_news": "ReadTimeout: timed out"},
    )

    with patch("modssc.cli.datasets.api") as mock_api:
        mock_api.download_all_datasets.return_value = report

        result = runner.invoke(app, ["download", "--all", "--cache-dir", str(tmp_path)])
        assert result.exit_code == 2
        out = result.stdout

        assert "Downloaded: 1" in out
        assert "Skipped (already cached): 1" in out
        assert "Skipped (missing extras): 1" in out
        assert "Failed: 1" in out

        assert "Missing extras:" in out
        assert "vision: cifar10" in out
        assert "Failed:" in out
        assert "ag_news: ReadTimeout: timed out" in out


def test_download_requires_dataset_id() -> None:
    result = runner.invoke(app, ["download"])
    assert result.exit_code != 0
    # Use result.output which contains combined stdout/stderr output
    # Check for key parts of the error message (ANSI codes may break up the exact string)
    assert "Provide" in result.output and "--dataset" in result.output


def test_download_handles_data_loader_error() -> None:
    runner = CliRunner()
    with patch("modssc.cli.datasets.api.download_dataset", side_effect=DataLoaderError("boom")):
        result = runner.invoke(app, ["download", "--dataset", "toy"])
    assert result.exit_code == 2
    assert "boom" in result.output


def test_providers_with_log_level() -> None:
    result = runner.invoke(app, ["providers", "--log-level", "basic"])
    assert result.exit_code == 0


def test_list_datasets_modalities_error_with_log_level() -> None:
    with patch("modssc.cli.datasets.api") as mock_api:
        mock_api.available_datasets.return_value = ["d1"]
        mock_api.dataset_info.side_effect = DataLoaderError("boom")
        result = runner.invoke(app, ["list", "--modalities", "vision", "--log-level", "detailed"])
    assert result.exit_code == 2
    assert "boom" in result.output


def test_list_datasets_modalities_error_default_log_level() -> None:
    with patch("modssc.cli.datasets.api") as mock_api:
        mock_api.available_datasets.return_value = ["d1"]
        mock_api.dataset_info.side_effect = DataLoaderError("boom")
        result = runner.invoke(app, ["list", "--modalities", "vision"])
    assert result.exit_code == 2
    assert "boom" in result.output


def test_info_handles_data_loader_error_with_log_level() -> None:
    with patch("modssc.cli.datasets.api.dataset_info", side_effect=DataLoaderError("boom")):
        result = runner.invoke(app, ["info", "--dataset", "bad", "--log-level", "detailed"])
    assert result.exit_code == 2
    assert "boom" in result.output


def test_info_handles_data_loader_error_default_log_level() -> None:
    with patch("modssc.cli.datasets.api.dataset_info", side_effect=DataLoaderError("boom")):
        result = runner.invoke(app, ["info", "--dataset", "bad"])
    assert result.exit_code == 2
    assert "boom" in result.output


def test_download_handles_data_loader_error_with_log_level() -> None:
    with patch("modssc.cli.datasets.api.download_dataset", side_effect=DataLoaderError("boom")):
        result = runner.invoke(app, ["download", "--dataset", "toy", "--log-level", "detailed"])
    assert result.exit_code == 2
    assert "boom" in result.output


def test_cache_commands_with_log_level(tmp_path) -> None:
    with (
        patch(
            "modssc.cli.datasets.api.cache.index_list",
            return_value=[{"fingerprint": "fp", "canonical_uri": "toy://x", "size_bytes": 0}],
        ),
        patch("modssc.cli.datasets.api.cache.purge_dataset", return_value=[]),
        patch("modssc.cli.datasets.api.cache.gc_keep_latest", return_value=[]),
    ):
        result = runner.invoke(
            app, ["cache", "ls", "--cache-dir", str(tmp_path), "--log-level", "basic"]
        )
        assert result.exit_code == 0


def test_list_datasets_logs_exception_on_debug(monkeypatch) -> None:
    def boom(_):
        raise DataLoaderError("boom")

    monkeypatch.setattr(datasets_mod.api, "available_datasets", lambda: ["d1"])
    monkeypatch.setattr(datasets_mod.api, "dataset_info", boom)

    prev_level = datasets_mod.logger.level
    datasets_mod.logger.setLevel(logging.DEBUG)
    try:
        with pytest.raises(typer.Exit):
            datasets_mod.list_datasets(modalities=["vision"], log_level=None)
    finally:
        datasets_mod.logger.setLevel(prev_level)


def test_info_logs_exception_on_debug(monkeypatch) -> None:
    monkeypatch.setattr(
        datasets_mod.api,
        "dataset_info",
        lambda *_: (_ for _ in ()).throw(DataLoaderError("boom")),
    )
    prev_level = datasets_mod.logger.level
    datasets_mod.logger.setLevel(logging.DEBUG)
    try:
        with pytest.raises(typer.Exit):
            datasets_mod.info(dataset_id="bad", log_level=None)
    finally:
        datasets_mod.logger.setLevel(prev_level)


def test_cache_purge_and_gc_direct(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(datasets_mod.api, "_layout", lambda *_: "layout")
    monkeypatch.setattr(datasets_mod.api.cache, "purge_dataset", lambda *_: [])
    monkeypatch.setattr(datasets_mod.api.cache, "purge_fingerprint", lambda *_: None)
    monkeypatch.setattr(datasets_mod.api.cache, "gc_keep_latest", lambda *_: [])

    datasets_mod.cache_purge("toy", cache_dir=tmp_path, fingerprint=False, log_level="basic")
    datasets_mod.cache_purge("fp", cache_dir=tmp_path, fingerprint=True, log_level="basic")
    datasets_mod.cache_gc(keep_latest=True, cache_dir=tmp_path, log_level="basic")
