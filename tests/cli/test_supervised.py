from __future__ import annotations

from typer.testing import CliRunner

from modssc.cli.supervised import app


def test_supervised_list_with_log_level(monkeypatch) -> None:
    items = [
        {
            "key": "knn",
            "description": "KNN baseline",
            "backends": {"sklearn": {"required_extra": "sklearn"}, "numpy": {}},
        }
    ]
    monkeypatch.setattr("modssc.cli.supervised.available_classifiers", lambda **_: items)
    runner = CliRunner()
    result = runner.invoke(app, ["list", "--log-level", "basic"])
    assert result.exit_code == 0
    assert "knn: KNN baseline" in result.stdout
    assert "extra=sklearn" in result.stdout


def test_supervised_info_with_log_level(monkeypatch) -> None:
    monkeypatch.setattr(
        "modssc.cli.supervised.classifier_info", lambda *_: {"classifier_id": "knn"}
    )
    runner = CliRunner()
    result = runner.invoke(app, ["info", "knn", "--log-level", "basic"])
    assert result.exit_code == 0
    assert '"classifier_id": "knn"' in result.stdout


def test_supervised_list_json(monkeypatch) -> None:
    items = [
        {
            "key": "knn",
            "description": "KNN baseline",
            "backends": {},
        }
    ]
    monkeypatch.setattr("modssc.cli.supervised.available_classifiers", lambda **_: items)
    runner = CliRunner()
    result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0
    assert '"key": "knn"' in result.stdout


def test_supervised_info_without_log_level(monkeypatch) -> None:
    monkeypatch.setattr(
        "modssc.cli.supervised.classifier_info", lambda *_: {"classifier_id": "knn"}
    )
    runner = CliRunner()
    result = runner.invoke(app, ["info", "knn"])
    assert result.exit_code == 0
