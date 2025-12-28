import json

import numpy as np
from typer.testing import CliRunner

from modssc.cli.evaluation import DEFAULT_METRICS, app
from modssc.evaluation import evaluate, list_metrics

runner = CliRunner()


def test_list_json():
    result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["metrics"] == list_metrics()


def test_list_text():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines == list_metrics()


def test_compute_json(tmp_path):
    y_true = np.array([0, 1, 1, 0])
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.6, 0.4]])
    y_true_path = tmp_path / "y_true.npy"
    y_pred_path = tmp_path / "y_pred.npy"
    np.save(y_true_path, y_true)
    np.save(y_pred_path, scores)

    result = runner.invoke(
        app,
        [
            "compute",
            "--y-true",
            str(y_true_path),
            "--y-pred",
            str(y_pred_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    expected = evaluate(y_true, scores, list(DEFAULT_METRICS))
    assert payload == expected


def test_compute_text(tmp_path):
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_true_path = tmp_path / "y_true.npy"
    y_pred_path = tmp_path / "y_pred.npy"
    np.save(y_true_path, y_true)
    np.save(y_pred_path, y_pred)

    result = runner.invoke(
        app,
        [
            "compute",
            "--y-true",
            str(y_true_path),
            "--y-pred",
            str(y_pred_path),
        ],
    )

    assert result.exit_code == 0
    assert "accuracy:" in result.stdout


def test_compute_rejects_non_npy(tmp_path):
    y_true_path = tmp_path / "y_true.txt"
    y_pred_path = tmp_path / "y_pred.npy"
    y_true_path.write_text("not npy")
    np.save(y_pred_path, np.array([0, 1]))

    result = runner.invoke(
        app,
        [
            "compute",
            "--y-true",
            str(y_true_path),
            "--y-pred",
            str(y_pred_path),
        ],
    )

    assert result.exit_code != 0
    assert "Only .npy is supported" in result.output


def test_compute_rejects_invalid_npy(tmp_path):
    y_true_path = tmp_path / "y_true.npy"
    y_pred_path = tmp_path / "y_pred.npy"
    y_true_path.write_text("not a real npy")
    np.save(y_pred_path, np.array([0, 1]))

    result = runner.invoke(
        app,
        [
            "compute",
            "--y-true",
            str(y_true_path),
            "--y-pred",
            str(y_pred_path),
        ],
    )

    assert result.exit_code != 0
    assert "Failed to load" in result.output


def test_compute_unknown_metric(tmp_path):
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_true_path = tmp_path / "y_true.npy"
    y_pred_path = tmp_path / "y_pred.npy"
    np.save(y_true_path, y_true)
    np.save(y_pred_path, y_pred)

    result = runner.invoke(
        app,
        [
            "compute",
            "--y-true",
            str(y_true_path),
            "--y-pred",
            str(y_pred_path),
            "--metric",
            "not-a-metric",
        ],
    )

    assert result.exit_code == 2
    assert "Unknown metric: not-a-metric" in result.output
    assert "Traceback" not in result.output


def test_list_with_log_level():
    result = runner.invoke(app, ["list", "--log-level", "basic"])
    assert result.exit_code == 0


def test_compute_with_log_level(tmp_path):
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_true_path = tmp_path / "y_true.npy"
    y_pred_path = tmp_path / "y_pred.npy"
    np.save(y_true_path, y_true)
    np.save(y_pred_path, y_pred)

    result = runner.invoke(
        app,
        [
            "compute",
            "--y-true",
            str(y_true_path),
            "--y-pred",
            str(y_pred_path),
            "--log-level",
            "basic",
        ],
    )
    assert result.exit_code == 0
