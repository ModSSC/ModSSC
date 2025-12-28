import json
import logging
from unittest.mock import MagicMock, patch

import numpy as np
from typer.testing import CliRunner

from modssc.cli.preprocess import app
from modssc.data_loader.errors import DataLoaderError
from modssc.preprocess.errors import PreprocessError

runner = CliRunner()


def test_steps_list():
    result = runner.invoke(app, ["steps", "list"])
    assert result.exit_code == 0
    assert "core.ensure_2d" in result.stdout


def test_steps_list_json():
    result = runner.invoke(app, ["steps", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert "core.ensure_2d" in data


def test_steps_info():
    result = runner.invoke(app, ["steps", "info", "core.ensure_2d"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["id"] == "core.ensure_2d"


def test_models_list():
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0


def test_models_list_json():
    result = runner.invoke(app, ["models", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)


def test_models_info():
    with patch("modssc.cli.preprocess.model_info") as mock_info:
        mock_info.return_value = {"model_id": "test_model", "modality": "vision"}
        result = runner.invoke(app, ["models", "info", "test_model"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["model_id"] == "test_model"


def test_steps_list_with_log_level():
    result = runner.invoke(app, ["steps", "list", "--log-level", "basic"])
    assert result.exit_code == 0


def test_steps_info_with_log_level():
    result = runner.invoke(app, ["steps", "info", "core.ensure_2d", "--log-level", "basic"])
    assert result.exit_code == 0


def test_models_list_with_log_level():
    result = runner.invoke(app, ["models", "list", "--log-level", "basic"])
    assert result.exit_code == 0


def test_models_info_with_log_level():
    with patch("modssc.cli.preprocess.model_info") as mock_info:
        mock_info.return_value = {"model_id": "test_model", "modality": "vision"}
        result = runner.invoke(app, ["models", "info", "test_model", "--log-level", "basic"])
        assert result.exit_code == 0


def test_run(tmp_path):
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text("steps: []")

    with (
        patch("modssc.cli.preprocess.load_dataset") as mock_load_ds,
        patch("modssc.cli.preprocess.load_plan") as mock_load_plan,
        patch("modssc.cli.preprocess.preprocess") as mock_preprocess,
    ):
        mock_ds = MagicMock()
        mock_ds.train.y.shape = (10,)
        mock_load_ds.return_value = mock_ds

        mock_plan = MagicMock()
        mock_plan.output_key = "test_output"
        mock_load_plan.return_value = mock_plan

        mock_result = MagicMock()
        mock_result.dataset.train.X.shape = (10, 5)
        mock_result.preprocess_fingerprint = "prep-fp"
        mock_result.cache_dir = "/tmp/preprocess-cache"
        mock_preprocess.return_value = mock_result

        result = runner.invoke(app, ["run", "--plan", str(plan_file), "--dataset", "toy"])

        assert result.exit_code == 0
        assert "test_output" in result.stdout
        assert "train_X_shape" in result.stdout

        mock_load_ds.assert_called_with("toy", cache_dir=None)
        mock_load_plan.assert_called_with(plan_file)
        mock_preprocess.assert_called()


def test_run_with_dataset_cache_dir(tmp_path):
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text("steps: []")

    with (
        patch("modssc.cli.preprocess.load_dataset") as mock_load_ds,
        patch("modssc.cli.preprocess.load_plan") as mock_load_plan,
        patch("modssc.cli.preprocess.preprocess") as mock_preprocess,
    ):
        mock_ds = MagicMock()
        mock_ds.train.y.shape = (10,)
        mock_load_ds.return_value = mock_ds

        mock_plan = MagicMock()
        mock_plan.output_key = "test_output"
        mock_load_plan.return_value = mock_plan

        mock_result = MagicMock()
        mock_result.dataset.train.X.shape = (10, 5)
        mock_result.preprocess_fingerprint = "prep-fp"
        mock_result.cache_dir = "/tmp/preprocess-cache"
        mock_preprocess.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "run",
                "--plan",
                str(plan_file),
                "--dataset",
                "toy",
                "--dataset-cache-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        mock_load_ds.assert_called_with("toy", cache_dir=tmp_path)


def test_run_handles_data_loader_error(tmp_path, caplog):
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text("steps: []")

    with (
        patch("modssc.cli.preprocess.load_dataset", side_effect=DataLoaderError("boom")),
        caplog.at_level(logging.DEBUG, logger="modssc.cli.preprocess"),
    ):
        result = runner.invoke(
            app,
            ["run", "--plan", str(plan_file), "--dataset", "toy", "--log-level", "detailed"],
        )
    assert result.exit_code == 2
    assert "boom" in result.output


def test_run_handles_plan_parse_error(tmp_path, caplog):
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text("steps: []")

    mock_ds = MagicMock()
    mock_ds.train.y = [0, 1, 2]

    with (
        patch("modssc.cli.preprocess.load_dataset", return_value=mock_ds),
        patch("modssc.cli.preprocess.load_plan", side_effect=ValueError("bad plan")),
        caplog.at_level(logging.DEBUG, logger="modssc.cli.preprocess"),
    ):
        result = runner.invoke(
            app,
            ["run", "--plan", str(plan_file), "--dataset", "toy", "--log-level", "detailed"],
        )
    assert result.exit_code == 2
    assert "Invalid plan file" in result.output


def test_run_handles_preprocess_error(tmp_path, caplog):
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text("steps: []")

    mock_ds = MagicMock()
    mock_ds.train.y = np.array([0, 1, 2])
    mock_plan = MagicMock()
    mock_plan.output_key = "out"

    with (
        patch("modssc.cli.preprocess.load_dataset", return_value=mock_ds),
        patch("modssc.cli.preprocess.load_plan", return_value=mock_plan),
        patch("modssc.cli.preprocess.preprocess", side_effect=PreprocessError("boom")),
        caplog.at_level(logging.DEBUG, logger="modssc.cli.preprocess"),
    ):
        result = runner.invoke(
            app,
            ["run", "--plan", str(plan_file), "--dataset", "toy", "--log-level", "detailed"],
        )
    assert result.exit_code == 2
    assert "boom" in result.output


def test_run_handles_data_loader_error_without_debug(tmp_path):
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text("steps: []")

    with patch("modssc.cli.preprocess.load_dataset", side_effect=DataLoaderError("boom")):
        result = runner.invoke(app, ["run", "--plan", str(plan_file), "--dataset", "toy"])
    assert result.exit_code == 2


def test_run_handles_plan_parse_error_without_debug(tmp_path):
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text("steps: []")

    mock_ds = MagicMock()
    mock_ds.train.y = np.array([0, 1, 2])

    with (
        patch("modssc.cli.preprocess.load_dataset", return_value=mock_ds),
        patch("modssc.cli.preprocess.load_plan", side_effect=ValueError("bad plan")),
    ):
        result = runner.invoke(app, ["run", "--plan", str(plan_file), "--dataset", "toy"])
    assert result.exit_code == 2


def test_run_handles_preprocess_error_without_debug(tmp_path):
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text("steps: []")

    mock_ds = MagicMock()
    mock_ds.train.y = np.array([0, 1, 2])
    mock_plan = MagicMock()
    mock_plan.output_key = "out"

    with (
        patch("modssc.cli.preprocess.load_dataset", return_value=mock_ds),
        patch("modssc.cli.preprocess.load_plan", return_value=mock_plan),
        patch("modssc.cli.preprocess.preprocess", side_effect=PreprocessError("boom")),
    ):
        result = runner.invoke(app, ["run", "--plan", str(plan_file), "--dataset", "toy"])
    assert result.exit_code == 2
