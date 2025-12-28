import json
import logging
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from modssc.cli import sampling as sampling_mod
from modssc.cli.sampling import app
from modssc.data_loader.errors import DataLoaderError

runner = CliRunner()


def test_create(tmp_path):
    plan_file = tmp_path / "plan.json"
    plan_content = {
        "split": {"kind": "holdout", "test_fraction": 0.2},
        "labeling": {"mode": "fraction", "value": 0.1},
        "imbalance": {"kind": "none"},
        "policy": {},
    }
    plan_file.write_text(json.dumps(plan_content))
    out_dir = tmp_path / "output"

    with (
        patch("modssc.cli.sampling.load_dataset") as mock_load_ds,
        patch("modssc.cli.sampling.sample") as mock_sample,
        patch("modssc.cli.sampling.save_split") as mock_save_split,
    ):
        mock_ds = MagicMock()
        mock_ds.meta = {"dataset_fingerprint": "fp123"}
        mock_load_ds.return_value = mock_ds

        mock_result = MagicMock()
        mock_result.dataset_fingerprint = "fp123"
        mock_result.split_fingerprint = "split123"
        mock_result.stats = {"train": {"n": 1}}
        mock_sample.return_value = (mock_result, None)

        result = runner.invoke(
            app,
            [
                "create",
                "--dataset",
                "toy",
                "--plan",
                str(plan_file),
                "--out",
                str(out_dir),
                "--seed",
                "42",
                "--log-level",
                "basic",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["output_dir"] == str(out_dir)

        mock_load_ds.assert_called()
        mock_sample.assert_called()
        mock_save_split.assert_called_with(mock_result, out_dir, overwrite=False)


def test_create_kfold(tmp_path):
    plan_file = tmp_path / "plan_kfold.json"
    plan_content = {
        "split": {"kind": "kfold", "k": 5, "fold": 0},
    }
    plan_file.write_text(json.dumps(plan_content))
    out_dir = tmp_path / "output_kfold"

    with (
        patch("modssc.cli.sampling.load_dataset") as mock_load_ds,
        patch("modssc.cli.sampling.sample") as mock_sample,
        patch("modssc.cli.sampling.save_split"),
    ):
        mock_ds = MagicMock()
        mock_ds.meta = {"dataset_fingerprint": "fp123"}
        mock_load_ds.return_value = mock_ds
        mock_result = MagicMock()
        mock_result.dataset_fingerprint = "fp123"
        mock_result.split_fingerprint = "split123"
        mock_result.stats = {"train": {"n": 1}}
        mock_sample.return_value = (mock_result, None)

        result = runner.invoke(
            app, ["create", "--dataset", "toy", "--plan", str(plan_file), "--out", str(out_dir)]
        )

        assert result.exit_code == 0


def test_create_missing_fingerprint(tmp_path):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text("{}")
    out_dir = tmp_path / "output"

    with patch("modssc.cli.sampling.load_dataset") as mock_load_ds:
        mock_ds = MagicMock()
        mock_ds.meta = {}
        mock_load_ds.return_value = mock_ds

        result = runner.invoke(
            app, ["create", "--dataset", "toy", "--plan", str(plan_file), "--out", str(out_dir)]
        )

        assert result.exit_code != 0
        assert "Dataset fingerprint is missing" in result.stdout or isinstance(
            result.exception, SystemExit
        )


def test_create_rejects_unknown_plan_keys(tmp_path):
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text("split:\n  kind: holdout\nunknown: 123\n")
    out_dir = tmp_path / "output"

    with patch("modssc.cli.sampling.load_dataset") as mock_load_ds:
        mock_ds = MagicMock()
        mock_ds.meta = {"dataset_fingerprint": "fp123"}
        mock_load_ds.return_value = mock_ds

        result = runner.invoke(
            app, ["create", "--dataset", "toy", "--plan", str(plan_file), "--out", str(out_dir)]
        )

        assert result.exit_code != 0
        assert "Unknown keys in plan" in result.output


def test_create_yaml_plan(tmp_path):
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text("split:\n  kind: holdout\n  test_fraction: 0.2\n")
    out_dir = tmp_path / "output_yaml"

    with (
        patch("modssc.cli.sampling.load_dataset") as mock_load_ds,
        patch("modssc.cli.sampling.sample") as mock_sample,
        patch("modssc.cli.sampling.save_split") as mock_save_split,
    ):
        mock_ds = MagicMock()
        mock_ds.meta = {"dataset_fingerprint": "fp123"}
        mock_load_ds.return_value = mock_ds

        mock_result = MagicMock()
        mock_result.dataset_fingerprint = "fp123"
        mock_result.split_fingerprint = "split123"
        mock_result.stats = {"train": {"n": 1}}
        mock_sample.return_value = (mock_result, None)

        result = runner.invoke(
            app,
            [
                "create",
                "--dataset",
                "toy",
                "--plan",
                str(plan_file),
                "--out",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0
        mock_save_split.assert_called_with(mock_result, out_dir, overwrite=False)


def test_create_rejects_non_mapping_plan(tmp_path):
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text("- just\n- a\n- list\n")
    out_dir = tmp_path / "output_bad"

    with patch("modssc.cli.sampling.load_dataset") as mock_load_ds:
        mock_ds = MagicMock()
        mock_ds.meta = {"dataset_fingerprint": "fp123"}
        mock_load_ds.return_value = mock_ds

        result = runner.invoke(
            app,
            [
                "create",
                "--dataset",
                "toy",
                "--plan",
                str(plan_file),
                "--out",
                str(out_dir),
            ],
        )

        assert result.exit_code != 0
        assert "Plan file must contain a mapping" in result.output


def test_create_uses_dataset_cache_dir(tmp_path):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text("{}")
    out_dir = tmp_path / "output_cache"

    with (
        patch("modssc.cli.sampling.load_dataset") as mock_load_ds,
        patch("modssc.cli.sampling.sample") as mock_sample,
        patch("modssc.cli.sampling.save_split"),
    ):
        mock_ds = MagicMock()
        mock_ds.meta = {"dataset_fingerprint": "fp123"}
        mock_load_ds.return_value = mock_ds
        mock_result = MagicMock()
        mock_result.dataset_fingerprint = "fp123"
        mock_result.split_fingerprint = "split123"
        mock_result.stats = {"train": {"n": 1}}
        mock_sample.return_value = (mock_result, None)

        result = runner.invoke(
            app,
            [
                "create",
                "--dataset",
                "toy",
                "--plan",
                str(plan_file),
                "--out",
                str(out_dir),
                "--dataset-cache-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        assert mock_load_ds.call_args.kwargs["cache_dir"] == tmp_path


def test_show(tmp_path):
    split_dir = tmp_path / "split"
    split_dir.mkdir()

    with patch("modssc.cli.sampling.load_split") as mock_load_split:
        mock_res = MagicMock()
        mock_res.plan = {"some": "plan"}
        mock_res.stats = {"some": "stats"}
        mock_load_split.return_value = mock_res

        result = runner.invoke(app, ["show", str(split_dir), "--log-level", "basic"])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["plan"] == {"some": "plan"}
        assert data["stats"] == {"some": "stats"}


def test_show_without_log_level(tmp_path):
    split_dir = tmp_path / "split"
    split_dir.mkdir()

    with patch("modssc.cli.sampling.load_split") as mock_load_split:
        mock_res = MagicMock()
        mock_res.plan = {"some": "plan"}
        mock_res.stats = {"some": "stats"}
        mock_load_split.return_value = mock_res

        sampling_mod.show(split_dir, log_level=None)


def test_validate(tmp_path):
    split_dir = tmp_path / "split"
    split_dir.mkdir()

    with (
        patch("modssc.cli.sampling.load_dataset") as mock_load_ds,
        patch("modssc.cli.sampling.load_split") as mock_load_split,
    ):
        mock_ds = MagicMock()
        mock_ds.train.y = [0, 1, 0, 1]
        mock_ds.test.y = [0, 1]

        mock_ds.train.edges = None
        mock_ds.train.masks = None

        mock_load_ds.return_value = mock_ds

        mock_res = MagicMock()
        mock_load_split.return_value = mock_res

        result = runner.invoke(
            app, ["validate", str(split_dir), "--dataset", "toy", "--log-level", "basic"]
        )

        assert result.exit_code == 0
        assert "OK" in result.stdout
        mock_res.validate.assert_called()


def test_validate_graph(tmp_path):
    split_dir = tmp_path / "split"
    split_dir.mkdir()

    with (
        patch("modssc.cli.sampling.load_dataset") as mock_load_ds,
        patch("modssc.cli.sampling.load_split") as mock_load_split,
    ):
        mock_ds = MagicMock()
        mock_ds.train.y = [0, 1, 0, 1]
        mock_ds.test.y = [0, 1]
        mock_ds.train.edges = "some_edges"

        mock_load_ds.return_value = mock_ds

        mock_res = MagicMock()
        mock_load_split.return_value = mock_res

        result = runner.invoke(
            app, ["validate", str(split_dir), "--dataset", "toy", "--log-level", "basic"]
        )

        assert result.exit_code == 0
        assert "OK" in result.stdout

        args, kwargs = mock_res.validate.call_args
        assert kwargs.get("n_nodes") == 4


def test_create_handles_data_loader_error(tmp_path, caplog):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text("{}")
    out_dir = tmp_path / "output"

    with (
        patch("modssc.cli.sampling.load_dataset", side_effect=DataLoaderError("boom")),
        caplog.at_level(logging.DEBUG, logger="modssc.cli.sampling"),
    ):
        result = runner.invoke(
            app,
            [
                "create",
                "--dataset",
                "toy",
                "--plan",
                str(plan_file),
                "--out",
                str(out_dir),
                "--log-level",
                "detailed",
            ],
        )
    assert result.exit_code == 2
    assert "boom" in result.output


def test_validate_handles_data_loader_error(tmp_path, caplog):
    split_dir = tmp_path / "split"
    split_dir.mkdir()

    with (
        patch("modssc.cli.sampling.load_dataset", side_effect=DataLoaderError("boom")),
        caplog.at_level(logging.DEBUG, logger="modssc.cli.sampling"),
    ):
        result = runner.invoke(
            app,
            [
                "validate",
                str(split_dir),
                "--dataset",
                "toy",
                "--log-level",
                "detailed",
            ],
        )
    assert result.exit_code == 2
    assert "boom" in result.output


def test_create_handles_data_loader_error_without_debug(tmp_path):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text("{}")
    out_dir = tmp_path / "output"

    with patch("modssc.cli.sampling.load_dataset", side_effect=DataLoaderError("boom")):
        result = runner.invoke(
            app,
            [
                "create",
                "--dataset",
                "toy",
                "--plan",
                str(plan_file),
                "--out",
                str(out_dir),
            ],
        )
    assert result.exit_code == 2


def test_validate_handles_data_loader_error_without_debug(tmp_path):
    split_dir = tmp_path / "split"
    split_dir.mkdir()

    with patch("modssc.cli.sampling.load_dataset", side_effect=DataLoaderError("boom")):
        result = runner.invoke(app, ["validate", str(split_dir), "--dataset", "toy"])
    assert result.exit_code == 2
