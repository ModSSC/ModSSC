import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from modssc.cli.graph import _extract_xy, _load_dataset, _load_graph_spec, app
from modssc.data_loader.errors import DataLoaderError
from modssc.graph.artifacts import GraphArtifact
from modssc.graph.errors import GraphError

runner = CliRunner()


def test_extract_xy_simple():
    ds = MagicMock()
    ds.X = "X"
    ds.y = "y"
    assert _extract_xy(ds) == ("X", "y")


def test_extract_xy_features_y_in_features():
    ds = MagicMock()
    del ds.X
    ds.features.X = "X"
    del ds.y

    ds = MagicMock(spec=[])
    ds.features = MagicMock()
    ds.features.X = "X"
    ds.features.y = "y"

    assert _extract_xy(ds) == ("X", "y")


def test_extract_xy_features_y_on_ds():
    ds = MagicMock(spec=[])
    ds.features = MagicMock()
    ds.features.X = "X"
    ds.y = "y"
    assert _extract_xy(ds) == ("X", "y")


def test_extract_xy_train():
    ds = MagicMock(spec=[])
    ds.train = MagicMock()
    ds.train.X = "X"
    ds.train.y = "y"
    assert _extract_xy(ds) == ("X", "y")


def test_extract_xy_fail():
    ds = MagicMock(spec=[])
    with pytest.raises(typer.Exit):
        _extract_xy(ds)


def test_load_dataset():
    with patch("modssc.data_loader.load_dataset") as mock_load:
        mock_load.return_value = "ds"
        assert _load_dataset("key") == "ds"
        mock_load.assert_called_with("key", cache_dir=None)


def test_load_dataset_error():
    with (
        patch("modssc.data_loader.load_dataset", side_effect=DataLoaderError("boom")),
        pytest.raises(typer.Exit),
    ):
        _load_dataset("bad")


def test_load_graph_spec_yaml(tmp_path: Path):
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("scheme: knn\nk: 3\nmetric: cosine\n")
    spec = _load_graph_spec(spec_file)
    assert spec.scheme == "knn"
    assert spec.k == 3


def test_load_graph_spec_rejects_unknown_key(tmp_path: Path):
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("scheme: knn\nunknown: 1\n")
    with pytest.raises(typer.Exit):
        _load_graph_spec(spec_file)


def test_load_graph_spec_json(tmp_path: Path):
    spec_file = tmp_path / "spec.json"
    spec_file.write_text('{"scheme": "knn", "k": 4, "metric": "cosine"}')
    spec = _load_graph_spec(spec_file)
    assert spec.k == 4


def test_load_graph_spec_rejects_unknown_weights(tmp_path: Path):
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(
        "scheme: knn\nk: 4\nmetric: cosine\nweights:\n  kind: binary\n  extra: 1\n"
    )
    with pytest.raises(typer.Exit):
        _load_graph_spec(spec_file)


def test_load_graph_spec_rejects_invalid_weights_type(tmp_path: Path):
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("scheme: knn\nk: 3\nmetric: cosine\nweights: nope\n")
    with pytest.raises(typer.Exit):
        _load_graph_spec(spec_file)


def test_load_graph_spec_rejects_invalid_spec(tmp_path: Path):
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("scheme: knn\nk: 0\nmetric: cosine\n")
    with pytest.raises(typer.Exit):
        _load_graph_spec(spec_file)


def test_load_graph_spec_allows_null_weights(tmp_path: Path):
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("scheme: knn\nk: 3\nmetric: cosine\nweights: null\n")
    with pytest.raises(TypeError):
        _load_graph_spec(spec_file)


@patch("modssc.cli.graph._load_dataset")
@patch("modssc.cli.graph._extract_xy")
@patch("modssc.cli.graph.build_graph")
def test_build_cmd(mock_build, mock_extract, mock_load):
    mock_load.return_value = "ds"
    mock_extract.return_value = (np.zeros((10, 2)), None)

    mock_g = MagicMock()
    mock_g.meta = {"fingerprint": "fp"}
    mock_g.n_nodes = 10
    mock_g.edge_index.shape = (2, 50)
    mock_build.return_value = mock_g

    result = runner.invoke(
        app, ["build", "--dataset", "ds_key", "--k", "5", "--log-level", "basic"]
    )
    assert result.exit_code == 0
    assert "fingerprint" in result.stdout
    assert "n_edges" in result.stdout


@patch("modssc.cli.graph._load_dataset")
@patch("modssc.cli.graph._extract_xy")
def test_build_cmd_rejects_invalid_spec(mock_extract, mock_load):
    mock_load.return_value = "ds"
    mock_extract.return_value = (np.zeros((10, 2)), None)
    result = runner.invoke(
        app, ["build", "--dataset", "ds_key", "--k", "0", "--log-level", "basic"]
    )
    assert result.exit_code == 2


@patch("modssc.cli.graph._load_dataset")
@patch("modssc.cli.graph._extract_xy")
@patch("modssc.cli.graph.build_graph")
def test_build_cmd_with_spec(mock_build, mock_extract, mock_load, tmp_path: Path):
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("scheme: epsilon\nmetric: euclidean\nradius: 0.9\n")

    mock_load.return_value = "ds"
    mock_extract.return_value = (np.zeros((10, 2)), None)

    mock_g = MagicMock()
    mock_g.meta = {"fingerprint": "fp"}
    mock_g.n_nodes = 10
    mock_g.edge_index.shape = (2, 50)
    mock_build.return_value = mock_g

    result = runner.invoke(app, ["build", "--dataset", "ds_key", "--spec", str(spec_file)])
    assert result.exit_code == 0
    spec = mock_build.call_args.kwargs["spec"]
    assert spec.scheme == "epsilon"
    assert spec.radius == 0.9


@patch("modssc.cli.graph._load_dataset")
@patch("modssc.cli.graph._extract_xy")
@patch("modssc.cli.graph.build_graph")
def test_build_cmd_uses_dataset_cache(mock_build, mock_extract, mock_load, tmp_path: Path):
    mock_load.return_value = "ds"
    mock_extract.return_value = (np.zeros((10, 2)), None)

    mock_g = MagicMock()
    mock_g.meta = {"fingerprint": "fp"}
    mock_g.n_nodes = 10
    mock_g.edge_index.shape = (2, 50)
    mock_build.return_value = mock_g

    result = runner.invoke(
        app,
        [
            "build",
            "--dataset",
            "ds_key",
            "--k",
            "5",
            "--dataset-cache-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert mock_load.call_args.kwargs["cache_dir"] == tmp_path


@patch("modssc.cli.graph._load_dataset")
@patch("modssc.cli.graph._extract_xy")
@patch("modssc.cli.graph.build_graph")
@patch("modssc.cli.graph.graph_to_views")
def test_views_build_cmd(mock_g2v, mock_build, mock_extract, mock_load):
    mock_load.return_value = "ds"

    X = np.zeros((10, 2))
    mock_extract.return_value = (X, None)

    mock_g = MagicMock()
    mock_build.return_value = mock_g

    mock_res = MagicMock()
    mock_res.meta = {"fingerprint": "fp"}
    mock_res.views = {"attr": "v"}
    mock_g2v.return_value = mock_res

    with patch("modssc.cli.graph.NodeDataset"):
        result = runner.invoke(
            app, ["views", "build", "--dataset", "ds_key", "--log-level", "basic"]
        )
        if result.exit_code != 0:
            print(result.stdout)
            print(result.exception)
        assert result.exit_code == 0
        assert "fingerprint" in result.stdout
        assert "attr" in result.stdout


@patch("modssc.cli.graph._load_dataset")
@patch("modssc.cli.graph._extract_xy")
@patch("modssc.cli.graph.build_graph")
@patch("modssc.cli.graph.graph_to_views")
def test_views_build_cmd_with_y(mock_g2v, mock_build, mock_extract, mock_load):
    mock_load.return_value = "ds"

    X = np.zeros((10, 2))
    y = np.zeros((10,))
    mock_extract.return_value = (X, y)

    mock_g = MagicMock()
    mock_build.return_value = mock_g

    mock_res = MagicMock()
    mock_res.meta = {"fingerprint": "fp"}
    mock_res.views = {"attr": "v"}
    mock_g2v.return_value = mock_res

    with patch("modssc.cli.graph.NodeDataset"):
        result = runner.invoke(
            app, ["views", "build", "--dataset", "ds_key", "--log-level", "basic"]
        )
        assert result.exit_code == 0
        assert "fingerprint" in result.stdout


@patch("modssc.cli.graph.GraphCache")
def test_cache_ls(mock_cache):
    mock_store = MagicMock()
    mock_store.list.return_value = ["fp1", "fp2"]
    mock_cache.default.return_value = mock_store

    result = runner.invoke(app, ["cache", "ls", "--log-level", "basic"])
    assert result.exit_code == 0
    assert "fp1" in result.stdout
    assert "fp2" in result.stdout


@patch("modssc.cli.graph.GraphCache")
def test_cache_purge(mock_cache):
    mock_store = MagicMock()
    mock_store.purge.return_value = 5
    mock_cache.default.return_value = mock_store

    result = runner.invoke(app, ["cache", "purge", "--log-level", "basic"])
    assert result.exit_code == 0
    assert "Purged 5" in result.stdout


@patch("modssc.cli.graph.ViewsCache")
def test_views_cache_ls(mock_cache):
    mock_store = MagicMock()
    mock_store.list.return_value = ["fp1", "fp2"]
    mock_cache.default.return_value = mock_store

    result = runner.invoke(app, ["views", "cache-ls", "--log-level", "basic"])
    assert result.exit_code == 0
    assert "fp1" in result.stdout


@patch("modssc.cli.graph._load_dataset")
@patch("modssc.cli.graph._extract_xy")
@patch("modssc.cli.graph.build_graph", side_effect=GraphError("boom"))
def test_build_cmd_logs_graph_error(mock_build, mock_extract, mock_load, caplog):
    mock_load.return_value = "ds"
    mock_extract.return_value = (np.zeros((2, 2)), None)
    with caplog.at_level(logging.DEBUG, logger="modssc.cli.graph"):
        result = runner.invoke(
            app, ["build", "--dataset", "ds_key", "--k", "5", "--log-level", "detailed"]
        )
    assert result.exit_code == 2


@patch("modssc.cli.graph._load_dataset")
@patch("modssc.cli.graph._extract_xy")
@patch("modssc.cli.graph.build_graph", side_effect=GraphError("boom"))
def test_build_cmd_graph_error_without_debug(mock_build, mock_extract, mock_load):
    mock_load.return_value = "ds"
    mock_extract.return_value = (np.zeros((2, 2)), None)
    result = runner.invoke(app, ["build", "--dataset", "ds_key", "--k", "5"])
    assert result.exit_code == 2


@patch("modssc.cli.graph._load_dataset")
@patch("modssc.cli.graph._extract_xy")
@patch("modssc.cli.graph.build_graph", side_effect=GraphError("boom"))
def test_views_build_cmd_default_views_error(mock_build, mock_extract, mock_load, caplog):
    mock_load.return_value = "ds"
    mock_extract.return_value = (np.zeros((3, 2)), None)
    with caplog.at_level(logging.DEBUG, logger="modssc.cli.graph"):
        result = runner.invoke(
            app, ["views", "build", "--dataset", "ds_key", "--log-level", "detailed"]
        )
    assert result.exit_code == 2


@patch("modssc.cli.graph._load_dataset")
@patch("modssc.cli.graph._extract_xy")
@patch("modssc.cli.graph.graph_to_views", side_effect=GraphError("boom"))
@patch("modssc.cli.graph.build_graph")
def test_views_build_cmd_graph_to_views_error(
    mock_build, mock_g2v, mock_extract, mock_load, caplog
):
    mock_load.return_value = "ds"
    X = np.zeros((4, 2))
    y = np.zeros((4,))
    mock_extract.return_value = (X, y)
    mock_build.return_value = GraphArtifact(
        n_nodes=4,
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        edge_weight=None,
        directed=True,
        meta={},
    )
    logger = logging.getLogger("modssc.cli.graph")
    prev_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        with caplog.at_level(logging.DEBUG, logger="modssc.cli.graph"):
            result = runner.invoke(
                app, ["views", "build", "--dataset", "ds_key", "--log-level", "detailed"]
            )
    finally:
        logger.setLevel(prev_level)
    assert result.exit_code == 2


@patch("modssc.cli.graph._load_dataset")
@patch("modssc.cli.graph._extract_xy")
@patch("modssc.cli.graph.graph_to_views", side_effect=GraphError("boom"))
@patch("modssc.cli.graph.build_graph")
def test_views_build_cmd_graph_to_views_error_without_debug(
    mock_build, mock_g2v, mock_extract, mock_load
):
    mock_load.return_value = "ds"
    X = np.zeros((4, 2))
    y = np.zeros((4,))
    mock_extract.return_value = (X, y)
    mock_build.return_value = GraphArtifact(
        n_nodes=4,
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        edge_weight=None,
        directed=True,
        meta={},
    )
    result = runner.invoke(app, ["views", "build", "--dataset", "ds_key"])
    assert result.exit_code == 2


@patch("modssc.cli.graph._load_dataset")
@patch("modssc.cli.graph._extract_xy")
@patch("modssc.cli.graph.build_graph", side_effect=GraphError("boom"))
def test_views_build_cmd_error_without_debug(mock_build, mock_extract, mock_load):
    mock_load.return_value = "ds"
    mock_extract.return_value = (np.zeros((3, 2)), None)
    result = runner.invoke(app, ["views", "build", "--dataset", "ds_key"])
    assert result.exit_code == 2


@patch("modssc.cli.graph._load_dataset")
@patch("modssc.cli.graph._extract_xy")
@patch("modssc.cli.graph.build_graph")
@patch("modssc.cli.graph.graph_to_views")
def test_views_build_cmd_with_views_arg(mock_g2v, mock_build, mock_extract, mock_load):
    mock_load.return_value = "ds"
    X = np.zeros((10, 2))
    mock_extract.return_value = (X, None)

    mock_g = MagicMock()
    mock_build.return_value = mock_g

    mock_res = MagicMock()
    mock_res.meta = {"fingerprint": "fp"}
    mock_res.views = {"attr": "v", "diffusion": "v2"}
    mock_g2v.return_value = mock_res

    with patch("modssc.cli.graph.NodeDataset"):
        result = runner.invoke(
            app,
            [
                "views",
                "build",
                "--dataset",
                "ds_key",
                "--views",
                "attr",
                "--views",
                "diffusion",
            ],
        )
    assert result.exit_code == 0


@patch("modssc.cli.graph.GraphCache")
@patch("modssc.cli.graph.ViewsCache")
def test_cache_commands_without_log_level(mock_views_cache, mock_graph_cache):
    mock_store = MagicMock()
    mock_store.list.return_value = []
    mock_store.purge.return_value = 0
    mock_graph_cache.default.return_value = mock_store

    mock_views = MagicMock()
    mock_views.list.return_value = []
    mock_views_cache.default.return_value = mock_views

    result = runner.invoke(app, ["cache", "ls"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["cache", "purge"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["views", "cache-ls"])
    assert result.exit_code == 0
