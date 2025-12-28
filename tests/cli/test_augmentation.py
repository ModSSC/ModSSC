from typer.testing import CliRunner

from modssc.cli.augmentation import app

runner = CliRunner()


def test_list_ops():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "vision.random_crop_pad" in result.stdout


def test_list_ops_modality():
    result = runner.invoke(app, ["list", "--modality", "vision"])
    assert result.exit_code == 0
    assert "vision.random_crop_pad" in result.stdout
    assert "text.word_dropout" not in result.stdout


def test_info():
    result = runner.invoke(app, ["info", "vision.random_crop_pad"])
    assert result.exit_code == 0
    assert "op_id: vision.random_crop_pad" in result.stdout
    assert "modality: vision" in result.stdout


def test_info_json():
    result = runner.invoke(app, ["info", "vision.random_crop_pad", "--as-json"])
    assert result.exit_code == 0
    assert '"op_id": "vision.random_crop_pad"' in result.stdout


def test_list_ops_with_log_level():
    result = runner.invoke(app, ["list", "--log-level", "basic"])
    assert result.exit_code == 0


def test_info_includes_doc_with_log_level():
    result = runner.invoke(app, ["info", "core.identity", "--log-level", "basic"])
    assert result.exit_code == 0
    assert "Return the input as-is." in result.stdout


def test_info_without_doc(monkeypatch):
    monkeypatch.setattr(
        "modssc.cli.augmentation.op_info",
        lambda *_: {"op_id": "x", "modality": "any", "doc": "", "defaults": {}},
    )
    result = runner.invoke(app, ["info", "x"])
    assert result.exit_code == 0
    assert "Defaults:" in result.stdout
