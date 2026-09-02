from importlib import metadata

from bench.utils import import_tools
from bench.utils.import_tools import check_extra_installed, distributions_for_extras


def test_check_extra_installed_finds_project_metadata_outside_repo_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    missing = check_extra_installed("transductive-torch")

    assert isinstance(missing, list)


def test_check_extra_installed_uses_wheel_metadata_without_checkout(monkeypatch):
    class _Metadata:
        @staticmethod
        def get_all(name):
            assert name == "Provides-Extra"
            return ["demo", "other"]

    monkeypatch.setattr(import_tools, "_find_pyproject", lambda: None)
    monkeypatch.setattr(metadata, "metadata", lambda _name: _Metadata())
    monkeypatch.setattr(
        metadata,
        "requires",
        lambda _name: [
            "PyYAML>=6; extra == 'demo'",
            'missing-demo-package>=1; python_version >= "3.11" and extra == "demo"',
            "numpy; extra == 'other'",
        ],
    )

    assert check_extra_installed("demo") == ["missing_demo_package"]


def test_annoy_extra_expands_to_runtime_distribution() -> None:
    assert distributions_for_extras(("graph-annoy",)) == ("annoy", "numpy", "scipy")
