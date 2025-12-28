from pathlib import Path
from unittest.mock import patch

from modssc.sampling.api import default_split_cache_dir


def test_default_split_cache_dir_dev_repo(tmp_path):
    (tmp_path / "pyproject.toml").touch()

    with patch("pathlib.Path.cwd", return_value=tmp_path):
        path = default_split_cache_dir()
        assert path == tmp_path / "cache" / "splits"


def test_default_split_cache_dir_dev_repo_parent(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    with patch("pathlib.Path.cwd", return_value=subdir):
        path = default_split_cache_dir()
        assert path == tmp_path / "cache" / "splits"


def test_default_split_cache_dir_user_cache_fallback(tmp_path):
    original_exists = Path.exists

    def side_effect(self):
        if self.name == "pyproject.toml":
            return False
        return original_exists(self)

    with (
        patch("pathlib.Path.exists", new=side_effect),
        patch("modssc.sampling.api.user_cache_dir", return_value=str(tmp_path / "user_cache")),
    ):
        path = default_split_cache_dir()
        assert path == tmp_path / "user_cache" / "splits"
