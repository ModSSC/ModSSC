from bench.utils.import_tools import check_extra_installed


def test_check_extra_installed_finds_project_metadata_outside_repo_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    missing = check_extra_installed("transductive-torch")

    assert isinstance(missing, list)
