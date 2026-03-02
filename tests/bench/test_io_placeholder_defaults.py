import importlib.util
from pathlib import Path


def _load_yaml_helper():
    module_path = Path(__file__).resolve().parents[2] / "bench" / "utils" / "io.py"
    spec = importlib.util.spec_from_file_location("bench_utils_io_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_yaml


load_yaml = _load_yaml_helper()


def test_load_yaml_placeholder_defaults_honor_cache_root(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "\n".join(
            [
                "run:",
                "  output_dir: ${MODSSC_OUTPUT_DIR}/p01",
                "dataset:",
                "  cache_dir: ${MODSSC_DATASET_CACHE_DIR}",
                "preprocess:",
                "  cache_dir: ${MODSSC_PREPROCESS_CACHE_DIR}",
            ]
        ),
        encoding="utf-8",
    )

    root = (tmp_path / "global_root").resolve()
    monkeypatch.setenv("MODSSC_CACHE_ROOT", str(root))
    monkeypatch.delenv("MODSSC_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("MODSSC_DATASET_CACHE_DIR", raising=False)
    monkeypatch.delenv("MODSSC_PREPROCESS_CACHE_DIR", raising=False)

    data = load_yaml(cfg)
    assert data["run"]["output_dir"] == str(root / "output" / "p01")
    assert data["dataset"]["cache_dir"] == str(root / "datasets")
    assert data["preprocess"]["cache_dir"] == str(root / "preprocess")


def test_load_yaml_placeholder_dataset_uses_cache_dir_alias(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("dataset:\n  cache_dir: ${MODSSC_DATASET_CACHE_DIR}\n", encoding="utf-8")

    root = (tmp_path / "global_root").resolve()
    dataset_alias = (tmp_path / "custom_dataset_cache").resolve()
    monkeypatch.setenv("MODSSC_CACHE_ROOT", str(root))
    monkeypatch.setenv("MODSSC_CACHE_DIR", str(dataset_alias))
    monkeypatch.delenv("MODSSC_DATASET_CACHE_DIR", raising=False)

    data = load_yaml(cfg)
    assert data["dataset"]["cache_dir"] == str(dataset_alias)
