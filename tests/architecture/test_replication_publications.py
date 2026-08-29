from __future__ import annotations

from pathlib import Path

from modssc.evaluation import verify_paper_publication

REPO_ROOT = Path(__file__).resolve().parents[2]
REPLICATIONS_ROOT = REPO_ROOT / "docs" / "replications"
RESULTS_ROOT = REPLICATIONS_ROOT / "results"
TRACKS = ("paper",)

FORBIDDEN_PARTS = {".staging", "raw"}
FORBIDDEN_ENDINGS = {
    ".bin",
    ".ckpt",
    ".err",
    ".h5",
    ".hdf5",
    ".joblib",
    ".log",
    ".npy",
    ".npz",
    ".onnx",
    ".out",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
}
MAX_TEXT_FILE_BYTES = 256 * 1024
MAX_OBSERVATIONS_BYTES = 2 * 1024 * 1024


def _release_dirs() -> list[tuple[str, Path]]:
    releases: list[tuple[str, Path]] = []
    for track in TRACKS:
        track_root = RESULTS_ROOT / track
        releases.extend((track, path) for path in sorted(track_root.iterdir()) if path.is_dir())
    return releases


def test_replication_tree_contains_only_portable_text_evidence() -> None:
    violations: list[str] = []
    for path in sorted(REPLICATIONS_ROOT.rglob("*")):
        relative = path.relative_to(REPLICATIONS_ROOT)
        if path.is_symlink():
            violations.append(f"{relative}: symlinks are forbidden")
            continue
        if not path.is_file():
            continue
        if FORBIDDEN_PARTS.intersection(relative.parts):
            violations.append(f"{relative}: private staging/raw path is forbidden")
        if any(path.name.lower().endswith(ending) for ending in FORBIDDEN_ENDINGS):
            violations.append(f"{relative}: binary/archive/log suffix is forbidden")
        limit = MAX_OBSERVATIONS_BYTES if path.name == "observations.jsonl" else MAX_TEXT_FILE_BYTES
        if path.stat().st_size > limit:
            violations.append(f"{relative}: {path.stat().st_size} bytes exceeds {limit}")
    assert not violations, "invalid replication publication files:\n" + "\n".join(violations)


def test_every_replication_release_passes_the_native_verifier() -> None:
    for track, release_dir in _release_dirs():
        files = {
            path.relative_to(release_dir).as_posix(): path.read_bytes()
            for path in sorted(release_dir.rglob("*"))
            if path.is_file()
        }
        verification = verify_paper_publication(files)
        assert verification.track == track
        assert verification.release_id == release_dir.name
