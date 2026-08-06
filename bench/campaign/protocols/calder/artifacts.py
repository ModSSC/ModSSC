"""Portable preparation of the frozen Calder et al. Table 1 inputs.

The public reproduction path owns scientific resources only.  It accepts
explicit package and cache paths, authenticates every packaged input, and
materializes the derived preprocessing/graph caches through ModSSC itself.
It deliberately knows nothing about Git worktrees, schedulers, clusters, or
site-specific environment variables.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from bench.campaign.protocols.calder.official import verify_calder_official_assets


class CalderArtifactError(RuntimeError):
    """Raised when portable Calder inputs or derived artifacts are invalid."""


LOCK_KIND = "modssc.calder2020-mnist-table1-portable-artifacts"
LOCK_SCHEMA_VERSION = 1
CANONICAL_CONFIG = Path("poisson_learning/mnist-table1-1-label-per-class.yaml")
CALDER_CONFIGS = tuple(
    Path(method) / f"mnist-table1-{budget}-label-per-class.yaml"
    for method in ("laplace_learning", "poisson_learning")
    for budget in range(1, 6)
)
_REPRODUCTIONS_RELATIVE = Path("bench/configs/reproductions")
_PROTOCOL_INPUTS_RELATIVE = Path("bench/assets/calder2020/protocol_inputs")
_EXPECTED_PREPROCESS_STEPS = (
    "labels.encode",
    "vision.ensure_num_channels",
    "vision.resize",
    "core.ensure_2d",
)
_EXPECTED_GRAPH_SPEC = {
    "scheme": "knn",
    "metric": "euclidean",
    "k": 10,
    "symmetrize": "mean",
    "weights": {"kind": "knn_gaussian"},
    "normalize": "none",
    "self_loops": True,
    "include_self_in_knn": True,
    "edge_weight_dtype": "float64",
    "backend": "precomputed",
    "chunk_size": 1024,
    "precomputed_path": (
        "${MODSSC_ROOT}/bench/assets/calder2020/protocol_inputs/graph/mnist-vae-knn30.npz"
    ),
    "precomputed_sha256": "5b42bb234888c83eed763958a17fdfb8a55c09a2f0071b55a61635d86dc90db5",
    "feature_field": "features.X",
}
_SCIENTIFIC_MODULES = (
    "bench.campaign.protocols.calder.artifacts",
    "bench.campaign.protocols.calder.official",
    "bench.campaign.protocols.calder.oracle",
    "bench.orchestrators.graph",
    "bench.orchestrators.preprocess",
    "bench.orchestrators.sampling",
    "modssc.graph.construction.builder",
    "modssc.graph.specs",
    "modssc.sampling.partition_artifact",
    "modssc.transductive.methods.classic.laplace_learning",
    "modssc.transductive.methods.pde.poisson_learning",
)


@dataclass(frozen=True)
class CalderConfigFamily:
    canonical_path: Path
    canonical_raw: dict[str, Any]
    files: tuple[dict[str, Any], ...]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalderArtifactError(f"{name} must be a mapping")
    return value


def _step_ids(plan: Mapping[str, Any]) -> tuple[str, ...]:
    steps = plan.get("steps")
    if not isinstance(steps, list):
        raise CalderArtifactError("preprocess.plan.steps must be a list")
    return tuple(
        str(_mapping(step, name="preprocess step").get("id") or step.get("step_id"))
        for step in steps
    )


def _contained_file(root: Path, relative: Path, *, label: str) -> Path:
    try:
        resolved_root = root.expanduser().resolve(strict=True)
        candidate = (resolved_root / relative).resolve(strict=True)
        candidate.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise CalderArtifactError(f"{label} is missing or outside the packaged root") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise CalderArtifactError(f"{label} must be a regular packaged file")
    return candidate


def load_calder_config_family(package_root: Path) -> CalderConfigFamily:
    """Load the ten packaged cards and authenticate their common protocol."""

    root = package_root.expanduser().resolve(strict=True)
    reproduction_root = root / _REPRODUCTIONS_RELATIVE
    loaded: dict[Path, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for relative in CALDER_CONFIGS:
        path = _contained_file(
            reproduction_root,
            relative,
            label=f"Calder card {relative.as_posix()}",
        )
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise CalderArtifactError(f"cannot read packaged Calder card: {relative}") from exc
        if not isinstance(raw, dict):
            raise CalderArtifactError(f"Calder card root is not a mapping: {relative}")
        loaded[relative] = raw
        records.append(
            {
                "resource": (_REPRODUCTIONS_RELATIVE / relative).as_posix(),
                "sha256": _sha256_file(path),
            }
        )

    reference = loaded[CANONICAL_CONFIG]
    reference_preprocess = reference.get("preprocess")
    reference_graph = reference.get("graph")
    for relative, raw in loaded.items():
        if raw.get("preprocess") != reference_preprocess:
            raise CalderArtifactError(
                f"Calder cards do not share one preprocess definition: {relative}"
            )
        if raw.get("graph") != reference_graph:
            raise CalderArtifactError(f"Calder cards do not share one graph definition: {relative}")
        dataset = _mapping(raw.get("dataset"), name=f"{relative}.dataset")
        if dataset.get("id") != "mnist" or dataset.get("download") is not False:
            raise CalderArtifactError(f"Calder card is not pinned to cached MNIST: {relative}")

    preprocess = _mapping(reference_preprocess, name="preprocess")
    plan = _mapping(preprocess.get("plan"), name="preprocess.plan")
    if _step_ids(plan) != _EXPECTED_PREPROCESS_STEPS:
        raise CalderArtifactError("Calder preprocessing sequence differs")
    if plan.get("output_key") != "features.X":
        raise CalderArtifactError("Calder preprocessing must retain flattened MNIST features")
    if preprocess.get("seed") != 1 or preprocess.get("fit_on") != "train":
        raise CalderArtifactError("Calder preprocessing seed/fit scope differs")

    graph = _mapping(reference_graph, name="graph")
    if graph.get("enabled") is not True or graph.get("seed") != 1:
        raise CalderArtifactError("Calder graph enablement/seed differs")
    if graph.get("spec") != _EXPECTED_GRAPH_SPEC:
        raise CalderArtifactError("Calder graph specification differs")
    return CalderConfigFamily(
        canonical_path=reproduction_root / CANONICAL_CONFIG,
        canonical_raw=reference,
        files=tuple(records),
    )


def materialized_calder_graph_spec(
    raw_spec: Mapping[str, Any],
    *,
    package_root: Path,
) -> dict[str, Any]:
    """Resolve the authenticated graph resource without a source checkout."""

    actual = dict(raw_spec)
    if actual != _EXPECTED_GRAPH_SPEC:
        raise CalderArtifactError("Calder graph specification differs")
    path = _contained_file(
        package_root,
        _PROTOCOL_INPUTS_RELATIVE / "graph/mnist-vae-knn30.npz",
        label="Calder VAE kNN graph",
    )
    actual["precomputed_path"] = str(path)
    from modssc.graph.specs import GraphBuilderSpec

    spec = GraphBuilderSpec.from_dict(actual)
    spec.validate()
    return spec.to_dict()


def _materialize_card(
    raw: Mapping[str, Any],
    *,
    package_root: Path,
    dataset_cache: Path,
    cache_root: Path,
) -> dict[str, Any]:
    replacements = {
        "${MODSSC_ROOT}": str(package_root),
        "$MODSSC_ROOT": str(package_root),
        "${MODSSC_DATASET_CACHE_DIR}": str(dataset_cache),
        "$MODSSC_DATASET_CACHE_DIR": str(dataset_cache),
        "${MODSSC_PREPROCESS_CACHE_DIR}": str(cache_root / "preprocess"),
        "$MODSSC_PREPROCESS_CACHE_DIR": str(cache_root / "preprocess"),
        "${MODSSC_GRAPH_CACHE_DIR}": str(cache_root / "graph"),
        "$MODSSC_GRAPH_CACHE_DIR": str(cache_root / "graph"),
        "${MODSSC_OUTPUT_DIR}": str(cache_root / "output"),
        "$MODSSC_OUTPUT_DIR": str(cache_root / "output"),
    }

    def resolve(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): resolve(child) for key, child in value.items()}
        if isinstance(value, list):
            return [resolve(child) for child in value]
        if isinstance(value, str):
            result = value
            for token, replacement in replacements.items():
                result = result.replace(token, replacement)
            if "${MODSSC_" in result or "$MODSSC_" in result:
                raise CalderArtifactError(f"unresolved path in Calder card: {result}")
            return result
        return value

    return resolve(copy.deepcopy(dict(raw)))


def _module_file(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str):
        raise CalderArtifactError(f"scientific module has no file identity: {module_name}")
    path = Path(raw).resolve(strict=True)
    if path.suffix == ".pyc" and path.with_suffix(".py").is_file():
        path = path.with_suffix(".py")
    if not path.is_file():
        raise CalderArtifactError(f"scientific module is not a regular file: {module_name}")
    return path


def scientific_payload_identity(family: CalderConfigFamily) -> dict[str, Any]:
    """Hash the installed scientific implementation by logical resource name."""

    modules = [
        {"module": name, "sha256": _sha256_file(_module_file(name))} for name in _SCIENTIFIC_MODULES
    ]
    payload = {"cards": list(family.files), "modules": modules}
    return {
        "kind": "modssc_scientific_payload",
        "sha256": _canonical_sha256(payload),
        **payload,
    }


def artifact_tree_inventory(root: Path) -> dict[str, Any]:
    """Hash a derived cache tree without following symlinks."""

    candidate = root.expanduser()
    if candidate.is_symlink():
        raise CalderArtifactError(f"artifact root must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise CalderArtifactError(f"artifact root is missing: {candidate}") from exc
    if not resolved.is_dir():
        raise CalderArtifactError(f"artifact root is not a directory: {resolved}")
    records: list[dict[str, Any]] = []
    for path in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise CalderArtifactError(f"artifact tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise CalderArtifactError(f"artifact tree contains a special file: {path}")
        relative = path.relative_to(resolved)
        if "_work" in relative.parts or path.name.endswith(".tmp"):
            raise CalderArtifactError(f"artifact tree is incomplete: {path}")
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    if not records:
        raise CalderArtifactError(f"artifact tree is empty: {resolved}")
    return {
        "root": str(resolved),
        "files": records,
        "tree_sha256": _canonical_sha256(records),
    }


def verify_artifact_tree(inventory: Mapping[str, Any]) -> None:
    root = inventory.get("root")
    if not isinstance(root, str) or not root:
        raise CalderArtifactError("artifact inventory has no root")
    if artifact_tree_inventory(Path(root)) != dict(inventory):
        raise CalderArtifactError(f"artifact tree differs from its lock: {root}")


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed.pop("lock_sha256", None)
    sealed["lock_sha256"] = _canonical_sha256(sealed)
    return sealed


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalderArtifactError(f"invalid Calder artifact lock: {path}") from exc
    if not isinstance(raw, dict):
        raise CalderArtifactError("Calder artifact lock root must be a mapping")
    return raw


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise CalderArtifactError(f"refusing to replace immutable lock: {path}")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            if path.read_text(encoding="utf-8") != rendered:
                raise CalderArtifactError(f"concurrent Calder lock differs: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_lock(output: Path) -> Iterator[None]:
    lock_dir = output.with_suffix(output.suffix + ".preparing")
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_dir.mkdir()
    except FileExistsError as exc:
        raise CalderArtifactError(f"another Calder preparation owns {lock_dir}") from exc
    try:
        yield
    finally:
        lock_dir.rmdir()


def _require_under(path: Path, root: Path, *, name: str) -> Path:
    resolved = path.expanduser().resolve()
    allowed = root.expanduser().resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise CalderArtifactError(f"{name} must be below {allowed}: {resolved}") from exc
    return resolved


def _prepare(
    *,
    package_root: Path,
    cache_root: Path,
    dataset_cache: Path,
) -> dict[str, Any]:
    import numpy as np

    from bench.orchestrators import dataset as dataset_orchestrator
    from bench.orchestrators import graph as graph_orchestrator
    from bench.orchestrators import preprocess as preprocess_orchestrator
    from bench.orchestrators import sampling as sampling_orchestrator
    from bench.schema import ExperimentConfig
    from modssc.data_loader import verify_dataset_content

    root = package_root.expanduser().resolve(strict=True)
    cache = cache_root.expanduser().resolve()
    dataset_dir = _require_under(dataset_cache, cache, name="dataset cache")
    family = load_calder_config_family(root)
    raw = _materialize_card(
        family.canonical_raw,
        package_root=root,
        dataset_cache=dataset_dir,
        cache_root=cache,
    )
    cfg = ExperimentConfig.from_dict(raw)
    if cfg.graph is None:
        raise CalderArtifactError("canonical Calder card has no graph section")
    for name, value in (
        ("preprocess cache", cfg.preprocess.cache_dir),
        ("graph cache", cfg.graph.cache_dir),
    ):
        if value is None:
            raise CalderArtifactError(f"Calder {name} is not explicit")
        _require_under(Path(value), cache, name=name)

    dataset, _dataset_info = dataset_orchestrator.load(cfg.dataset)
    source_fingerprint = dataset.meta.get("dataset_fingerprint")
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise CalderArtifactError("cached MNIST dataset has no source fingerprint")
    content_evidence = verify_dataset_content(
        cfg.dataset.id,
        cache_dir=dataset_dir,
        options=cfg.dataset.options,
        rehash=True,
    )
    dataset = sampling_orchestrator.prepare_dataset(dataset, plan_dict=cfg.sampling.plan)
    prepared_fingerprint = dataset.meta.get("dataset_fingerprint")
    if not isinstance(prepared_fingerprint, str) or not prepared_fingerprint:
        raise CalderArtifactError("merged MNIST dataset has no fingerprint")
    n_nodes = int(np.asarray(dataset.train.y).shape[0])
    if dataset.test is not None or n_nodes != 70_000:
        raise CalderArtifactError(f"Calder requires one 70,000-node MNIST pool, got {n_nodes}")

    protocol_inputs = _contained_file(
        root,
        _PROTOCOL_INPUTS_RELATIVE / "MANIFEST.json",
        label="Calder protocol manifest",
    ).parent
    official = verify_calder_official_assets(
        protocol_inputs,
        dataset_labels=np.asarray(dataset.train.y),
    )

    sampling_seed = int(cfg.sampling.seed if cfg.sampling.seed is not None else cfg.run.seed)
    sampled = sampling_orchestrator.run(
        dataset,
        plan_dict=cfg.sampling.plan,
        seed=sampling_seed,
        dataset_id=cfg.dataset.id,
    )
    fit_indices = preprocess_orchestrator.resolve_fit_indices(
        dataset=dataset,
        sampling=sampled,
        fit_on=cfg.preprocess.fit_on,
    )
    preprocess_seed = int(cfg.preprocess.seed if cfg.preprocess.seed is not None else cfg.run.seed)
    preprocessed = preprocess_orchestrator.run(
        dataset,
        plan_dict=cfg.preprocess.plan,
        seed=preprocess_seed,
        fit_indices=fit_indices,
        cache=cfg.preprocess.cache,
        cache_dir=cfg.preprocess.cache_dir,
    )

    graph_seed = int(cfg.graph.seed if cfg.graph.seed is not None else cfg.run.seed)
    graph = graph_orchestrator.build(
        preprocessed,
        spec_dict=cfg.graph.spec,
        seed=graph_seed,
        dataset_fingerprint=prepared_fingerprint,
        cache=True,
        require_cache_hit=False,
        cache_dir=cfg.graph.cache_dir,
        include_test=False,
        expected_fingerprint=None,
        expected_preprocess_fingerprint=preprocessed.preprocess_fingerprint,
    )
    graph_fingerprint = graph.meta.get("fingerprint")
    if graph_fingerprint != cfg.graph.expected_fingerprint:
        raise CalderArtifactError("derived graph fingerprint differs from the packaged card")
    if preprocessed.preprocess_fingerprint != cfg.graph.expected_preprocess_fingerprint:
        raise CalderArtifactError(
            "derived preprocessing fingerprint differs from the packaged card"
        )
    if graph.n_nodes != 70_000 or int(graph.edge_index.shape[1]) <= 0:
        raise CalderArtifactError("derived Calder graph has an invalid shape")

    verified_graph = graph_orchestrator.build(
        preprocessed,
        spec_dict=cfg.graph.spec,
        seed=graph_seed,
        dataset_fingerprint=prepared_fingerprint,
        cache=True,
        require_cache_hit=True,
        cache_dir=cfg.graph.cache_dir,
        include_test=False,
        expected_fingerprint=str(graph_fingerprint),
        expected_preprocess_fingerprint=preprocessed.preprocess_fingerprint,
    )
    if verified_graph.meta.get("fingerprint") != graph_fingerprint:
        raise CalderArtifactError("Calder graph cache did not replay deterministically")
    if preprocessed.cache_dir is None:
        raise CalderArtifactError("Calder preprocessing did not report its cache directory")

    preprocess_dir = _require_under(
        Path(preprocessed.cache_dir),
        cache,
        name="derived preprocess artifact",
    )
    graph_dir = _require_under(
        Path(cfg.graph.cache_dir) / str(graph_fingerprint),
        cache,
        name="derived graph artifact",
    )
    environment = {
        name: metadata.version(name)
        for name in ("modssc", "numpy", "scipy")
        if _distribution_version_exists(name)
    }
    payload = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "kind": LOCK_KIND,
        "scientific_payload": scientific_payload_identity(family),
        "environment": environment,
        "dataset": {
            "id": cfg.dataset.id,
            "source_fingerprint": source_fingerprint,
            "prepared_fingerprint": prepared_fingerprint,
            "content_evidence": dict(content_evidence),
            "n_nodes": n_nodes,
            "official_splits_merged": True,
        },
        "protocol": {
            "preprocess_seed": preprocess_seed,
            "graph_seed": graph_seed,
            "graph_fingerprint": graph_fingerprint,
            "preprocess_fingerprint": preprocessed.preprocess_fingerprint,
        },
        "official_inputs": {
            "manifest_sha256": _sha256_file(protocol_inputs / "MANIFEST.json"),
            "commit": official["commit"],
            "knn_sha256": official["knn_sha256"],
            "permutations_sha256": official["permutations_sha256"],
            "permutations_artifact_sha256": official["permutations_artifact_sha256"],
        },
        "artifacts": {
            "preprocess": artifact_tree_inventory(preprocess_dir),
            "graph": artifact_tree_inventory(graph_dir),
        },
        "graph": {
            "n_nodes": int(graph.n_nodes),
            "n_edges": int(graph.edge_index.shape[1]),
        },
    }
    return _seal(payload)


def _distribution_version_exists(name: str) -> bool:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def verify_calder_artifact_lock(
    lock: Mapping[str, Any],
    *,
    package_root: Path,
) -> None:
    """Re-authenticate a portable lock against this installed ModSSC payload."""

    if lock.get("schema_version") != LOCK_SCHEMA_VERSION or lock.get("kind") != LOCK_KIND:
        raise CalderArtifactError("invalid portable Calder artifact lock schema")
    expected = lock.get("lock_sha256")
    unsigned = dict(lock)
    unsigned.pop("lock_sha256", None)
    if not isinstance(expected, str) or _canonical_sha256(unsigned) != expected:
        raise CalderArtifactError("portable Calder artifact lock SHA-256 differs")
    family = load_calder_config_family(package_root)
    if lock.get("scientific_payload") != scientific_payload_identity(family):
        raise CalderArtifactError("Calder lock belongs to a different scientific payload")
    official_inputs = _mapping(lock.get("official_inputs"), name="official_inputs")
    protocol_root = _contained_file(
        package_root,
        _PROTOCOL_INPUTS_RELATIVE / "MANIFEST.json",
        label="Calder protocol manifest",
    ).parent
    official = verify_calder_official_assets(protocol_root)
    expected_official = {
        "manifest_sha256": _sha256_file(protocol_root / "MANIFEST.json"),
        "commit": official["commit"],
        "knn_sha256": official["knn_sha256"],
        "permutations_sha256": official["permutations_sha256"],
        "permutations_artifact_sha256": official["permutations_artifact_sha256"],
    }
    if dict(official_inputs) != expected_official:
        raise CalderArtifactError("packaged Calder inputs differ from the artifact lock")
    dataset = _mapping(lock.get("dataset"), name="dataset")
    if (
        dataset.get("id") != "mnist"
        or dataset.get("n_nodes") != 70_000
        or dataset.get("official_splits_merged") is not True
    ):
        raise CalderArtifactError("Calder lock dataset protocol differs")
    protocol = _mapping(lock.get("protocol"), name="protocol")
    if protocol.get("graph_fingerprint") != (
        "209e8c9a6427fcd1403d76f1111654fc202e92d18d771ab37a5da92e14de693c"
    ) or protocol.get("preprocess_fingerprint") != (
        "preprocess:7d44ae1b3a7f09a1c241a9b5e16ec7ff4502e3b4ef7c8aeadb4a6561caa25f20"
    ):
        raise CalderArtifactError("Calder lock semantic fingerprints differ")
    artifacts = _mapping(lock.get("artifacts"), name="artifacts")
    if set(artifacts) != {"preprocess", "graph"}:
        raise CalderArtifactError("Calder lock has unexpected derived artifacts")
    for name, inventory in artifacts.items():
        verify_artifact_tree(_mapping(inventory, name=f"artifacts.{name}"))


def prepare_calder_artifact_lock(
    *,
    package_root: Path,
    cache_root: Path,
    dataset_cache: Path,
    output: Path,
) -> dict[str, Any]:
    """Prepare or replay a Calder lock using only explicit local paths."""

    root = package_root.expanduser().resolve(strict=True)
    cache = cache_root.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = _require_under(output, cache, name="Calder artifact lock")
    with _exclusive_lock(destination):
        if destination.exists():
            lock = _read_json(destination)
            verify_calder_artifact_lock(lock, package_root=root)
            return lock
        lock = _prepare(
            package_root=root,
            cache_root=cache,
            dataset_cache=dataset_cache,
        )
        verify_calder_artifact_lock(lock, package_root=root)
        _write_immutable_json(destination, lock)
        return lock


def verify_calder_artifact_lock_file(
    path: Path,
    *,
    package_root: Path,
) -> dict[str, Any]:
    lock = _read_json(path.expanduser().resolve(strict=True))
    verify_calder_artifact_lock(lock, package_root=package_root)
    return lock


__all__ = [
    "CALDER_CONFIGS",
    "CANONICAL_CONFIG",
    "CalderArtifactError",
    "CalderConfigFamily",
    "artifact_tree_inventory",
    "load_calder_config_family",
    "materialized_calder_graph_spec",
    "prepare_calder_artifact_lock",
    "scientific_payload_identity",
    "verify_artifact_tree",
    "verify_calder_artifact_lock",
    "verify_calder_artifact_lock_file",
]
