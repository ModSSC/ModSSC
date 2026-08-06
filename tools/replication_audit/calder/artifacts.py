from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import socket
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from tools.hpc.execution_context import is_scheduled_execution


class CalderArtifactError(RuntimeError):
    """Raised when repository-only Calder audit artifacts cannot be frozen safely."""


ExecutionSite = Literal["scheduled", "local"]


LOCK_KIND = "modssc.calder2020-mnist-table1-artifacts"
LOCK_SCHEMA_VERSION = 3
EFFECTIVE_CONFIG_KIND = "modssc.calder2020-mnist-table1-effective-configs"
CANONICAL_CONFIG = Path("poisson_learning/mnist-table1-1-label-per-class.yaml")
CALDER_CONFIGS = tuple(
    Path(method) / f"mnist-table1-{budget}-label-per-class.yaml"
    for method in ("laplace_learning", "poisson_learning")
    for budget in range(1, 6)
)
_REPRODUCTIONS_RELATIVE = Path("bench/configs/reproductions")

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
    "precomputed_sha256": ("5b42bb234888c83eed763958a17fdfb8a55c09a2f0071b55a61635d86dc90db5"),
    "feature_field": "features.X",
}


@dataclass(frozen=True)
class CalderConfigFamily:
    canonical_path: Path
    canonical_raw: dict[str, Any]
    files: tuple[dict[str, Any], ...]


def materialized_calder_graph_spec(raw_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Expand the already-verified Calder graph card into its explicit form."""

    actual = dict(raw_spec)
    if actual != _EXPECTED_GRAPH_SPEC:
        raise CalderArtifactError("Calder graph specification differs from the frozen protocol")
    from modssc.graph.specs import GraphBuilderSpec

    graph_spec = GraphBuilderSpec.from_dict(actual)
    graph_spec.validate()
    return graph_spec.to_dict()


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


def load_calder_config_family(repo_root: Path) -> CalderConfigFamily:
    """Load and validate the ten cards that must share one VAE and graph."""

    root = repo_root.expanduser().resolve()
    reproduction_root = root / _REPRODUCTIONS_RELATIVE
    loaded: dict[Path, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for relative in CALDER_CONFIGS:
        path = reproduction_root / relative
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise CalderArtifactError(f"cannot read Calder configuration: {path}") from exc
        if not isinstance(raw, dict):
            raise CalderArtifactError(f"Calder configuration root is not a mapping: {path}")
        loaded[relative] = raw
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
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
            raise CalderArtifactError(f"Calder card is not pinned to offline MNIST: {relative}")

    preprocess = _mapping(reference_preprocess, name="preprocess")
    plan = _mapping(preprocess.get("plan"), name="preprocess.plan")
    if _step_ids(plan) != _EXPECTED_PREPROCESS_STEPS:
        raise CalderArtifactError("Calder preprocessing step sequence differs from the paper card")
    if plan.get("output_key") != "features.X":
        raise CalderArtifactError("Calder preprocessing must retain flattened MNIST features")
    if preprocess.get("seed") != 1 or preprocess.get("fit_on") != "train":
        raise CalderArtifactError("Calder preprocessing seed/fit scope is not frozen")
    if preprocess.get("cache") is not True:
        raise CalderArtifactError("Calder preprocessing cache must be enabled")

    graph = _mapping(reference_graph, name="graph")
    if graph.get("enabled") is not True or graph.get("cache") is not True:
        raise CalderArtifactError("Calder graph cache must be enabled")
    if graph.get("require_cache_hit") is not True or graph.get("seed") != 1:
        raise CalderArtifactError("Calder graph seed/cache policy is not frozen")
    if graph.get("spec") != _EXPECTED_GRAPH_SPEC:
        raise CalderArtifactError("Calder graph specification differs from the paper card")

    return CalderConfigFamily(
        canonical_path=reproduction_root / CANONICAL_CONFIG,
        canonical_raw=reference,
        files=tuple(records),
    )


def artifact_tree_inventory(root: Path) -> dict[str, Any]:
    """Hash a committed artifact tree without following symlinks."""

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
    actual = artifact_tree_inventory(Path(root))
    if actual != dict(inventory):
        raise CalderArtifactError(f"artifact tree differs from its SHA lock: {root}")


def seal_calder_artifact_lock(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed.pop("lock_sha256", None)
    sealed["lock_sha256"] = _canonical_sha256(sealed)
    return sealed


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish JSON once; an existing different document is never replaced."""

    destination = path.expanduser().resolve()
    rendered = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise CalderArtifactError(f"refusing to replace immutable lock: {destination}")
        return

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            if destination.read_text(encoding="utf-8") != rendered:
                raise CalderArtifactError(
                    f"concurrent immutable lock differs: {destination}"
                ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_immutable_text(path: Path, rendered: str) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise CalderArtifactError(
                f"refusing to replace immutable effective configuration: {destination}"
            )
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            if destination.read_text(encoding="utf-8") != rendered:
                raise CalderArtifactError(
                    f"concurrent effective configuration differs: {destination}"
                ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalderArtifactError(f"invalid Calder artifact lock: {path}") from exc
    if not isinstance(raw, dict):
        raise CalderArtifactError(f"Calder artifact lock root is not a mapping: {path}")
    return raw


def verify_calder_artifact_lock(lock: Mapping[str, Any]) -> None:
    """Re-hash all frozen artifacts and validate their own cache manifests."""

    if lock.get("schema_version") != LOCK_SCHEMA_VERSION or lock.get("kind") != LOCK_KIND:
        raise CalderArtifactError("invalid Calder artifact lock schema")
    expected_lock_hash = lock.get("lock_sha256")
    unsigned = dict(lock)
    unsigned.pop("lock_sha256", None)
    if not isinstance(expected_lock_hash, str) or _canonical_sha256(unsigned) != expected_lock_hash:
        raise CalderArtifactError("Calder artifact lock SHA-256 differs")

    pins = _mapping(lock.get("pins"), name="pins")
    required_pins = (
        "preprocess_fingerprint",
        "graph_fingerprint",
        "official_commit",
        "official_knn_sha256",
        "official_permutations_sha256",
        "permutations_artifact_sha256",
    )
    for name in required_pins:
        if not isinstance(pins.get(name), str) or not pins[name]:
            raise CalderArtifactError(f"Calder artifact lock has no {name}")
    builder = _mapping(lock.get("builder"), name="builder")
    environment = _mapping(builder.get("environment"), name="builder.environment")
    source_identity = _mapping(builder.get("source_identity"), name="builder.source_identity")
    identity_kind = source_identity.get("kind")
    if identity_kind == "git":
        identity_value = source_identity.get("sha")
        identity_matches = (
            isinstance(identity_value, str)
            and identity_value
            and builder.get("git_sha") == identity_value
            and environment.get("git_sha") == identity_value
            and environment.get("git_dirty") is False
        )
    elif identity_kind == "installed_distribution":
        identity_value = source_identity.get("sha256")
        identity_matches = (
            isinstance(identity_value, str)
            and len(identity_value) == 64
            and builder.get("git_sha") is None
            and environment.get("git_sha") is None
            and environment.get("git_dirty") is None
            and environment.get("distribution_sha256") == identity_value
        )
    else:
        identity_matches = False
    if not identity_matches:
        raise CalderArtifactError("Calder artifact builder identity is inconsistent")
    artifacts = _mapping(lock.get("artifacts"), name="artifacts")
    if set(artifacts) != {"protocol_inputs", "preprocess", "graph"}:
        raise CalderArtifactError(
            "Calder artifact lock must contain protocol inputs, preprocess, and graph"
        )
    for inventory in artifacts.values():
        verify_artifact_tree(_mapping(inventory, name="artifact inventory"))

    protocol_inputs_root = Path(
        str(_mapping(artifacts["protocol_inputs"], name="protocol_inputs")["root"])
    )
    graph_root = Path(str(_mapping(artifacts["graph"], name="graph")["root"]))
    graph_manifest = _read_json(graph_root / "manifest.json")
    if graph_manifest.get("fingerprint") != pins.get("graph_fingerprint"):
        raise CalderArtifactError("graph cache manifest fingerprint differs from the lock")
    if graph_manifest.get("preprocess_fingerprint") != pins.get("preprocess_fingerprint"):
        raise CalderArtifactError("graph cache preprocess fingerprint differs from the lock")
    dataset = _mapping(lock.get("dataset"), name="dataset")
    if (
        dataset.get("id") != "mnist"
        or dataset.get("n_nodes") != 70_000
        or dataset.get("official_splits_merged") is not True
    ):
        raise CalderArtifactError("Calder artifact lock dataset protocol differs")
    if graph_manifest.get("dataset_fingerprint") != dataset.get("prepared_fingerprint"):
        raise CalderArtifactError("graph cache dataset fingerprint differs from the lock")
    graph_spec_raw = _mapping(graph_manifest.get("spec"), name="graph cache spec")
    from modssc.graph.fingerprint import fingerprint_dict
    from modssc.graph.specs import GraphBuilderSpec

    graph_spec = GraphBuilderSpec.from_dict(dict(graph_spec_raw))
    graph_spec.validate()
    canonical_graph_spec = graph_spec.to_dict()
    expected_precomputed_path = (
        (protocol_inputs_root / "graph" / "mnist-vae-knn30.npz").expanduser().resolve()
    )
    actual_precomputed_path = (
        Path(str(canonical_graph_spec.get("precomputed_path"))).expanduser().resolve()
    )
    expected_graph_spec = GraphBuilderSpec.from_dict(
        {
            **_EXPECTED_GRAPH_SPEC,
            "precomputed_path": str(expected_precomputed_path),
        }
    ).to_dict()
    comparable_graph_spec = dict(canonical_graph_spec)
    comparable_graph_spec["precomputed_path"] = str(actual_precomputed_path)
    if comparable_graph_spec != expected_graph_spec:
        raise CalderArtifactError("graph cache specification differs from Calder Table 1")
    # Graph fingerprints historically included the absolute artifact path. New
    # caches use the authenticated artifact content as their identity so the
    # same frozen graph remains portable between machines. Keep accepting the
    # legacy representation when verifying already-sealed evidence.
    semantic_graph_spec = graph_spec.fingerprint_payload()
    legacy_spec_fingerprint = fingerprint_dict(canonical_graph_spec)
    semantic_spec_fingerprint = fingerprint_dict(semantic_graph_spec)
    manifest_spec_fingerprint = graph_manifest.get("spec_fingerprint")
    if manifest_spec_fingerprint == legacy_spec_fingerprint:
        fingerprint_graph_spec = canonical_graph_spec
    elif manifest_spec_fingerprint == semantic_spec_fingerprint:
        fingerprint_graph_spec = semantic_graph_spec
    else:
        raise CalderArtifactError("graph cache specification fingerprint differs")
    graph_seed = graph_manifest.get("seed")
    if isinstance(graph_seed, bool) or not isinstance(graph_seed, int):
        raise CalderArtifactError("graph cache seed is missing")
    expected_graph_fingerprint = fingerprint_dict(
        {
            "dataset_fingerprint": graph_manifest["dataset_fingerprint"],
            "preprocess_fingerprint": graph_manifest["preprocess_fingerprint"],
            "spec": fingerprint_graph_spec,
            "seed": graph_seed,
        }
    )
    if graph_manifest["fingerprint"] != expected_graph_fingerprint:
        raise CalderArtifactError("graph cache fingerprint is internally inconsistent")
    protocol = _mapping(lock.get("protocol"), name="protocol")
    if protocol.get("graph_seed") != graph_seed:
        raise CalderArtifactError("graph cache seed differs from the lock")
    pinned_protocol_graph = _mapping(protocol.get("graph"), name="protocol.graph")
    protocol_spec = _mapping(pinned_protocol_graph.get("spec"), name="protocol.graph.spec")
    if protocol_spec != _EXPECTED_GRAPH_SPEC:
        raise CalderArtifactError("Calder artifact lock graph protocol differs")
    if pinned_protocol_graph.get("expected_fingerprint") != pins["graph_fingerprint"]:
        raise CalderArtifactError("protocol graph fingerprint pin differs from the lock")
    if (
        pinned_protocol_graph.get("expected_preprocess_fingerprint")
        != pins["preprocess_fingerprint"]
    ):
        raise CalderArtifactError("protocol preprocess fingerprint pin differs from the lock")
    effective_payload = {
        "dataset": _mapping(protocol.get("dataset"), name="protocol.dataset"),
        "sampling": _mapping(protocol.get("sampling"), name="protocol.sampling"),
        "preprocess": _mapping(protocol.get("preprocess"), name="protocol.preprocess"),
        "graph": pinned_protocol_graph,
    }
    if protocol.get("effective_sha256") != _canonical_sha256(effective_payload):
        raise CalderArtifactError("Calder effective protocol SHA-256 differs")
    from bench.campaign.protocols.calder.official import verify_calder_official_assets

    evidence = verify_calder_official_assets(protocol_inputs_root)
    if evidence["commit"] != pins.get("official_commit"):
        raise CalderArtifactError("official GraphLearning commit differs from the lock")
    if evidence["knn_sha256"] != pins.get("official_knn_sha256"):
        raise CalderArtifactError("official GraphLearning kNN SHA-256 differs from the lock")
    if evidence["permutations_sha256"] != pins.get("official_permutations_sha256"):
        raise CalderArtifactError(
            "official GraphLearning permutations SHA-256 differs from the lock"
        )
    if evidence["permutations_artifact_sha256"] != pins.get("permutations_artifact_sha256"):
        raise CalderArtifactError("safe permutation artifact SHA-256 differs from the lock")


def require_scheduled_compute_node(
    *,
    environ: Mapping[str, str] | None = None,
    hostname: str | None = None,
) -> None:
    """Reject accidental scientific execution outside its scheduled worker."""

    env = os.environ if environ is None else environ
    job_id = env.get("MODSSC_EXECUTION_JOB_ID")
    allocated = env.get("MODSSC_EXECUTION_NODE")
    if not job_id or not allocated:
        raise CalderArtifactError(
            "Calder artifact preparation/verification requires a scheduled compute allocation"
        )
    current = (hostname or socket.gethostname()).split(".", 1)[0]
    if current != allocated:
        raise CalderArtifactError(
            f"refusing Calder workload on {current}; allocated compute node is {allocated}"
        )


def require_execution_site(
    execution_site: ExecutionSite,
    *,
    local_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    hostname: str | None = None,
) -> None:
    """Validate the compute boundary for a scheduler or an explicit local root."""

    env = os.environ if environ is None else environ
    if execution_site == "scheduled":
        require_scheduled_compute_node(environ=env, hostname=hostname)
        return
    if execution_site != "local":
        raise CalderArtifactError(f"unknown Calder execution site: {execution_site!r}")

    if is_scheduled_execution(env):
        raise CalderArtifactError("refusing local Calder execution inside a scheduled allocation")
    if local_root is None:
        raise CalderArtifactError("--local-root is required for local Calder execution")
    root = local_root.expanduser().resolve()
    work_value = env.get("MODSSC_WORK")
    scratch_value = env.get("MODSSC_SCRATCH")
    if not work_value:
        raise CalderArtifactError("MODSSC_WORK is required")
    if not scratch_value:
        raise CalderArtifactError("MODSSC_SCRATCH is required")
    _require_under(Path(work_value), root, name="MODSSC_WORK")
    _require_under(Path(scratch_value), root, name="MODSSC_SCRATCH")


def _require_under(path: Path, root: Path, *, name: str) -> Path:
    resolved = path.expanduser().resolve()
    allowed = root.expanduser().resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise CalderArtifactError(f"{name} must be below {allowed}: {resolved}") from exc
    return resolved


def _required_environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise CalderArtifactError(f"{name} is required")
    return Path(value)


@contextmanager
def _exclusive_preparation_lock(output: Path) -> Iterator[None]:
    import fcntl

    lock_path = output.with_suffix(output.suffix + ".prepare.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CalderArtifactError(
                f"another Calder artifact preparation owns {lock_path}"
            ) from exc
        yield


def _runtime_provenance(repo_root: Path) -> dict[str, Any]:
    from bench.utils.runtime import collect_runtime_versions

    runtime = collect_runtime_versions(repo_root=repo_root)
    git_sha = runtime.get("git_sha")
    if isinstance(git_sha, str) and git_sha and runtime.get("git_dirty") is False:
        runtime["source_identity"] = {"kind": "git", "sha": git_sha}
        return runtime
    distribution_sha256 = runtime.get("distribution_sha256")
    if (
        git_sha is None
        and runtime.get("git_dirty") is None
        and isinstance(distribution_sha256, str)
        and len(distribution_sha256) == 64
    ):
        runtime["source_identity"] = {
            "kind": "installed_distribution",
            "sha256": distribution_sha256,
        }
        return runtime
    raise CalderArtifactError(
        "new Calder artifacts require either a clean Git snapshot or an authenticated "
        "installed ModSSC distribution"
    )


def _prepare_calder_artifacts(repo_root: Path) -> dict[str, Any]:
    import numpy as np

    from bench.campaign.protocols.calder.official import verify_calder_official_assets
    from bench.orchestrators import dataset as dataset_orchestrator
    from bench.orchestrators import graph as graph_orchestrator
    from bench.orchestrators import preprocess as preprocess_orchestrator
    from bench.orchestrators import sampling as sampling_orchestrator
    from bench.schema import ExperimentConfig
    from bench.utils.io import load_yaml
    from modssc.data_loader import verify_dataset_content

    repo_root = repo_root.expanduser().resolve()
    runtime = _runtime_provenance(repo_root)
    family = load_calder_config_family(repo_root)
    raw = load_yaml(family.canonical_path)
    cfg = ExperimentConfig.from_dict(raw)
    if cfg.graph is None:
        raise CalderArtifactError("canonical Calder configuration has no graph section")

    scratch = _required_environment_path("MODSSC_SCRATCH")
    if (
        cfg.dataset.cache_dir is None
        or cfg.preprocess.cache_dir is None
        or cfg.graph.cache_dir is None
    ):
        raise CalderArtifactError("Calder dataset, preprocess, and graph caches must be explicit")
    dataset_cache = _require_under(Path(cfg.dataset.cache_dir), scratch, name="dataset cache")
    _require_under(Path(cfg.preprocess.cache_dir), scratch, name="preprocess cache")
    _require_under(Path(cfg.graph.cache_dir), scratch, name="graph cache")

    dataset, _dataset_info = dataset_orchestrator.load(cfg.dataset)
    source_fingerprint = dataset.meta.get("dataset_fingerprint")
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise CalderArtifactError("cached MNIST dataset has no source fingerprint")
    content_evidence = verify_dataset_content(
        cfg.dataset.id,
        cache_dir=dataset_cache,
        options=cfg.dataset.options,
        rehash=True,
    )
    dataset = sampling_orchestrator.prepare_dataset(dataset, plan_dict=cfg.sampling.plan)
    prepared_fingerprint = dataset.meta.get("dataset_fingerprint")
    if not isinstance(prepared_fingerprint, str) or not prepared_fingerprint:
        raise CalderArtifactError("merged MNIST dataset has no fingerprint")
    n_nodes = int(np.asarray(dataset.train.y).shape[0])
    if dataset.test is not None or n_nodes != 70_000:
        raise CalderArtifactError(
            f"Calder Table 1 requires one 70,000-node MNIST pool, got {n_nodes}"
        )

    precomputed_path = cfg.graph.spec.get("precomputed_path")
    if not isinstance(precomputed_path, str):
        raise CalderArtifactError("Calder graph has no official precomputed_path")
    protocol_inputs_root = Path(precomputed_path).expanduser().resolve().parent.parent
    try:
        official_evidence = verify_calder_official_assets(
            protocol_inputs_root,
            dataset_labels=np.asarray(dataset.train.y),
        )
    except Exception as exc:
        raise CalderArtifactError("Calder protocol inputs failed verification") from exc

    sampling_seed = int(cfg.sampling.seed if cfg.sampling.seed is not None else cfg.run.seed)
    sampling = sampling_orchestrator.run(
        dataset,
        plan_dict=cfg.sampling.plan,
        seed=sampling_seed,
        dataset_id=cfg.dataset.id,
    )
    fit_indices = preprocess_orchestrator.resolve_fit_indices(
        dataset=dataset,
        sampling=sampling,
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
        cache=cfg.graph.cache,
        require_cache_hit=False,
        cache_dir=cfg.graph.cache_dir,
        include_test=False,
        expected_fingerprint=None,
        expected_preprocess_fingerprint=preprocessed.preprocess_fingerprint,
    )
    graph_fingerprint = graph.meta.get("fingerprint")
    if not isinstance(graph_fingerprint, str):
        raise CalderArtifactError("Calder graph did not report a fingerprint")
    if graph.n_nodes != 70_000 or int(graph.edge_index.shape[1]) <= 0:
        raise CalderArtifactError("Calder graph is empty or has the wrong node count")

    verified_graph = graph_orchestrator.build(
        preprocessed,
        spec_dict=cfg.graph.spec,
        seed=graph_seed,
        dataset_fingerprint=prepared_fingerprint,
        cache=True,
        require_cache_hit=True,
        cache_dir=cfg.graph.cache_dir,
        include_test=False,
        expected_fingerprint=graph_fingerprint,
        expected_preprocess_fingerprint=preprocessed.preprocess_fingerprint,
    )
    if verified_graph.meta.get("fingerprint") != graph_fingerprint:
        raise CalderArtifactError("frozen graph cache reloaded a different fingerprint")

    if preprocessed.cache_dir is None:
        raise CalderArtifactError("production preprocessing did not report its cache directory")
    preprocess_cache_dir = _require_under(
        Path(preprocessed.cache_dir), scratch, name="production preprocess cache"
    )
    graph_cache_dir = _require_under(
        Path(cfg.graph.cache_dir) / graph_fingerprint,
        scratch,
        name="graph artifact cache",
    )

    pinned_preprocess = copy.deepcopy(dict(family.canonical_raw["preprocess"]))
    pinned_graph = copy.deepcopy(dict(family.canonical_raw["graph"]))
    pinned_graph["expected_fingerprint"] = graph_fingerprint
    pinned_graph["expected_preprocess_fingerprint"] = preprocessed.preprocess_fingerprint

    payload = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "kind": LOCK_KIND,
        "builder": {
            "git_sha": runtime["git_sha"],
            "source_identity": runtime["source_identity"],
            "environment": runtime,
            "config_files": list(family.files),
        },
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
            "dataset": copy.deepcopy(family.canonical_raw["dataset"]),
            "sampling": copy.deepcopy(family.canonical_raw["sampling"]),
            "preprocess": pinned_preprocess,
            "graph": pinned_graph,
            "effective_sha256": _canonical_sha256(
                {
                    "dataset": family.canonical_raw["dataset"],
                    "sampling": family.canonical_raw["sampling"],
                    "preprocess": pinned_preprocess,
                    "graph": pinned_graph,
                }
            ),
        },
        "pins": {
            "preprocess_fingerprint": preprocessed.preprocess_fingerprint,
            "graph_fingerprint": graph_fingerprint,
            "official_commit": official_evidence["commit"],
            "official_knn_sha256": official_evidence["knn_sha256"],
            "official_permutations_sha256": official_evidence["permutations_sha256"],
            "permutations_artifact_sha256": official_evidence["permutations_artifact_sha256"],
        },
        "graph": {
            "n_nodes": int(graph.n_nodes),
            "n_edges": int(graph.edge_index.shape[1]),
        },
        "artifacts": {
            "protocol_inputs": artifact_tree_inventory(protocol_inputs_root),
            "preprocess": artifact_tree_inventory(preprocess_cache_dir),
            "graph": artifact_tree_inventory(graph_cache_dir),
        },
        "official_evidence": official_evidence,
    }
    lock = seal_calder_artifact_lock(payload)
    verify_calder_artifact_lock(lock)
    return lock


def prepare_calder_artifact_lock(
    *,
    repo_root: Path,
    output: Path,
    execution_site: ExecutionSite = "local",
    local_root: Path | None = None,
) -> dict[str, Any]:
    require_execution_site(execution_site, local_root=local_root)
    work = _required_environment_path("MODSSC_WORK")
    destination = _require_under(output, work, name="Calder artifact lock")
    with _exclusive_preparation_lock(destination):
        if destination.exists():
            existing = _read_json(destination)
            verify_calder_artifact_lock(existing)
            return existing
        lock = _prepare_calder_artifacts(repo_root)
        write_immutable_json(destination, lock)
        return lock


def verify_calder_artifact_lock_file(
    path: Path,
    *,
    execution_site: ExecutionSite = "local",
    local_root: Path | None = None,
) -> dict[str, Any]:
    require_execution_site(execution_site, local_root=local_root)
    work = _required_environment_path("MODSSC_WORK")
    resolved = _require_under(path, work, name="Calder artifact lock")
    lock = _read_json(resolved)
    verify_calder_artifact_lock(lock)
    return lock


def materialize_calder_effective_configs(
    *,
    repo_root: Path,
    lock: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Write immutable cards whose graph cache pins come from a verified lock."""

    verify_calder_artifact_lock(lock)
    family = load_calder_config_family(repo_root)
    pins = _mapping(lock.get("pins"), name="pins")
    builder = _mapping(lock.get("builder"), name="builder")
    builder_identity = _mapping(builder.get("source_identity"), name="builder.source_identity")
    identity_kind = builder_identity.get("kind")
    if identity_kind == "git":
        identity_value = builder_identity.get("sha")
    elif identity_kind == "installed_distribution":
        identity_value = builder_identity.get("sha256")
    else:
        raise CalderArtifactError("Calder artifact lock has an invalid builder identity")
    if not isinstance(identity_value, str) or not identity_value:
        raise CalderArtifactError("Calder artifact lock has no authenticated builder identity")
    if builder.get("config_files") != list(family.files):
        raise CalderArtifactError(
            "Calder source configurations differ from the artifact builder snapshot"
        )
    root = repo_root.expanduser().resolve()
    destination = _require_under(
        output_dir,
        root / "bench/generated",
        name="Calder effective configuration directory",
    )
    records: list[dict[str, Any]] = []
    reproduction_root = root / _REPRODUCTIONS_RELATIVE
    for relative in CALDER_CONFIGS:
        source = reproduction_root / relative
        try:
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise CalderArtifactError(f"cannot read Calder configuration: {source}") from exc
        if not isinstance(raw, dict):
            raise CalderArtifactError(f"Calder configuration root is not a mapping: {source}")
        graph = dict(_mapping(raw.get("graph"), name=f"{relative}.graph"))
        graph["spec"] = materialized_calder_graph_spec(
            _mapping(graph.get("spec"), name=f"{relative}.graph.spec")
        )
        graph["expected_fingerprint"] = pins["graph_fingerprint"]
        graph["expected_preprocess_fingerprint"] = pins["preprocess_fingerprint"]
        graph["require_cache_hit"] = True
        raw["graph"] = graph
        rendered = yaml.safe_dump(raw, sort_keys=False)
        target = destination / relative
        _write_immutable_text(target, rendered)
        records.append(
            {
                "path": relative.as_posix(),
                "repo_path": target.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            }
        )

    payload = {
        "schema_version": 1,
        "kind": EFFECTIVE_CONFIG_KIND,
        "artifact_lock_sha256": lock["lock_sha256"],
        "artifact_builder": dict(builder_identity),
        "source_configs": list(family.files),
        "pins": {
            "preprocess_fingerprint": pins["preprocess_fingerprint"],
            "graph_fingerprint": pins["graph_fingerprint"],
        },
        "configs": records,
    }
    manifest = seal_calder_artifact_lock(payload)
    write_immutable_json(destination / "MANIFEST.json", manifest)
    return manifest


def _summary(lock: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "lock_sha256": lock.get("lock_sha256"),
        "pins": lock.get("pins"),
        "graph": lock.get("graph"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or verify the shared Calder 2020 MNIST Table 1 artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repo-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--execution-site", choices=("scheduled", "local"), default="local")
    prepare.add_argument("--local-root", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--execution-site", choices=("scheduled", "local"), default="local")
    verify.add_argument("--local-root", type=Path)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--repo-root", type=Path, required=True)
    materialize.add_argument("--lock", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.add_argument("--execution-site", choices=("scheduled", "local"), default="local")
    materialize.add_argument("--local-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            lock = prepare_calder_artifact_lock(
                repo_root=args.repo_root,
                output=args.output,
                execution_site=args.execution_site,
                local_root=args.local_root,
            )
        elif args.command == "verify":
            lock = verify_calder_artifact_lock_file(
                args.output,
                execution_site=args.execution_site,
                local_root=args.local_root,
            )
        else:
            lock = verify_calder_artifact_lock_file(
                args.lock,
                execution_site=args.execution_site,
                local_root=args.local_root,
            )
            work = _required_environment_path("MODSSC_WORK")
            output_dir = _require_under(
                args.output_dir,
                work,
                name="Calder effective configuration directory",
            )
            manifest = materialize_calder_effective_configs(
                repo_root=args.repo_root,
                lock=lock,
                output_dir=output_dir,
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
    except CalderArtifactError as exc:
        parser.exit(2, f"calder-artifacts: {exc}\n")
    print(json.dumps(_summary(lock), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CALDER_CONFIGS",
    "CANONICAL_CONFIG",
    "CalderArtifactError",
    "CalderConfigFamily",
    "artifact_tree_inventory",
    "load_calder_config_family",
    "materialized_calder_graph_spec",
    "materialize_calder_effective_configs",
    "prepare_calder_artifact_lock",
    "require_execution_site",
    "require_scheduled_compute_node",
    "seal_calder_artifact_lock",
    "verify_artifact_tree",
    "verify_calder_artifact_lock",
    "verify_calder_artifact_lock_file",
    "write_immutable_json",
]
