"""Run the article-reproduction cards shipped with ModSSC.

This module deliberately belongs to :mod:`bench`, not ``modssc.experiments``.
The wheel and source checkout both ship the runner, cards, and fixed scientific
inputs. It offers a small, fail-closed interface around those cards without
changing their scientific configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from modssc.data_loader import cache_dir as default_dataset_cache_dir
from modssc.data_loader import dataset_info, download_dataset, verify_dataset_content
from modssc.data_loader.errors import DataLoaderError

from .schema import BenchConfigError, ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
CARDS_ROOT = Path(__file__).resolve().parent / "configs" / "reproductions"
DATASET_INTEGRITY_REGISTRY = (
    Path(__file__).resolve().parent / "assets" / "dataset-integrity-registry.json"
)

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
_PLACEHOLDER_RE = re.compile(r"(?:REPLACE_WITH|PLACEHOLDER|\bTBD\b|\bTODO\b)", re.IGNORECASE)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CACHE_ENV_NAMES = {
    "MODSSC_CACHE_DIR",
    "MODSSC_CACHE_ROOT",
    "MODSSC_DATASET_CACHE_DIR",
    "MODSSC_GRAPH_CACHE_DIR",
    "MODSSC_GRAPH_VIEWS_CACHE_DIR",
    "MODSSC_OUTPUT_DIR",
    "MODSSC_PREPROCESS_CACHE_DIR",
    "MODSSC_SPLIT_CACHE_DIR",
}
_SUPPORTED_ENV_NAMES = _CACHE_ENV_NAMES | {"MODSSC_ROOT"}
_EXTERNAL_CODE_SUFFIXES = {
    ".class",
    ".dll",
    ".dylib",
    ".exe",
    ".jar",
    ".java",
    ".py",
    ".sh",
    ".so",
}
_DEPLOYMENT_CARD_TOKENS = ("a100", "h100", "v100", "canary", "dev", "screening")
_CANONICAL_CARD_IDS = frozenset(
    {
        "co_training/webkb_course_nigam_ghani_2000",
        "democratic_co_learning/adult",
        "democratic_co_learning/vote",
        "fixmatch/cifar10-250",
        "flexmatch/cifar10-250",
        "free_match/cifar10-40",
        "grand/cora",
        "laplace_learning/mnist-table1-1-label-per-class",
        "laplace_learning/mnist-table1-2-label-per-class",
        "laplace_learning/mnist-table1-3-label-per-class",
        "laplace_learning/mnist-table1-4-label-per-class",
        "laplace_learning/mnist-table1-5-label-per-class",
        "poisson_learning/mnist-table1-1-label-per-class",
        "poisson_learning/mnist-table1-2-label-per-class",
        "poisson_learning/mnist-table1-3-label-per-class",
        "poisson_learning/mnist-table1-4-label-per-class",
        "poisson_learning/mnist-table1-5-label-per-class",
        "pseudo_label/mnist",
        "self_training/wine_table3",
        "self_training/wine_table3_confirmation_v2",
        "softmatch/cifar10-250",
        "tri_training/vote_table3_j48",
        "tri_training/wdbc_table3_j48",
    }
)
_DCL_VOTE_LOCK_RELATIVE = Path(
    "bench/campaigns/locks/dcl-vote-zhou-goldman-2004-v1/selected-partitions.json"
)
_DCL_VOTE_REPLAY_ROOT_RELATIVE = Path("bench/campaigns/locks/dcl-vote-zhou-goldman-2004-v1/splits")


class ReproductionRegistryError(ValueError):
    """Raised when reproduction cards cannot be discovered unambiguously."""


@dataclass(frozen=True)
class ReproductionCard:
    """Metadata read directly from one canonical reproduction YAML card."""

    card_id: str
    config_path: Path
    method_id: str
    profile: str
    dataset_id: str
    run_name: str
    repetitions: int

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        try:
            display_path = self.config_path.relative_to(REPO_ROOT)
        except ValueError:
            display_path = self.config_path
        payload["config_path"] = str(display_path)
        return payload


@dataclass(frozen=True)
class VerificationIssue:
    code: str
    location: str
    message: str


@dataclass(frozen=True)
class VerificationReport:
    card: ReproductionCard
    issues: tuple[VerificationIssue, ...]

    @property
    def execution_ready(self) -> bool:
        """Whether static execution prerequisites passed.

        This deliberately says nothing about numerical or protocol equivalence
        with a paper.  Scientific status is established only by the dedicated
        acceptance reports produced after complete runs.
        """

        return not self.issues

    @property
    def ready(self) -> bool:
        """Backward-compatible alias for :attr:`execution_ready`."""

        return self.execution_ready

    def as_dict(self) -> dict[str, Any]:
        return {
            "card": self.card.as_dict(),
            "execution_ready": self.execution_ready,
            "scientific_status": "not_evaluated",
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True)
class PreparationResult:
    card_id: str
    dataset_id: str
    cache_dir: str
    dataset_fingerprint: str | None
    dataset_content_sha256: str | None
    dataset_integrity: str
    protocol_checks: tuple[str, ...]
    dry_run: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetIntegrityPin:
    """Previously observed dataset identity for one exact paper profile."""

    profile: str
    dataset_id: str
    options: Mapping[str, Any]
    fingerprint: str
    content_sha256: str
    evidence: str


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ReproductionRegistryError(f"Cannot read reproduction card {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ReproductionRegistryError(f"Reproduction card must contain a mapping: {path}")
    return dict(value)


def _load_dataset_integrity_registry(
    path: Path = DATASET_INTEGRITY_REGISTRY,
) -> dict[str, DatasetIntegrityPin]:
    """Load the packaged, evidence-backed dataset pins.

    Every runnable paper profile must have one entry. Provider bytes alone are
    insufficient because they do not authenticate the protocol's prepared
    dataset identity.
    """

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproductionRegistryError(
            f"Cannot read dataset-integrity registry {path}: {exc}"
        ) from exc
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ReproductionRegistryError(
            f"Dataset-integrity registry has an unsupported schema: {path}"
        )
    protocols = raw.get("protocols")
    if not isinstance(protocols, Mapping):
        raise ReproductionRegistryError(
            f"Dataset-integrity registry protocols must be a mapping: {path}"
        )
    pins: dict[str, DatasetIntegrityPin] = {}
    for profile, payload in protocols.items():
        if not isinstance(profile, str) or not profile.startswith("paper:"):
            raise ReproductionRegistryError(
                f"Dataset-integrity registry contains an invalid profile: {profile!r}"
            )
        if not isinstance(payload, Mapping):
            raise ReproductionRegistryError(
                f"Dataset-integrity registry entry must be a mapping: {profile}"
            )
        dataset_id = payload.get("dataset_id")
        options = payload.get("options")
        fingerprint = payload.get("fingerprint")
        content_sha256 = payload.get("content_sha256")
        evidence = payload.get("evidence")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ReproductionRegistryError(f"Invalid dataset_id pin for {profile}")
        if not isinstance(options, Mapping):
            raise ReproductionRegistryError(f"Invalid dataset options pin for {profile}")
        if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint):
            raise ReproductionRegistryError(f"Invalid dataset fingerprint pin for {profile}")
        if not isinstance(content_sha256, str) or not _SHA256_RE.fullmatch(content_sha256):
            raise ReproductionRegistryError(f"Invalid dataset content pin for {profile}")
        if not isinstance(evidence, str) or not evidence:
            raise ReproductionRegistryError(f"Missing dataset evidence for {profile}")
        pins[profile] = DatasetIntegrityPin(
            profile=profile,
            dataset_id=dataset_id,
            options=dict(options),
            fingerprint=fingerprint,
            content_sha256=content_sha256,
            evidence=evidence,
        )
    return pins


def _dataset_integrity_pin(profile: str) -> DatasetIntegrityPin | None:
    return _load_dataset_integrity_registry().get(profile)


def _required_mapping(value: Any, *, field: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReproductionRegistryError(f"{path}: {field} must be a mapping")
    return value


def _required_text(value: Any, *, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReproductionRegistryError(f"{path}: {field} must be a non-empty string")
    return value.strip()


def _card_from_path(path: Path, *, cards_root: Path) -> ReproductionCard | None:
    raw = _load_yaml_mapping(path)
    run = _required_mapping(raw.get("run"), field="run", path=path)
    dataset = _required_mapping(raw.get("dataset"), field="dataset", path=path)
    method = _required_mapping(raw.get("method"), field="method", path=path)
    profile = _required_text(method.get("profile"), field="method.profile", path=path)
    run_name = _required_text(run.get("name"), field="run.name", path=path)

    # Diagnostic/screening cards live elsewhere whenever possible.  Keep the
    # guard here so a misplaced diagnostic cannot silently become runnable.
    if not profile.startswith("paper:") or run_name.startswith(("diagnostic_", "screening_")):
        return None

    seeds = run.get("seeds")
    repetitions = len(seeds) if isinstance(seeds, list) and seeds else 1
    relative = path.relative_to(cards_root).with_suffix("")
    return ReproductionCard(
        card_id=relative.as_posix(),
        config_path=path.resolve(),
        method_id=_required_text(method.get("id"), field="method.id", path=path),
        profile=profile,
        dataset_id=_required_text(dataset.get("id"), field="dataset.id", path=path),
        run_name=run_name,
        repetitions=repetitions,
    )


def _canonical_rank(card: ReproductionCard) -> tuple[int, int, str]:
    name = card.config_path.stem.lower()
    deployment_tokens = sum(token in name for token in _DEPLOYMENT_CARD_TOKENS)
    return deployment_tokens, len(card.card_id), card.card_id


def discover_cards(cards_root: Path = CARDS_ROOT) -> tuple[ReproductionCard, ...]:
    """Discover and de-duplicate paper cards from ``bench/configs/reproductions``.

    Cards with the same method, profile, and dataset are execution variants of
    one scientific protocol.  The hardware-neutral, shortest path wins.
    """

    root = cards_root.resolve()
    if not root.is_dir():
        raise ReproductionRegistryError(f"Reproduction cards directory does not exist: {root}")
    use_canonical_registry = root == CARDS_ROOT.resolve()
    selected: dict[tuple[str, str, str], ReproductionCard] = {}
    for path in sorted(root.rglob("*.yaml")):
        card = _card_from_path(path, cards_root=root)
        if card is None:
            continue
        if use_canonical_registry and card.card_id not in _CANONICAL_CARD_IDS:
            continue
        key = card.method_id, card.profile, card.dataset_id
        previous = selected.get(key)
        if previous is None or _canonical_rank(card) < _canonical_rank(previous):
            selected[key] = card
    cards = tuple(sorted(selected.values(), key=lambda item: item.card_id))
    if use_canonical_registry:
        discovered_ids = {card.card_id for card in cards}
        missing = sorted(_CANONICAL_CARD_IDS - discovered_ids)
        if missing:
            raise ReproductionRegistryError(
                "Canonical reproduction card(s) are missing: " + ", ".join(missing)
            )
    return cards


def resolve_card(query: str, cards: Sequence[ReproductionCard] | None = None) -> ReproductionCard:
    """Resolve a card id, exact paper profile, or unambiguous method id."""

    known = tuple(cards) if cards is not None else discover_cards()
    normalized = query.strip()
    exact = [card for card in known if card.card_id == normalized]
    if not exact:
        exact = [
            card
            for card in known
            if card.profile == normalized or card.profile.removeprefix("paper:") == normalized
        ]
    if not exact:
        exact = [card for card in known if card.method_id == normalized]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        choices = ", ".join(card.card_id for card in exact)
        raise ReproductionRegistryError(f"Ambiguous reproduction card {query!r}: {choices}")
    raise ReproductionRegistryError(f"Unknown reproduction card: {query!r}")


def _walk(
    value: Any, path: tuple[str | int, ...] = ()
) -> Iterator[tuple[tuple[str | int, ...], Any]]:
    yield path, value
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk(item, path + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, path + (index,))


def _format_location(path: tuple[str | int, ...]) -> str:
    if not path:
        return "<root>"
    output = ""
    for part in path:
        if isinstance(part, int):
            output += f"[{part}]"
        else:
            output += ("." if output else "") + part
    return output


def _environment_names(value: str) -> set[str]:
    return {match.group(1) or match.group(2) for match in _ENV_RE.finditer(value)}


def _resolve_required_path(value: str, *, repo_root: Path) -> Path | None:
    names = _environment_names(value)
    if names & _CACHE_ENV_NAMES:
        return None
    expanded = value.replace("${MODSSC_ROOT}", str(repo_root)).replace(
        "$MODSSC_ROOT", str(repo_root)
    )
    if _environment_names(expanded):
        return None
    path = Path(expanded).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _resolve_packaged_resource(value: str, *, repo_root: Path) -> tuple[Path | None, str | None]:
    """Resolve a card-owned file and enforce the wheel/repository boundary.

    Dataset/cache locations are runtime state and return ``(None, None)``.
    Card-owned resources must use a relative path or ``${MODSSC_ROOT}``, and
    their fully resolved target must remain below the packaged repository root.
    Resolving before the containment check also rejects symlink escapes.
    """

    names = _environment_names(value)
    if names & _CACHE_ENV_NAMES:
        return None, None
    if _environment_names(value) - _SUPPORTED_ENV_NAMES:
        return None, None
    expanded_user = Path(value).expanduser()
    if expanded_user.is_absolute():
        return None, "literal absolute paths are forbidden; use a packaged relative path"
    expanded = value.replace("${MODSSC_ROOT}", str(repo_root)).replace(
        "$MODSSC_ROOT", str(repo_root)
    )
    if _environment_names(expanded):
        return None, None
    candidate = Path(expanded).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved_root = repo_root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None, f"resource resolves outside the packaged repository: {value}"
    return resolved, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_digest(parent: Mapping[str, Any], key: str) -> str | None:
    candidates = ["sha256"] if key == "path" else [f"{key.removesuffix('_path')}_sha256", "sha256"]
    for candidate in candidates:
        value = parent.get(candidate)
        if isinstance(value, str) and value:
            return value.lower()
    return None


def _path_parent(raw: Mapping[str, Any], path: tuple[str | int, ...]) -> Mapping[str, Any]:
    value: Any = raw
    for part in path[:-1]:
        value = value[part] if isinstance(part, int) else value.get(part)
    return value if isinstance(value, Mapping) else {}


def verify_card(card: ReproductionCard, *, repo_root: Path = REPO_ROOT) -> VerificationReport:
    """Statically verify that a card is checkout-self-contained and runnable."""

    issues: list[VerificationIssue] = []
    try:
        raw = _load_yaml_mapping(card.config_path)
    except ReproductionRegistryError as exc:
        issue = VerificationIssue("E_REPRO_CONFIG", "<root>", str(exc))
        return VerificationReport(card=card, issues=(issue,))

    dataset = raw.get("dataset")
    if not isinstance(dataset, Mapping):
        issues.append(VerificationIssue("E_REPRO_CONFIG", "dataset", "dataset must be a mapping"))
    else:
        if dataset.get("download") is not False:
            issues.append(
                VerificationIssue(
                    "E_REPRO_DOWNLOAD_POLICY",
                    "dataset.download",
                    "paper cards must keep dataset.download=false; use prepare instead",
                )
            )
        try:
            dataset_info(card.dataset_id)
        except DataLoaderError as exc:
            issues.append(VerificationIssue("E_REPRO_DATASET", "dataset.id", str(exc)))
        try:
            pin = _dataset_integrity_pin(card.profile)
        except ReproductionRegistryError as exc:
            issues.append(VerificationIssue("E_REPRO_DATASET_REGISTRY", "dataset", str(exc)))
        else:
            options = dataset.get("options", {})
            if pin is None:
                issues.append(
                    VerificationIssue(
                        "E_REPRO_DATASET_UNPINNED",
                        "dataset",
                        f"no packaged dataset-integrity pin exists for {card.profile}",
                    )
                )
            elif (
                card.dataset_id != pin.dataset_id
                or not isinstance(options, Mapping)
                or dict(options) != dict(pin.options)
            ):
                issues.append(
                    VerificationIssue(
                        "E_REPRO_DATASET_PIN",
                        "dataset",
                        "card dataset id/options differ from the packaged protocol pin",
                    )
                )

    try:
        ExperimentConfig.from_dict(raw)
    except (BenchConfigError, TypeError, ValueError) as exc:
        issues.append(VerificationIssue("E_REPRO_CONFIG", "<root>", str(exc)))

    seen_external: set[str] = set()
    for location, value in _walk(raw):
        label = _format_location(location)
        if isinstance(value, str):
            if _PLACEHOLDER_RE.search(value):
                issues.append(
                    VerificationIssue(
                        "E_REPRO_PLACEHOLDER", label, f"unresolved placeholder: {value}"
                    )
                )
            unknown_env = _environment_names(value) - _SUPPORTED_ENV_NAMES
            if unknown_env:
                issues.append(
                    VerificationIssue(
                        "E_REPRO_ENV",
                        label,
                        f"unsupported environment placeholder(s): {', '.join(sorted(unknown_env))}",
                    )
                )

        key = str(location[-1]) if location else ""
        if isinstance(value, str) and (
            key == "classifier_backend"
            and value.lower() == "weka"
            or "jar" in key.lower()
            or "factory" in key.lower()
        ):
            marker = f"{label}:{value}"
            if marker not in seen_external:
                seen_external.add(marker)
                issues.append(
                    VerificationIssue(
                        "E_REPRO_EXTERNAL_CODE",
                        label,
                        "the protocol requires external executable/source code",
                    )
                )

        if not isinstance(value, str) or not (key == "path" or key.endswith("_path")):
            continue
        required_path, boundary_error = _resolve_packaged_resource(value, repo_root=repo_root)
        if boundary_error is not None:
            issues.append(VerificationIssue("E_REPRO_RESOURCE_BOUNDARY", label, boundary_error))
            continue
        if required_path is None:
            continue
        if required_path.suffix.lower() in _EXTERNAL_CODE_SUFFIXES:
            issues.append(
                VerificationIssue(
                    "E_REPRO_EXTERNAL_CODE",
                    label,
                    f"external executable/source resource is forbidden: {value}",
                )
            )
            continue
        if not required_path.is_file():
            issues.append(
                VerificationIssue(
                    "E_REPRO_RESOURCE_MISSING", label, f"required resource is missing: {value}"
                )
            )
            continue
        expected = _expected_digest(_path_parent(raw, location), key)
        if expected is None or not _SHA256_RE.fullmatch(expected):
            issues.append(
                VerificationIssue(
                    "E_REPRO_RESOURCE_UNPINNED",
                    label,
                    "packaged resources require an explicit 64-character SHA-256 pin",
                )
            )
            continue
        observed = _sha256(required_path)
        if observed != expected:
            issues.append(
                VerificationIssue(
                    "E_REPRO_RESOURCE_HASH",
                    label,
                    f"SHA-256 mismatch: expected {expected}, observed {observed}",
                )
            )

    if _is_dcl_vote_card(card):
        try:
            _, evidence_by_seed = _authenticate_dcl_vote_replays(raw, repo_root=repo_root)
            if len(evidence_by_seed) != 20:
                raise ReproductionRegistryError(
                    "DCL Vote requires exactly 20 authenticated partition replays"
                )
        except ReproductionRegistryError as exc:
            issues.append(
                VerificationIssue(
                    "E_REPRO_DCL_PARTITIONS",
                    "sampling",
                    str(exc),
                )
            )

    return VerificationReport(card=card, issues=tuple(issues))


@contextmanager
def _dataset_cache_environment(cache_dir: Path | None) -> Iterator[Path]:
    configured = os.environ.get("MODSSC_DATASET_CACHE_DIR")
    resolved = cache_dir or (Path(configured) if configured else default_dataset_cache_dir())
    resolved = resolved.expanduser().resolve()
    cache_root = (
        resolved.parent
        if cache_dir is not None
        else Path(os.environ.get("MODSSC_CACHE_ROOT", resolved.parent)).expanduser().resolve()
    )
    defaults = {
        "MODSSC_ROOT": str(REPO_ROOT),
        "MODSSC_CACHE_DIR": str(resolved),
        "MODSSC_CACHE_ROOT": str(cache_root),
        "MODSSC_DATASET_CACHE_DIR": str(resolved),
        "MODSSC_GRAPH_CACHE_DIR": str(cache_root / "graph"),
        "MODSSC_GRAPH_VIEWS_CACHE_DIR": str(cache_root / "graph_views"),
        "MODSSC_OUTPUT_DIR": str(cache_root / "output"),
        "MODSSC_PREPROCESS_CACHE_DIR": str(cache_root / "preprocess"),
        "MODSSC_SPLIT_CACHE_DIR": str(cache_root / "splits"),
    }
    for name in (
        "MODSSC_GRAPH_CACHE_DIR",
        "MODSSC_GRAPH_VIEWS_CACHE_DIR",
        "MODSSC_OUTPUT_DIR",
        "MODSSC_PREPROCESS_CACHE_DIR",
        "MODSSC_SPLIT_CACHE_DIR",
    ):
        if name in os.environ:
            defaults[name] = os.environ[name]
    previous = {name: os.environ.get(name) for name in defaults}
    os.environ.update(defaults)
    try:
        yield resolved
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _require_static_verification(card: ReproductionCard) -> None:
    report = verify_card(card)
    if report.execution_ready:
        return
    details = "; ".join(f"{issue.code}@{issue.location}" for issue in report.issues)
    raise ReproductionRegistryError(f"Card {card.card_id!r} is not self-contained: {details}")


def _prepare_match_splits(
    raw: Mapping[str, Any],
    *,
    dataset: Any,
    repo_root: Path,
) -> str:
    from modssc.sampling.partition_artifact import load_ordered_partition
    from modssc.sampling.plan import OrderedPartitionArtifactSpec

    sampling = _required_mapping(raw.get("sampling"), field="sampling", path=Path("<card>"))
    plan = _required_mapping(sampling.get("plan"), field="sampling.plan", path=Path("<card>"))
    partition = _required_mapping(
        plan.get("partition"), field="sampling.plan.partition", path=Path("<card>")
    )
    artifact_raw = _required_mapping(
        partition.get("ordered_indices_artifact"),
        field="sampling.plan.partition.ordered_indices_artifact",
        path=Path("<card>"),
    )
    materialized = dict(artifact_raw)
    source = materialized.get("path")
    if not isinstance(source, str):
        raise ReproductionRegistryError("Match ordered partition artifact has no path")
    resolved_source = _resolve_required_path(source, repo_root=repo_root)
    if resolved_source is None:
        raise ReproductionRegistryError("Match ordered partition artifact path is not static")
    materialized["path"] = str(resolved_source)
    spec = OrderedPartitionArtifactSpec.from_dict(materialized)

    run = _required_mapping(raw.get("run"), field="run", path=Path("<card>"))
    seeds_raw = run.get("seeds")
    seeds = list(seeds_raw) if isinstance(seeds_raw, list) and seeds_raw else [run.get("seed")]
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ReproductionRegistryError("Match card seeds must be integers")
    train = getattr(dataset, "train", None)
    labels = getattr(train, "y", None)
    if labels is None:
        raise ReproductionRegistryError("Match dataset has no training labels")
    test = getattr(dataset, "test", None)
    test_labels = getattr(test, "y", None)
    n_test = None if test_labels is None else len(test_labels)
    metadata = getattr(dataset, "meta", None)
    fingerprint = metadata.get("dataset_fingerprint") if isinstance(metadata, Mapping) else None
    for seed in seeds:
        load_ordered_partition(
            spec=spec,
            run_seed=int(seed),
            y_train=labels,
            n_test=n_test,
            dataset_fingerprint=str(fingerprint) if fingerprint is not None else None,
        )
    return f"match-splits:{len(seeds)}-seed(s)"


def _is_dcl_vote_card(card: ReproductionCard) -> bool:
    from .campaign.dcl_partition_lock import (
        DCL_DATASET_ID,
        DCL_METHOD_ID,
        DCL_METHOD_PROFILE,
    )

    return (
        card.method_id == DCL_METHOD_ID
        and card.profile == DCL_METHOD_PROFILE
        and card.dataset_id == DCL_DATASET_ID
    )


def _declared_run_seeds(raw: Mapping[str, Any]) -> tuple[int, ...]:
    run = _required_mapping(raw.get("run"), field="run", path=Path("<card>"))
    configured = run.get("seeds")
    if configured is None:
        configured = [run.get("seed")]
    if not isinstance(configured, list) or not configured:
        raise ReproductionRegistryError("paper card run.seeds must be a non-empty list")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in configured
    ):
        raise ReproductionRegistryError("paper card seeds must be non-negative integers")
    seeds = tuple(int(value) for value in configured)
    if len(seeds) != len(set(seeds)):
        raise ReproductionRegistryError("paper card seeds must be unique")
    return seeds


def _authenticate_dcl_vote_replays(
    raw: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    dataset_fingerprint: str | None = None,
    dataset_content_sha256: str | None = None,
) -> tuple[Any, dict[int, dict[str, Any]]]:
    """Authenticate the immutable DCL selection and every packaged split."""

    from .campaign.dcl_partition_lock import (
        DCL_SELECTION_COUNT,
        DCL_SELECTION_FILE_SHA256,
        build_task_partition_selection,
        load_dcl_partition_selection,
        verify_dcl_partition_replay,
    )

    root = (repo_root or REPO_ROOT).resolve()
    selection_path = (root / _DCL_VOTE_LOCK_RELATIVE).resolve()
    replay_root = (root / _DCL_VOTE_REPLAY_ROOT_RELATIVE).resolve()
    try:
        lock = load_dcl_partition_selection(
            selection_path,
            expected_sha256=DCL_SELECTION_FILE_SHA256,
            expected_dataset_fingerprint=dataset_fingerprint,
            expected_dataset_content_sha256=dataset_content_sha256,
        )
        configured_seeds = _declared_run_seeds(raw)
        locked_seeds = tuple(entry.seed for entry in lock.selected)
        if configured_seeds != locked_seeds or len(locked_seeds) != DCL_SELECTION_COUNT:
            raise ReproductionRegistryError(
                "DCL Vote card seeds differ from the 20 frozen selected partitions"
            )
        sampling = _required_mapping(raw.get("sampling"), field="sampling", path=Path("<card>"))
        plan = _required_mapping(sampling.get("plan"), field="sampling.plan", path=Path("<card>"))
        evidence_by_seed: dict[int, dict[str, Any]] = {}
        for entry in lock.selected:
            replay_dir = replay_root / f"seed-{entry.seed:03d}"
            evidence = build_task_partition_selection(
                selection_path=str(selection_path),
                lock=lock,
                entry=entry,
                replay_path=str(replay_dir),
            )
            verify_dcl_partition_replay(
                evidence,
                expected_seed=entry.seed,
                expected_dataset_fingerprint=lock.dataset_fingerprint,
                expected_plan=plan,
            )
            evidence_by_seed[entry.seed] = evidence
    except ReproductionRegistryError:
        raise
    except Exception as exc:
        raise ReproductionRegistryError(
            f"DCL Vote frozen-partition authentication failed: {exc}"
        ) from exc
    return lock, evidence_by_seed


def _select_dcl_vote_seeds(
    raw: Mapping[str, Any],
    *,
    seed: int | None,
    num_runs: int | None,
) -> tuple[int, ...]:
    if seed is not None and num_runs is not None:
        raise ValueError("seed and num_runs are mutually exclusive")
    declared = _declared_run_seeds(raw)
    if seed is not None:
        if isinstance(seed, bool) or seed not in declared:
            raise ValueError(f"seed must be one of the frozen DCL Vote seeds: {list(declared)}")
        return (int(seed),)
    if num_runs is not None:
        if isinstance(num_runs, bool) or num_runs <= 0:
            raise ValueError("num_runs must be > 0")
        if num_runs > len(declared):
            raise ValueError(
                f"num_runs cannot exceed the {len(declared)} frozen DCL Vote partitions"
            )
        return declared[:num_runs]
    return declared


def _run_dcl_vote_card(
    card: ReproductionCard,
    raw: dict[str, Any],
    *,
    seed: int | None,
    num_runs: int | None,
) -> int:
    """Execute DCL Vote only through its authenticated, per-seed split replay."""

    from .context import next_available_run_dir
    from .main import run_experiment_single
    from .orchestrators import reporting as report_orch
    from .seed_sweep import apply_global_seed, sweep_run_name

    cfg = ExperimentConfig.from_dict(raw)
    selected_seeds = _select_dcl_vote_seeds(raw, seed=seed, num_runs=num_runs)
    _, evidence_by_seed = _authenticate_dcl_vote_replays(raw)
    sweep = seed is None
    sweep_root: Path | None = None
    if sweep:
        sweep_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        sweep_root = next_available_run_dir(
            Path(cfg.run.output_dir).expanduser().resolve()
            / f"{cfg.run.name}-sweep-{sweep_timestamp}"
        )

    results: list[Any] = []
    for index, selected_seed in enumerate(selected_seeds):
        run_name = sweep_run_name(
            cfg.run.name,
            seed=selected_seed,
            index=index,
            total=len(selected_seeds),
        )
        effective = apply_global_seed(
            raw,
            seed=selected_seed,
            run_name=run_name,
            seeded_sections=cfg.run.seeded_sections,
        )
        sampling = effective.get("sampling")
        if not isinstance(sampling, dict) or "replay" in sampling:
            raise ReproductionRegistryError(
                "DCL Vote source card must contain sampling without a replay override"
            )
        sampling["replay"] = dict(evidence_by_seed[selected_seed])
        if sweep_root is not None:
            run = effective.get("run")
            if not isinstance(run, dict):
                raise ReproductionRegistryError("DCL Vote effective run must be a mapping")
            run["output_dir"] = str(sweep_root)
        effective_cfg = ExperimentConfig.from_dict(effective)
        result = run_experiment_single(card.config_path, raw=effective, cfg=effective_cfg)
        if result.code != 0:
            return 1
        results.append(result)

    if sweep_root is not None and results:
        report_orch.write_seed_sweep_summary(
            output_dir=sweep_root,
            config_path=card.config_path,
            base_name=cfg.run.name,
            requested_seeds=list(selected_seeds),
            run_json_paths=[result.run_json_path for result in results],
        )
    return 0


def _prepare_calder_prerequisites(*, dataset_cache: Path, dry_run: bool) -> tuple[str, ...]:
    protocol_inputs = (REPO_ROOT / "bench/assets/calder2020/protocol_inputs").resolve()
    try:
        from bench.campaign.protocols.calder.official import verify_calder_official_assets
        from bench.campaign.protocols.calder.oracle import verify_calder_numerical_oracle

        verify_calder_official_assets(protocol_inputs)
        verify_calder_numerical_oracle(REPO_ROOT)
    except Exception as exc:
        raise ReproductionRegistryError(
            f"Calder protocol-input authentication failed: {exc}"
        ) from exc
    checks = [
        "calder-protocol-inputs:authenticated",
        "calder-historical-replay-oracle:authenticated",
    ]
    if dry_run:
        checks.append("calder-graph-cache:planned")
        return tuple(checks)

    from bench.campaign.protocols.calder.artifacts import prepare_calder_artifact_lock

    cache_root = dataset_cache.resolve().parent
    lock_path = cache_root / "calder2020" / "artifact-lock-v1.json"
    try:
        lock = prepare_calder_artifact_lock(
            package_root=REPO_ROOT,
            cache_root=cache_root,
            dataset_cache=dataset_cache,
            output=lock_path,
        )
    except Exception as exc:
        raise ReproductionRegistryError(f"Calder prerequisite preparation failed: {exc}") from exc
    pins = lock.get("pins")
    if not isinstance(pins, Mapping) or not pins.get("graph_fingerprint"):
        raise ReproductionRegistryError("Calder artifact lock has no graph fingerprint")
    checks.append(f"calder-graph-cache:{pins['graph_fingerprint']}")
    return tuple(checks)


def _verify_prepared_dataset_integrity(
    card: ReproductionCard,
    raw: Mapping[str, Any],
    loaded: Any,
    *,
    cache_dir: Path,
    options: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    """Authenticate provider bytes and, where known, the paper dataset identity."""

    from .orchestrators.sampling import prepare_dataset

    try:
        evidence = verify_dataset_content(
            card.dataset_id,
            cache_dir=cache_dir,
            options=dict(options),
            rehash=True,
        )
    except Exception as exc:
        raise ReproductionRegistryError(
            f"Dataset content verification failed for {card.card_id}: {exc}"
        ) from exc
    content_sha256 = evidence.get("content_sha256")
    if not isinstance(content_sha256, str) or not _SHA256_RE.fullmatch(content_sha256):
        raise ReproductionRegistryError(
            f"Dataset content verification returned no SHA-256 for {card.card_id}"
        )
    meta = loaded.meta if isinstance(getattr(loaded, "meta", None), Mapping) else {}
    if meta.get("dataset_content_sha256") != content_sha256:
        raise ReproductionRegistryError(
            f"Loaded dataset content evidence differs from the verified cache for {card.card_id}"
        )
    sampling = _required_mapping(raw.get("sampling"), field="sampling", path=card.config_path)
    plan = _required_mapping(sampling.get("plan"), field="sampling.plan", path=card.config_path)
    try:
        prepared = prepare_dataset(loaded, plan_dict=plan)
    except Exception as exc:
        raise ReproductionRegistryError(
            f"Dataset protocol preparation failed for {card.card_id}: {exc}"
        ) from exc
    prepared_meta = prepared.meta if isinstance(getattr(prepared, "meta", None), Mapping) else {}
    fingerprint = prepared_meta.get("dataset_fingerprint")
    if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint):
        raise ReproductionRegistryError(
            f"Prepared dataset has no SHA-256 fingerprint for {card.card_id}"
        )

    pin = _dataset_integrity_pin(card.profile)
    if pin is None:
        raise ReproductionRegistryError(
            f"No packaged dataset-integrity pin exists for {card.profile}"
        )
    if card.dataset_id != pin.dataset_id or dict(options) != dict(pin.options):
        raise ReproductionRegistryError(
            f"Dataset request differs from the packaged integrity pin for {card.profile}"
        )
    if fingerprint != pin.fingerprint:
        raise ReproductionRegistryError(
            f"Dataset fingerprint mismatch for {card.profile}: "
            f"expected {pin.fingerprint}, observed {fingerprint}"
        )
    if content_sha256 != pin.content_sha256:
        raise ReproductionRegistryError(
            f"Dataset content mismatch for {card.profile}: "
            f"expected {pin.content_sha256}, observed {content_sha256}"
        )
    return (
        fingerprint,
        content_sha256,
        "paper_identity_authenticated",
        "dataset-integrity:paper-identity-authenticated",
    )


def prepare_card(
    card: ReproductionCard,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> PreparationResult:
    """Prepare the canonical dataset and any protocol-owned immutable inputs."""

    _require_static_verification(card)
    raw = _load_yaml_mapping(card.config_path)
    dataset = _required_mapping(raw.get("dataset"), field="dataset", path=card.config_path)
    options = dataset.get("options")
    if options is not None and not isinstance(options, Mapping):
        raise ReproductionRegistryError(f"{card.config_path}: dataset.options must be a mapping")
    with _dataset_cache_environment(cache_dir) as resolved_cache:
        if dry_run:
            pin = _dataset_integrity_pin(card.profile)
            if pin is None:  # defensive: static verification is fail-closed too
                raise ReproductionRegistryError(
                    f"No packaged dataset-integrity pin exists for {card.profile}"
                )
            dataset_integrity = "paper_identity_planned"
            integrity_check = "dataset-integrity:paper-identity-pin-present"
            protocol_checks: tuple[str, ...] = (
                "dataset:planned",
                integrity_check,
                "resources:sha256",
            )
            if card.method_id in {"laplace_learning", "poisson_learning"}:
                protocol_checks += _prepare_calder_prerequisites(
                    dataset_cache=resolved_cache,
                    dry_run=True,
                )
            elif card.method_id in {"fixmatch", "flexmatch", "free_match", "softmatch"}:
                protocol_checks += ("match-splits:sha256-authenticated",)
            elif _is_dcl_vote_card(card):
                _, evidence_by_seed = _authenticate_dcl_vote_replays(raw)
                protocol_checks += (
                    f"dcl-vote-partitions:{len(evidence_by_seed)}/20-authenticated",
                )
            return PreparationResult(
                card_id=card.card_id,
                dataset_id=card.dataset_id,
                cache_dir=str(resolved_cache),
                dataset_fingerprint=None,
                dataset_content_sha256=None,
                dataset_integrity=dataset_integrity,
                protocol_checks=protocol_checks,
                dry_run=True,
            )

        loaded = download_dataset(
            card.dataset_id,
            cache_dir=resolved_cache,
            force=force,
            options=dict(options or {}),
        )
        fingerprint, content_sha256, dataset_integrity, integrity_check = (
            _verify_prepared_dataset_integrity(
                card,
                raw,
                loaded,
                cache_dir=resolved_cache,
                options=dict(options or {}),
            )
        )
        protocol_checks = ["dataset:cached", integrity_check, "resources:sha256"]
        if card.method_id in {"laplace_learning", "poisson_learning"}:
            protocol_checks.extend(
                _prepare_calder_prerequisites(dataset_cache=resolved_cache, dry_run=False)
            )
        elif card.method_id in {"fixmatch", "flexmatch", "free_match", "softmatch"}:
            try:
                protocol_checks.append(
                    _prepare_match_splits(raw, dataset=loaded, repo_root=REPO_ROOT)
                )
            except Exception as exc:
                raise ReproductionRegistryError(
                    f"Match split authentication failed for {card.card_id}: {exc}"
                ) from exc
        elif _is_dcl_vote_card(card):
            if not isinstance(fingerprint, str) or not isinstance(content_sha256, str):
                raise ReproductionRegistryError(
                    "DCL Vote dataset must expose fingerprint and content SHA-256 metadata"
                )
            _, evidence_by_seed = _authenticate_dcl_vote_replays(
                raw,
                dataset_fingerprint=fingerprint,
                dataset_content_sha256=content_sha256,
            )
            protocol_checks.append(f"dcl-vote-partitions:{len(evidence_by_seed)}/20-authenticated")
    return PreparationResult(
        card_id=card.card_id,
        dataset_id=card.dataset_id,
        cache_dir=str(resolved_cache),
        dataset_fingerprint=fingerprint,
        dataset_content_sha256=content_sha256,
        dataset_integrity=dataset_integrity,
        protocol_checks=tuple(protocol_checks),
        dry_run=False,
    )


def run_card(
    card: ReproductionCard,
    *,
    cache_dir: Path | None = None,
    force_download: bool = False,
    seed: int | None = None,
    num_runs: int | None = None,
) -> int:
    """Verify, prepare, then execute a card through the canonical bench runner."""

    _require_static_verification(card)
    dcl_raw: dict[str, Any] | None = None
    if _is_dcl_vote_card(card):
        dcl_raw = _load_yaml_mapping(card.config_path)
        _select_dcl_vote_seeds(dcl_raw, seed=seed, num_runs=num_runs)
    with _dataset_cache_environment(cache_dir), _working_directory(REPO_ROOT):
        prepare_card(card, cache_dir=cache_dir, force=force_download)
        if dcl_raw is not None:
            return _run_dcl_vote_card(
                card,
                dcl_raw,
                seed=seed,
                num_runs=num_runs,
            )
        from .main import run_experiment

        return run_experiment(card.config_path, seed=seed, num_runs=num_runs)


def _print_card(card: ReproductionCard) -> None:
    for key, value in card.as_dict().items():
        print(f"{key}: {value}")


def _print_report(report: VerificationReport) -> None:
    state = "execution-ready" if report.execution_ready else "execution-blocked"
    print(f"{report.card.card_id}: {state}; scientific-status=not-evaluated")
    for issue in report.issues:
        print(f"  {issue.code} {issue.location}: {issue.message}")


def _add_card_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("card", help="Card id, paper profile, or unambiguous method id.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and run ModSSC article reproductions")
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="List canonical paper cards.")
    list_parser.add_argument("--method", default=None, help="Filter by ModSSC method id.")
    list_parser.add_argument("--json", action="store_true", dest="json_output")

    show_parser = commands.add_parser("show", help="Show one paper card.")
    _add_card_argument(show_parser)
    show_parser.add_argument("--json", action="store_true", dest="json_output")
    show_parser.add_argument("--raw", action="store_true", help="Print the immutable YAML card.")

    prepare_parser = commands.add_parser("prepare", help="Download/cache a card's dataset.")
    _add_card_argument(prepare_parser)
    prepare_parser.add_argument(
        "--cache-dir",
        "--dataset-cache-dir",
        type=Path,
        default=None,
        help="Canonical dataset-cache directory; other caches default beside it.",
    )
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.add_argument(
        "--dry-run", action="store_true", help="Authenticate static inputs without downloading."
    )
    prepare_parser.add_argument("--json", action="store_true", dest="json_output")

    run_parser = commands.add_parser("run", help="Verify, prepare, and run one paper card.")
    _add_card_argument(run_parser)
    run_parser.add_argument(
        "--cache-dir",
        "--dataset-cache-dir",
        type=Path,
        default=None,
        help="Canonical dataset-cache directory; other caches default beside it.",
    )
    run_parser.add_argument("--force-download", action="store_true")
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Verify and print the preparation plan only."
    )
    run_sweep = run_parser.add_mutually_exclusive_group()
    run_sweep.add_argument("--seed", type=int, default=None)
    run_sweep.add_argument("--num-runs", type=int, default=None)

    verify_parser = commands.add_parser("verify", help="Check self-contained runtime inputs.")
    verify_parser.add_argument("card", nargs="?", default=None)
    verify_parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        cards = discover_cards()
        if args.command == "list":
            selected = [
                card for card in cards if args.method is None or card.method_id == args.method
            ]
            if args.json_output:
                print(json.dumps([card.as_dict() for card in selected], indent=2, sort_keys=True))
            else:
                for card in selected:
                    print(
                        f"{card.card_id}\t{card.method_id}\t{card.dataset_id}\t"
                        f"{card.repetitions} repetition(s)"
                    )
            return 0

        if args.command == "verify":
            selected = cards if args.card is None else (resolve_card(args.card, cards),)
            reports = [verify_card(card) for card in selected]
            if args.json_output:
                print(
                    json.dumps([report.as_dict() for report in reports], indent=2, sort_keys=True)
                )
            else:
                for report in reports:
                    _print_report(report)
            return 0 if all(report.execution_ready for report in reports) else 2

        card = resolve_card(args.card, cards)
        if args.command == "show":
            if args.raw:
                print(card.config_path.read_text(encoding="utf-8"), end="")
            elif args.json_output:
                print(json.dumps(card.as_dict(), indent=2, sort_keys=True))
            else:
                _print_card(card)
            return 0
        if args.command == "prepare":
            result = prepare_card(
                card,
                cache_dir=args.cache_dir,
                force=args.force,
                dry_run=args.dry_run,
            )
            if args.json_output:
                print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
            else:
                state = (
                    "execution inputs planned" if result.dry_run else "execution inputs prepared"
                )
                print(
                    f"{result.card_id}: {state} in {result.cache_dir}; "
                    f"dataset-integrity={result.dataset_integrity}; "
                    "scientific-status=not-evaluated"
                )
                for check in result.protocol_checks:
                    print(f"  {check}")
            return 0
        if args.command == "run":
            if args.dry_run:
                result = prepare_card(card, cache_dir=args.cache_dir, dry_run=True)
                print(
                    f"{result.card_id}: execution plan ready; "
                    f"dataset-integrity={result.dataset_integrity}; "
                    "scientific-status=not-evaluated"
                )
                for check in result.protocol_checks:
                    print(f"  {check}")
                return 0
            return run_card(
                card,
                cache_dir=args.cache_dir,
                force_download=args.force_download,
                seed=args.seed,
                num_runs=args.num_runs,
            )
    except (DataLoaderError, OSError, ReproductionRegistryError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
