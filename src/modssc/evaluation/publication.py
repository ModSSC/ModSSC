"""Pure, allow-listed publication of compact paper-replication results."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Integral, Real
from pathlib import PurePosixPath
from typing import Any, Literal, cast

from modssc.runtime.execution import RunIdentity

from .acceptance import AcceptanceReport
from .aggregation import aggregate_metric_records
from .reconciliation import SeedReconciliation

PublicationTrack = Literal["paper"]
SCHEMA_VERSION = 1
REQUIRED_FILES = frozenset(
    {"SHA256SUMS", "index.md", "manifest.json", "observations.jsonl", "results.json"}
)
MAX_TEXT_FILE_BYTES = 256 * 1024
MAX_OBSERVATIONS_BYTES = 2 * 1024 * 1024
MAX_BUNDLE_BYTES = 5 * 1024 * 1024

_SHA = re.compile(r"[0-9a-f]{64}")
_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_METHOD = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_ERROR = re.compile(r"[A-Z][A-Z0-9_]{0,95}")
_PRIVATE = re.compile(
    r"(?:[A-Za-z]:)?[\\/](?:Users|home|linkhome|gpfs|lustre|scratch|work|tmp)"
    r"(?:[\\/]|\b)"
    r"|\bSLURM_[A-Z0-9_]+\b"
    r"|\bTraceback \(most recent call last\):"
    r"|\bfile://",
    re.IGNORECASE,
)
_MANIFEST_KEYS = {
    "schema_version",
    "release_id",
    "track",
    "created_at",
    "supersedes",
    "source",
    "raw_archive",
    "integrity",
    "cards",
}
_SOURCE_KEYS = {
    "git_commit",
    "git_tree",
    "clean",
    "distribution_sha256",
    "environment_manifest_sha256",
}
_ARCHIVE_KEYS = {
    "archive_id",
    "archive_ref",
    "format_version",
    "manifest_sha256",
    "archive_sha256",
    "bytes",
    "verified_after_transfer",
}
_CARD_KEYS = {
    "card_id",
    "card_path",
    "card_sha256",
    "method_id",
    "dataset",
    "requested_seeds",
    "effective_config_sha256_by_seed",
    "protocol_sha256_by_seed",
    "software_sha256_by_seed",
    "execution_identity_sha256_by_seed",
}
_RESULT_KEYS = {"schema_version", "release_id", "cards"}
_RESULT_CARD_KEYS = {"card_id", "reconciliation", "metrics", "acceptance"}
_RECON_KEYS = {
    "status",
    "certifiable",
    "execution_identity_complete",
    "requested_seeds",
    "categories",
}
_CATEGORIES = ("success", "failed", "not_evaluable", "missing")
_OBS_KEYS = {
    "card_id",
    "seed",
    "status",
    "run_id",
    "error_code",
    "metrics",
    "run_time_seconds",
    "protocol_sha256",
    "software_sha256",
    "execution_identity_sha256",
    "source_run_sha256",
}
_ACCEPTANCE_KEYS = {
    "schema_version",
    "protocol_id",
    "method_id",
    "assessment_status",
    "fidelity_status",
    "fidelity_ceiling",
    "repetitions_expected",
    "runs",
    "conformity",
    "primary_target",
    "secondary_targets",
    "informational_targets",
    "diagnostic_targets",
    "required_diagnostics",
    "diagnostic_failures",
    "deviations",
    "equivalences",
    "unknowns",
    "reasons",
    "acceptance_sha256",
}
_INTEGRITY = {
    "digest_algorithm": "sha256",
    "serialization": "canonical-json-v1",
    "text_encoding": "utf-8",
    "line_endings": "LF",
}


class PublicationError(ValueError):
    """Raised when publication would be ambiguous, unsafe, or unverifiable."""


@dataclass(frozen=True)
class PublicationSource:
    git_commit: str
    git_tree: str
    clean: bool
    distribution_sha256: str
    environment_manifest_sha256: str


@dataclass(frozen=True)
class PublicationRawArchive:
    archive_id: str
    archive_ref: str
    format_version: int
    manifest_sha256: str
    archive_sha256: str
    bytes: int
    verified_after_transfer: bool


@dataclass(frozen=True)
class PaperPublicationCard:
    card_id: str
    card_path: str
    card_sha256: str
    method_id: str
    dataset_id: str
    dataset_fingerprint: str
    reconciliation: SeedReconciliation
    effective_config_sha256_by_seed: Mapping[int, str]
    protocol_sha256_by_seed: Mapping[int, str]
    source_run_sha256_by_seed: Mapping[int, str]
    acceptance: AcceptanceReport | None = None


@dataclass(frozen=True)
class PublicationVerification:
    release_id: str
    track: PublicationTrack
    card_count: int
    observation_count: int
    certifiable_card_count: int


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationError(message)


def _map(value: Any, field: str) -> Mapping[str, Any]:
    _need(isinstance(value, Mapping), f"{field} must be a mapping")
    return cast(Mapping[str, Any], value)


def _seq(value: Any, field: str) -> Sequence[Any]:
    _need(
        isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray),
        f"{field} must be a sequence",
    )
    return cast(Sequence[Any], value)


def _fields(value: Any, expected: set[str], field: str) -> Mapping[str, Any]:
    mapping = _map(value, field)
    _need(set(mapping) == expected, f"{field} fields differ from schema")
    return mapping


def _text(value: Any, field: str, limit: int = 512) -> str:
    _need(isinstance(value, str) and 0 < len(value) <= limit, f"{field} is invalid")
    normalized = cast(str, value)
    _need(
        _PRIVATE.search(normalized) is None, f"{field} contains a private path or operational text"
    )
    _need("\x00" not in normalized, f"{field} contains a NUL byte")
    return normalized


def _match(value: Any, pattern: re.Pattern[str], field: str) -> str:
    normalized = _text(value, field, 128)
    _need(pattern.fullmatch(normalized) is not None, f"{field} has invalid format")
    return normalized


def _sha(value: Any, field: str) -> str:
    return _match(value, _SHA, field)


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    _need(
        isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= minimum,
        f"{field} must be an integer >= {minimum}",
    )
    return int(value)


def _number(value: Any, field: str) -> float:
    _need(
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)),
        f"{field} must be finite",
    )
    return float(value)


def _timestamp(value: Any, field: str) -> str:
    raw = _text(value, field, 64)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PublicationError(f"{field} must be ISO-8601") from exc
    _need(parsed.tzinfo is not None and parsed.utcoffset() is not None, f"{field} needs timezone")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _public(value: Any, field: str) -> Any:
    """Round-trip strict JSON and reject private text anywhere in the value."""

    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise PublicationError(f"{field} is not strict JSON") from exc
    _need(_PRIVATE.search(encoded) is None and "\\u0000" not in encoded, f"{field} is private")
    return decoded


def _ordered(value: Any) -> Any:
    if isinstance(value, Mapping):
        keys = list(value)
        numeric = bool(keys) and all(
            isinstance(key, str) and re.fullmatch(r"0|[1-9][0-9]*", key) is not None for key in keys
        )
        ordered = sorted(keys, key=lambda key: int(key) if numeric else str(key))
        return {key: _ordered(value[key]) for key in ordered}
    if isinstance(value, list | tuple):
        return [_ordered(item) for item in value]
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(_ordered(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode()


def _json_bytes_compat(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _status(categories: Mapping[str, Sequence[int]]) -> str:
    if not any(categories[name] for name in ("failed", "not_evaluable", "missing")):
        return "success"
    if categories["success"]:
        return "partial_failure"
    if categories["not_evaluable"] and not categories["failed"] and not categories["missing"]:
        return "not_evaluable"
    return "failed"


def _seed_map(
    value: Mapping[Any, Any], seeds: tuple[int, ...], field: str, nullable: bool = False
) -> dict[int, str | None]:
    _need(isinstance(value, Mapping), f"{field} must be a mapping")
    normalized: dict[int, str | None] = {}
    for seed, digest in value.items():
        normalized_seed = (
            int(seed)
            if isinstance(seed, str) and re.fullmatch(r"0|[1-9][0-9]*", seed) is not None
            else _integer(seed, f"{field}.seed")
        )
        _need(normalized_seed not in normalized, f"{field} has duplicate normalized seeds")
        normalized[normalized_seed] = None if digest is None else _sha(digest, field)
    _need(
        set(normalized) == set(seeds),
        f"{field} keys must exactly match requested seeds",
    )
    _need(nullable or all(normalized[seed] is not None for seed in seeds), f"{field} forbids null")
    return normalized


def _acceptance_payload(
    value: Mapping[str, Any], method_id: str, seeds: tuple[int, ...], runs: list[dict[str, Any]]
) -> dict[str, Any]:
    payload = dict(_fields(_public(value, "acceptance"), _ACCEPTANCE_KEYS, "acceptance"))
    digest = _sha(payload.pop("acceptance_sha256"), "acceptance.acceptance_sha256")
    _need(
        _digest(_json_bytes_compat(payload)) == digest,
        "acceptance_sha256 digest mismatch",
    )
    payload["acceptance_sha256"] = digest
    _need(payload["schema_version"] == 1, "acceptance schema is unsupported")
    _need(payload["method_id"] == method_id, "acceptance method differs")
    _need(payload["repetitions_expected"] == len(seeds), "acceptance repetitions differ")
    _need(payload["runs"] == runs, "acceptance runs differ")
    _need(
        payload["assessment_status"] in {"passed", "failed", "not_evaluable"}
        and payload["fidelity_status"] in {"paper_matched", "paper_approx", "not_claimable"},
        "acceptance status is invalid",
    )
    return payload


def _source(value: PublicationSource) -> dict[str, Any]:
    _need(isinstance(value, PublicationSource), "source type is invalid")
    _need(value.clean is True, "publication source must be a clean tree")
    return {
        "git_commit": _match(value.git_commit, _OID, "source.git_commit"),
        "git_tree": _match(value.git_tree, _OID, "source.git_tree"),
        "clean": True,
        "distribution_sha256": _sha(value.distribution_sha256, "source.distribution_sha256"),
        "environment_manifest_sha256": _sha(
            value.environment_manifest_sha256, "source.environment_manifest_sha256"
        ),
    }


def _archive(value: PublicationRawArchive) -> dict[str, Any]:
    _need(isinstance(value, PublicationRawArchive), "raw archive type is invalid")
    reference = _text(value.archive_ref, "raw_archive.archive_ref")
    _need(":" in reference and not reference.startswith("file:"), "archive_ref must be a URI")
    _need(value.verified_after_transfer is True, "raw archive is not verified after transfer")
    return {
        "archive_id": _match(value.archive_id, _ID, "raw_archive.archive_id"),
        "archive_ref": reference,
        "format_version": _integer(value.format_version, "raw_archive.format_version", 1),
        "manifest_sha256": _sha(value.manifest_sha256, "raw_archive.manifest_sha256"),
        "archive_sha256": _sha(value.archive_sha256, "raw_archive.archive_sha256"),
        "bytes": _integer(value.bytes, "raw_archive.bytes", 1),
        "verified_after_transfer": True,
    }


def _card_path(value: Any) -> str:
    raw = _text(value, "card_path")
    path = PurePosixPath(raw)
    _need(
        path.as_posix() == raw
        and not path.is_absolute()
        and ".." not in path.parts
        and path.parts[:3] == ("bench", "configs", "reproductions")
        and path.suffix in {".yaml", ".yml"},
        "paper card_path is invalid",
    )
    return raw


def _project_card(card: PaperPublicationCard) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
    _need(isinstance(card, PaperPublicationCard), "card type is invalid")
    _need(isinstance(card.reconciliation, SeedReconciliation), "reconciliation type is invalid")
    card_id = _match(card.card_id, _ID, "card_id")
    method_id = _match(card.method_id, _METHOD, "method_id")
    reconciliation = card.reconciliation
    seeds = tuple(sorted(reconciliation.requested_seeds))
    _need(bool(seeds) and len(seeds) == len(set(seeds)), "requested seeds are invalid")
    categories = {name: sorted(getattr(reconciliation, f"{name}_seeds")) for name in _CATEGORIES}
    members = [seed for name in _CATEGORIES for seed in categories[name]]
    _need(sorted(members) == list(seeds) and len(members) == len(set(members)), "categories differ")
    effective = _seed_map(card.effective_config_sha256_by_seed, seeds, "effective identities")
    protocols = _seed_map(card.protocol_sha256_by_seed, seeds, "protocol identities")
    observed_seeds = tuple(seed for seed in seeds if seed not in categories["missing"])
    source_hashes = _seed_map(card.source_run_sha256_by_seed, observed_seeds, "source runs")
    runs = {_integer(run.get("seed"), "run.seed"): _map(run, "run") for run in reconciliation.runs}
    _need(set(runs) == set(observed_seeds) and len(runs) == len(reconciliation.runs), "runs differ")

    status_by_seed = {seed: name for name in _CATEGORIES for seed in categories[name]}
    observations: list[dict[str, Any]] = []
    acceptance_runs: list[dict[str, Any]] = []
    software: dict[str, str | None] = {}
    executions: dict[str, str | None] = {}
    successful_metrics: list[Mapping[str, Any]] = []
    for seed in seeds:
        status = status_by_seed[seed]
        if status == "missing":
            software[str(seed)] = None
            executions[str(seed)] = None
            observations.append(
                {
                    "card_id": card_id,
                    "seed": seed,
                    "status": status,
                    "run_id": None,
                    "error_code": None,
                    "metrics": None,
                    "run_time_seconds": None,
                    "protocol_sha256": protocols[seed],
                    "software_sha256": None,
                    "execution_identity_sha256": None,
                    "source_run_sha256": None,
                }
            )
            continue
        run = runs[seed]
        _need(run.get("status") == status, "run status differs")
        hashes = _map(run.get("hashes"), "run.hashes")
        effective_digest = _sha(hashes.get("effective_config_hash"), "run.effective_config")
        protocol_digest = _sha(hashes.get("protocol_sha256"), "run.protocol")
        software_digest = _sha(hashes.get("software_sha256"), "run.software")
        _need(
            effective_digest == effective[seed] and protocol_digest == protocols[seed],
            "run identities differ from declared maps",
        )
        try:
            identity = RunIdentity.from_dict(_map(run.get("execution_identity"), "run identity"))
        except (TypeError, ValueError) as exc:
            raise PublicationError("run execution_identity is invalid") from exc
        execution_digest = _sha(run.get("execution_identity_sha256"), "run execution digest")
        _need(
            execution_digest == identity.sha256
            and identity.seed == seed
            and identity.config_sha256 == protocol_digest
            and identity.code_sha256 == software_digest
            and run.get("run_id") == identity.short_id,
            "run execution identity differs",
        )
        metrics = None
        if status == "success":
            metrics = cast(
                dict[str, Any], _public(_map(run.get("metrics"), "run.metrics"), "metrics")
            )
            successful_metrics.append(metrics)
        error_code = run.get("error_code")
        _need(
            error_code is None or _ERROR.fullmatch(str(error_code)) is not None,
            "error code invalid",
        )
        _need(status != "success" or error_code is None, "successful run has error code")
        run_info = {} if run.get("run_info") is None else _map(run.get("run_info"), "run_info")
        runtime = run_info.get("run_time_seconds")
        runtime = None if runtime is None else _number(runtime, "run_time_seconds")
        _need(runtime is None or runtime >= 0, "run time is negative")
        software[str(seed)] = software_digest
        executions[str(seed)] = execution_digest
        observation = {
            "card_id": card_id,
            "seed": seed,
            "status": status,
            "run_id": run["run_id"],
            "error_code": error_code,
            "metrics": metrics,
            "run_time_seconds": runtime,
            "protocol_sha256": protocol_digest,
            "software_sha256": software_digest,
            "execution_identity_sha256": execution_digest,
            "source_run_sha256": source_hashes[seed],
        }
        observations.append(observation)
        acceptance_runs.append({key: observation[key] for key in ("seed", "status", "run_id")})

    aggregate = aggregate_metric_records(successful_metrics)
    declared = _public(reconciliation.metrics, "aggregate metrics")
    _need(_aggregate_equal(aggregate, declared), "aggregate metrics differ")
    _need(
        isinstance(card.acceptance, AcceptanceReport),
        "paper publication requires native acceptance",
    )
    acceptance = _acceptance_payload(card.acceptance.to_dict(), method_id, seeds, acceptance_runs)
    status = _status(categories)
    complete = not categories["missing"]
    _need(reconciliation.status == status, "reconciliation status differs")
    _need(reconciliation.execution_identity_complete == complete, "identity completeness differs")
    acceptance_passed = (
        acceptance["assessment_status"] == "passed"
        and acceptance["fidelity_status"] == "paper_matched"
    )
    return (
        {
            "card_id": card_id,
            "card_path": _card_path(card.card_path),
            "card_sha256": _sha(card.card_sha256, "card_sha256"),
            "method_id": method_id,
            "dataset": {
                "id": _text(card.dataset_id, "dataset.id", 256),
                "fingerprint": _sha(card.dataset_fingerprint, "dataset.fingerprint"),
            },
            "requested_seeds": list(seeds),
            "effective_config_sha256_by_seed": {str(seed): effective[seed] for seed in seeds},
            "protocol_sha256_by_seed": {str(seed): protocols[seed] for seed in seeds},
            "software_sha256_by_seed": software,
            "execution_identity_sha256_by_seed": executions,
        },
        {
            "card_id": card_id,
            "reconciliation": {
                "status": status,
                "certifiable": status == "success" and complete and acceptance_passed,
                "execution_identity_complete": complete,
                "requested_seeds": list(seeds),
                "categories": categories,
            },
            "metrics": aggregate,
            "acceptance": acceptance,
        },
        observations,
    )


def _aggregate_equal(left: Any, right: Any) -> bool:
    def normalized(value: Any) -> Any:
        if isinstance(value, Mapping):
            output = {key: normalized(child) for key, child in value.items()}
            values = output.get("values")
            if isinstance(values, list) and all(isinstance(item, int | float) for item in values):
                output["values"] = sorted(values)
            return output
        if isinstance(value, list):
            return [normalized(child) for child in value]
        return value

    return _json_bytes(normalized(left)) == _json_bytes(normalized(right))


def build_paper_publication(
    *,
    release_id: str,
    created_at: str,
    source: PublicationSource,
    raw_archive: PublicationRawArchive,
    cards: Iterable[PaperPublicationCard],
    index_markdown: str,
    supersedes: str | None = None,
) -> dict[str, bytes]:
    """Return canonical bundle bytes without reading or writing any path."""

    release = _match(release_id, _ID, "release_id")
    replaced = None if supersedes is None else _match(supersedes, _ID, "supersedes")
    _need(replaced != release, "supersedes equals release_id")
    card_values = tuple(cards)
    _need(
        bool(card_values) and all(isinstance(card, PaperPublicationCard) for card in card_values),
        "cards invalid",
    )
    _need(len({card.card_id for card in card_values}) == len(card_values), "card ids duplicate")
    projected = [_project_card(card) for card in sorted(card_values, key=lambda item: item.card_id)]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release,
        "track": "paper",
        "created_at": _timestamp(created_at, "created_at"),
        "supersedes": replaced,
        "source": _source(source),
        "raw_archive": _archive(raw_archive),
        "integrity": dict(_INTEGRITY),
        "cards": [item[0] for item in projected],
    }
    results = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release,
        "cards": [item[1] for item in projected],
    }
    markdown = _text(index_markdown, "index_markdown", MAX_TEXT_FILE_BYTES)
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"
    _need(markdown.startswith("# "), "index_markdown needs a level-one heading")
    files = {
        "index.md": markdown.encode(),
        "manifest.json": _json_bytes(manifest),
        "observations.jsonl": b"".join(
            _json_bytes(observation) for item in projected for observation in item[2]
        ),
        "results.json": _json_bytes(results),
    }
    files["SHA256SUMS"] = _checksum_bytes(files)
    verify_paper_publication(files)
    return files


def _checksum_bytes(files: Mapping[str, bytes]) -> bytes:
    return "".join(f"{_digest(files[name])}  {name}\n" for name in sorted(files)).encode()


def _load(data: bytes, field: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = dict(pairs)
        _need(len(result) == len(pairs), f"{field} has duplicate JSON keys")
        return result

    try:
        return json.loads(
            data.decode(),
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PublicationError(f"{field} has non-finite {value}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise PublicationError(f"{field} is not UTF-8 JSON") from exc


def _validate_manifest(value: Any) -> tuple[str, list[dict[str, Any]]]:
    manifest = _fields(value, _MANIFEST_KEYS, "manifest")
    _need(
        manifest["schema_version"] == 1 and manifest["track"] == "paper",
        "manifest version invalid",
    )
    release = _match(manifest["release_id"], _ID, "manifest.release_id")
    _need(
        _timestamp(manifest["created_at"], "manifest.created_at") == manifest["created_at"],
        "timestamp not canonical",
    )
    _need(
        manifest["supersedes"] is None
        or _match(manifest["supersedes"], _ID, "supersedes") != release,
        "supersedes invalid",
    )
    source = _fields(manifest["source"], _SOURCE_KEYS, "manifest.source")
    _source(PublicationSource(**source))
    archive = _fields(manifest["raw_archive"], _ARCHIVE_KEYS, "manifest.raw_archive")
    _archive(PublicationRawArchive(**archive))
    _need(manifest["integrity"] == _INTEGRITY, "integrity contract differs")
    raw_cards = _seq(manifest["cards"], "manifest.cards")
    _need(bool(raw_cards), "manifest cards empty")
    cards: list[dict[str, Any]] = []
    for raw_card in raw_cards:
        card = _fields(raw_card, _CARD_KEYS, "manifest.card")
        seeds = tuple(
            _integer(seed, "requested seed")
            for seed in _seq(card["requested_seeds"], "requested seeds")
        )
        _need(list(seeds) == sorted(set(seeds)) and bool(seeds), "requested seeds not canonical")
        dataset = _fields(card["dataset"], {"id", "fingerprint"}, "dataset")
        cards.append(
            {
                "card_id": _match(card["card_id"], _ID, "card_id"),
                "method_id": _match(card["method_id"], _METHOD, "method_id"),
                "seeds": seeds,
                "effective": _seed_map(card["effective_config_sha256_by_seed"], seeds, "effective"),
                "protocols": _seed_map(card["protocol_sha256_by_seed"], seeds, "protocols"),
                "software": _seed_map(card["software_sha256_by_seed"], seeds, "software", True),
                "executions": _seed_map(
                    card["execution_identity_sha256_by_seed"], seeds, "executions", True
                ),
            }
        )
        _card_path(card["card_path"])
        _sha(card["card_sha256"], "card_sha256")
        _text(dataset["id"], "dataset.id", 256)
        _sha(dataset["fingerprint"], "dataset.fingerprint")
    _need(
        [card["card_id"] for card in cards] == sorted({card["card_id"] for card in cards}),
        "cards not canonical",
    )
    return release, cards


def _validate_observations(
    value: list[Any], cards: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    expected = [(card["card_id"], seed) for card in cards for seed in card["seeds"]]
    _need(len(value) == len(expected), "observation count differs")
    by_card = {card["card_id"]: [] for card in cards}
    lookup = {card["card_id"]: card for card in cards}
    for raw, pair in zip(value, expected, strict=True):
        obs = dict(_fields(raw, _OBS_KEYS, "observation"))
        seed = _integer(obs["seed"], "observation.seed")
        _need((obs["card_id"], seed) == pair, "observations not canonical")
        card = lookup[obs["card_id"]]
        status = obs["status"]
        _need(status in {*_CATEGORIES}, "observation status invalid")
        _need(
            obs["protocol_sha256"] == card["protocols"][seed],
            "observation protocol differs",
        )
        if status == "missing":
            _need(
                all(
                    obs[name] is None
                    for name in (
                        "run_id",
                        "error_code",
                        "metrics",
                        "run_time_seconds",
                        "software_sha256",
                        "execution_identity_sha256",
                        "source_run_sha256",
                    )
                )
                and card["software"][seed] is None
                and card["executions"][seed] is None,
                "missing observation carries run data",
            )
        else:
            _need(
                re.fullmatch(r"[0-9a-f]{20}", str(obs["run_id"])) is not None,
                "run_id invalid",
            )
            _need(
                obs["software_sha256"] == card["software"][seed] is not None,
                "software differs",
            )
            _need(
                obs["execution_identity_sha256"] == card["executions"][seed] is not None,
                "execution differs from manifest",
            )
            _sha(obs["source_run_sha256"], "source run")
            _need(
                obs["run_time_seconds"] is None or _number(obs["run_time_seconds"], "runtime") >= 0,
                "runtime invalid",
            )
            _need(
                obs["error_code"] is None or _ERROR.fullmatch(str(obs["error_code"])) is not None,
                "error code invalid",
            )
            _need(status != "success" or obs["error_code"] is None, "success has error")
            _need(
                (status == "success") == isinstance(obs["metrics"], Mapping),
                "metrics/status differ",
            )
            _public(obs["metrics"], "observation.metrics")
        by_card[obs["card_id"]].append(obs)
    return by_card


def _validate_results(
    value: Any,
    release: str,
    cards: list[dict[str, Any]],
    observations: Mapping[str, list[dict[str, Any]]],
) -> int:
    results = _fields(value, _RESULT_KEYS, "results")
    _need(
        results["schema_version"] == 1 and results["release_id"] == release,
        "results identity differs",
    )
    raw_cards = _seq(results["cards"], "results.cards")
    _need(len(raw_cards) == len(cards), "result cards differ")
    certified = 0
    for raw_result, card in zip(raw_cards, cards, strict=True):
        result = _fields(raw_result, _RESULT_CARD_KEYS, "result card")
        _need(result["card_id"] == card["card_id"], "result card id differs")
        recon = _fields(result["reconciliation"], _RECON_KEYS, "reconciliation")
        _need(recon["requested_seeds"] == list(card["seeds"]), "result seeds differ")
        category_map = _fields(recon["categories"], set(_CATEGORIES), "categories")
        categories = {name: list(_seq(category_map[name], name)) for name in _CATEGORIES}
        members = [seed for name in _CATEGORIES for seed in categories[name]]
        _need(
            sorted(members) == list(card["seeds"]) and len(members) == len(set(members)),
            "result categories differ",
        )
        status = _status(categories)
        complete = not categories["missing"]
        _need(
            recon["status"] == status and recon["execution_identity_complete"] is complete,
            "reconciliation differs",
        )
        card_obs = observations[card["card_id"]]
        _need(
            {
                name: [obs["seed"] for obs in card_obs if obs["status"] == name]
                for name in _CATEGORIES
            }
            == categories,
            "observation categories differ",
        )
        aggregate = aggregate_metric_records(
            cast(Mapping[str, Any], obs["metrics"])
            for obs in card_obs
            if obs["status"] == "success"
        )
        _need(
            _aggregate_equal(aggregate, _public(result["metrics"], "results.metrics")),
            "results metrics differ",
        )
        acceptance = result["acceptance"]
        _need(acceptance is not None, "paper publication requires native acceptance")
        acceptance_runs = [
            {key: obs[key] for key in ("seed", "status", "run_id")}
            for obs in card_obs
            if obs["status"] != "missing"
        ]
        accepted = _acceptance_payload(
            acceptance, card["method_id"], card["seeds"], acceptance_runs
        )
        acceptance_passed = (
            accepted["assessment_status"] == "passed"
            and accepted["fidelity_status"] == "paper_matched"
        )
        expected = status == "success" and complete and acceptance_passed
        _need(recon["certifiable"] is expected, "certifiable differs")
        certified += int(expected)
    return certified


def verify_paper_publication(files: Mapping[str, bytes]) -> PublicationVerification:
    """Verify exact schema, redaction, identities, aggregates, and checksums."""

    _need(
        isinstance(files, Mapping) and set(files) == REQUIRED_FILES,
        "publication files differ from schema",
    )
    for name, data in files.items():
        _need(
            isinstance(name, str) and "/" not in name and "\\" not in name,
            "file name invalid",
        )
        _need(isinstance(data, bytes), f"{name} is not bytes")
        limit = MAX_OBSERVATIONS_BYTES if name == "observations.jsonl" else MAX_TEXT_FILE_BYTES
        _need(
            len(data) <= limit and b"\r" not in data and data.endswith(b"\n"),
            f"{name} text invalid",
        )
        try:
            text = data.decode()
        except UnicodeDecodeError as exc:
            raise PublicationError(f"{name} is not UTF-8") from exc
        _need(
            _PRIVATE.search(text) is None and "\x00" not in text,
            f"{name} contains a private path or operational text",
        )
    _need(sum(len(data) for data in files.values()) <= MAX_BUNDLE_BYTES, "bundle too large")
    checksum_lines = files["SHA256SUMS"].decode().splitlines()
    expected_names = sorted(REQUIRED_FILES - {"SHA256SUMS"})
    expected_lines = [f"{_digest(files[name])}  {name}" for name in expected_names]
    _need(checksum_lines == expected_lines, "SHA256SUMS digest mismatch")
    _need(files["index.md"].startswith(b"# "), "index.md heading invalid")

    manifest = _load(files["manifest.json"], "manifest")
    results = _load(files["results.json"], "results")
    _need(_json_bytes(manifest) == files["manifest.json"], "manifest is not canonical JSON")
    _need(_json_bytes(results) == files["results.json"], "results is not canonical JSON")
    observations: list[Any] = []
    for line in files["observations.jsonl"].splitlines(keepends=True):
        parsed = _load(line, "observation")
        _need(_json_bytes(parsed) == line, "observation is not canonical")
        observations.append(parsed)
    release, cards = _validate_manifest(manifest)
    by_card = _validate_observations(observations, cards)
    certified = _validate_results(results, release, cards, by_card)
    return PublicationVerification(release, "paper", len(cards), len(observations), certified)


__all__ = [
    "MAX_BUNDLE_BYTES",
    "MAX_OBSERVATIONS_BYTES",
    "MAX_TEXT_FILE_BYTES",
    "PaperPublicationCard",
    "PublicationError",
    "PublicationRawArchive",
    "PublicationSource",
    "PublicationTrack",
    "PublicationVerification",
    "REQUIRED_FILES",
    "SCHEMA_VERSION",
    "build_paper_publication",
    "verify_paper_publication",
]
