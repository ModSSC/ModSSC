"""Authentication for the packaged Calder numerical parity oracle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bench.campaign.protocols.calder.official import (
    OFFICIAL_COMMIT,
    OFFICIAL_KNN_SHA256,
    OFFICIAL_PERMUTATIONS_SHA256,
)

NUMERICAL_ORACLE_RELATIVE = Path(
    "bench/assets/calder2020/reference_oracles/laplace-b5-permutation0-source-replay.json"
)
NUMERICAL_ORACLE_SHA256 = "cb6f0a7f05b375a35f1606694a8990021bb1ebdde59ef84e84c32bb113535653"
NUMERICAL_ORACLE_KIND = "modssc.calder2020-laplace-source-replay-oracle"
MODSSC_MODULE = "modssc.transductive.methods.classic.laplace_learning"
AUDITED_MODSSC_SOURCE_SHA256 = "4d2d8042c93d961f94871a8e8a26a9d5230ad35c99e454cf509a8616647887f0"
PREDICTION_SHA256 = "bef12df18225f501da01f4522115748f3d63a7b1568734c172d81acae7482432"
SCORE_SHA256 = "9a8150ef38e5a2c02f1bcc326f98c01ec9455a8aa30a45714df75f832591c059"


class CalderNumericalOracleError(RuntimeError):
    """Raised when the packaged numerical parity oracle differs."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalderNumericalOracleError(f"{label} must be a mapping")
    return value


def verify_calder_numerical_oracle(package_root: Path) -> dict[str, Any]:
    """Authenticate sealed historical parity evidence without its source tree.

    The recorded ModSSC source digest identifies the implementation snapshot
    used for the completed source replay.  It deliberately does not claim to be
    the digest of the evolving installed module; current behavior is protected
    separately by the packaged protocol inputs and live numerical tests.
    """

    root = package_root.expanduser().resolve(strict=True)
    candidate = root / NUMERICAL_ORACLE_RELATIVE
    if candidate.is_symlink():
        raise CalderNumericalOracleError("Calder numerical oracle must not be a symlink")
    try:
        path = candidate.resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise CalderNumericalOracleError("Calder numerical oracle is missing") from exc
    if not path.is_file() or _sha256_file(path) != NUMERICAL_ORACLE_SHA256:
        raise CalderNumericalOracleError("Calder numerical oracle SHA-256 differs")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalderNumericalOracleError("Calder numerical oracle is invalid JSON") from exc
    if not isinstance(raw, dict):
        raise CalderNumericalOracleError("Calder numerical oracle root must be a mapping")
    seal = raw.get("oracle_sha256")
    unsigned = dict(raw)
    unsigned.pop("oracle_sha256", None)
    if not isinstance(seal, str) or _canonical_sha256(unsigned) != seal:
        raise CalderNumericalOracleError("Calder numerical oracle seal differs")

    identity = _mapping(raw.get("identity"), label="identity")
    bindings = _mapping(raw.get("bindings"), label="bindings")
    modssc_source = _mapping(bindings.get("modssc_source"), label="modssc_source")
    replay = _mapping(raw.get("replay"), label="replay")
    predictions = _mapping(replay.get("prediction_sha256"), label="prediction_sha256")
    scores = _mapping(replay.get("score_sha256"), label="score_sha256")
    if (
        raw.get("schema_version") != 1
        or raw.get("kind") != NUMERICAL_ORACLE_KIND
        or identity
        != {
            "method_id": "laplace_learning",
            "budget_per_class": 5,
            "permutation": 0,
            "fixed_permutation_row": 4,
        }
        or bindings.get("official_commit") != OFFICIAL_COMMIT
        or bindings.get("official_graph_sha256") != OFFICIAL_KNN_SHA256
        or bindings.get("official_permutations_sha256") != OFFICIAL_PERMUTATIONS_SHA256
        or modssc_source.get("module") != MODSSC_MODULE
        or modssc_source.get("sha256") != AUDITED_MODSSC_SOURCE_SHA256
        or replay.get("prediction_count") != 70_000
        or replay.get("differing_predictions") != 0
        or predictions != {"official_source": PREDICTION_SHA256, "modssc": PREDICTION_SHA256}
        or scores != {"official_source": SCORE_SHA256, "modssc": SCORE_SHA256}
        or replay.get("max_absolute_score_delta") != 0.0
    ):
        raise CalderNumericalOracleError("Calder numerical oracle contents differ")
    return {
        "resource": NUMERICAL_ORACLE_RELATIVE.as_posix(),
        "sha256": NUMERICAL_ORACLE_SHA256,
        "seal_sha256": seal,
        "prediction_sha256": PREDICTION_SHA256,
        "score_sha256": SCORE_SHA256,
        "module": MODSSC_MODULE,
        "scope": "sealed_historical_replay",
        "audited_modssc_source_sha256": AUDITED_MODSSC_SOURCE_SHA256,
    }


__all__ = [
    "AUDITED_MODSSC_SOURCE_SHA256",
    "CalderNumericalOracleError",
    "MODSSC_MODULE",
    "NUMERICAL_ORACLE_KIND",
    "NUMERICAL_ORACLE_RELATIVE",
    "NUMERICAL_ORACLE_SHA256",
    "PREDICTION_SHA256",
    "SCORE_SHA256",
    "verify_calder_numerical_oracle",
]
