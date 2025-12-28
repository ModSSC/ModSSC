from __future__ import annotations

import math
import numbers
from typing import Any

import numpy as np

from .types import DistributionSpec, HpoError


def _require_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise HpoError(f"{name} must be a real number")
    return float(value)


def _require_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise HpoError(f"{name} must be an int")
    return int(value)


def validate_distribution(spec: DistributionSpec) -> None:
    dist = spec.dist
    if not isinstance(dist, str) or not dist.strip():
        raise HpoError("dist must be a non-empty string")

    if dist == "uniform":
        if spec.low is None or spec.high is None:
            raise HpoError("uniform requires low and high")
        low = _require_real(spec.low, name="uniform.low")
        high = _require_real(spec.high, name="uniform.high")
        if low >= high:
            raise HpoError("uniform requires low < high")
        if spec.values is not None:
            raise HpoError("uniform does not accept values")
        return

    if dist == "loguniform":
        if spec.low is None or spec.high is None:
            raise HpoError("loguniform requires low and high")
        low = _require_real(spec.low, name="loguniform.low")
        high = _require_real(spec.high, name="loguniform.high")
        if low <= 0:
            raise HpoError("loguniform requires low > 0")
        if low >= high:
            raise HpoError("loguniform requires low < high")
        if spec.values is not None:
            raise HpoError("loguniform does not accept values")
        return

    if dist == "randint":
        if spec.low is None or spec.high is None:
            raise HpoError("randint requires low and high")
        low = _require_int(spec.low, name="randint.low")
        high = _require_int(spec.high, name="randint.high")
        if low >= high:
            raise HpoError("randint requires low < high")
        if spec.values is not None:
            raise HpoError("randint does not accept values")
        return

    if dist == "choice":
        values = spec.values
        if values is None:
            raise HpoError("choice requires values")
        if not isinstance(values, list) or not values:
            raise HpoError("choice requires a non-empty list of values")
        if spec.low is not None or spec.high is not None:
            raise HpoError("choice does not accept low/high")
        return

    raise HpoError(f"Unknown dist: {dist}")


def sample_from_values(values: list[Any], rng: np.random.Generator) -> Any:
    if not isinstance(values, list) or not values:
        raise HpoError("values must be a non-empty list")
    idx = int(rng.integers(0, len(values)))
    return values[idx]


def sample_distribution(spec: DistributionSpec, rng: np.random.Generator) -> Any:
    validate_distribution(spec)
    dist = spec.dist

    if dist == "uniform":
        low = float(spec.low)
        high = float(spec.high)
        return float(rng.uniform(low, high))
    if dist == "loguniform":
        low = float(spec.low)
        high = float(spec.high)
        return float(math.exp(rng.uniform(math.log(low), math.log(high))))
    if dist == "randint":
        low = int(spec.low)
        high = int(spec.high)
        return int(rng.integers(low, high))
    return sample_from_values(spec.values or [], rng)
