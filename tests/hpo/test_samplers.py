from __future__ import annotations

import math

import numpy as np
import pytest

from modssc.hpo import HpoError, Space
from modssc.hpo.samplers import sample_distribution, sample_from_values, validate_distribution
from modssc.hpo.types import DistributionSpec


def test_random_determinism_and_ranges():
    space = Space.from_dict(
        {
            "a": {"dist": "uniform", "low": 0.0, "high": 1.0},
            "b": {"dist": "randint", "low": 1, "high": 4},
            "c": {"dist": "loguniform", "low": 0.1, "high": 10.0},
            "d": ["x", "y", "z"],
            "e": {"dist": "choice", "values": [0.1, 0.2]},
        }
    )

    trials_a = [t.params for t in space.iter_random(seed=123, n_trials=3)]
    trials_b = [t.params for t in space.iter_random(seed=123, n_trials=3)]
    assert trials_a == trials_b

    for params in trials_a:
        assert 0.0 <= params["a"] <= 1.0
        assert params["b"] in {1, 2, 3}
        assert 0.1 <= params["c"] <= 10.0
        assert params["d"] in {"x", "y", "z"}
        assert params["e"] in {0.1, 0.2}


def test_random_rejects_bad_n_trials():
    space = Space.from_dict({"a": [1]})
    with pytest.raises(HpoError, match="n_trials must be > 0"):
        list(space.iter_random(seed=0, n_trials=0))


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (DistributionSpec(dist=None), "dist must be a non-empty string"),  # type: ignore[arg-type]
        (DistributionSpec(dist=""), "dist must be a non-empty string"),
        (DistributionSpec(dist="uniform", high=1.0), "uniform requires low and high"),
        (DistributionSpec(dist="uniform", low=1.0, high=1.0), "uniform requires low < high"),
        (
            DistributionSpec(dist="uniform", low="bad", high=1.0),
            "uniform.low must be a real number",
        ),
        (
            DistributionSpec(dist="uniform", low=0.0, high=1.0, values=[1]),
            "uniform does not accept values",
        ),
        (DistributionSpec(dist="loguniform", high=1.0), "loguniform requires low and high"),
        (
            DistributionSpec(dist="loguniform", low="bad", high=1.0),
            "loguniform.low must be a real number",
        ),
        (DistributionSpec(dist="loguniform", low=0.0, high=1.0), "loguniform requires low > 0"),
        (DistributionSpec(dist="loguniform", low=2.0, high=1.0), "loguniform requires low < high"),
        (
            DistributionSpec(dist="loguniform", low=0.5, high=1.0, values=[1]),
            "loguniform does not accept values",
        ),
        (DistributionSpec(dist="randint", high=2), "randint requires low and high"),
        (
            DistributionSpec(dist="randint", low=1.5, high=2),
            "randint.low must be an int",
        ),
        (DistributionSpec(dist="randint", low=2, high=2), "randint requires low < high"),
        (
            DistributionSpec(dist="randint", low=1, high=2, values=[1]),
            "randint does not accept values",
        ),
        (DistributionSpec(dist="choice"), "choice requires values"),
        (
            DistributionSpec(dist="choice", values="bad"),
            "choice requires a non-empty list of values",
        ),
        (
            DistributionSpec(dist="choice", values=[], low=1),
            "choice requires a non-empty list of values",
        ),
        (
            DistributionSpec(dist="choice", values=[1], low=1),
            "choice does not accept low/high",
        ),
        (DistributionSpec(dist="nope"), "Unknown dist"),
    ],
)
def test_validate_distribution_rejects(spec, match):
    with pytest.raises(HpoError, match=match):
        validate_distribution(spec)


@pytest.mark.parametrize(
    "spec",
    [
        DistributionSpec(dist="uniform", low=0.0, high=1.0),
        DistributionSpec(dist="loguniform", low=0.1, high=1.0),
        DistributionSpec(dist="randint", low=0, high=5),
        DistributionSpec(dist="choice", values=["a", "b"]),
    ],
)
def test_validate_distribution_accepts(spec):
    validate_distribution(spec)


def test_sample_distribution_branches():
    rng = np.random.default_rng(0)

    uniform = sample_distribution(DistributionSpec(dist="uniform", low=0.0, high=1.0), rng)
    assert 0.0 <= uniform <= 1.0

    loguniform = sample_distribution(DistributionSpec(dist="loguniform", low=0.1, high=1.0), rng)
    assert 0.1 <= loguniform <= 1.0
    assert not math.isnan(loguniform)

    randint = sample_distribution(DistributionSpec(dist="randint", low=1, high=3), rng)
    assert randint in {1, 2}

    choice = sample_distribution(DistributionSpec(dist="choice", values=["a", "b"]), rng)
    assert choice in {"a", "b"}


def test_sample_from_values_validation():
    rng = np.random.default_rng(0)
    with pytest.raises(HpoError, match="values must be a non-empty list"):
        sample_from_values([], rng)
    assert sample_from_values(["a"], rng) == "a"
