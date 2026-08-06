from __future__ import annotations

from dataclasses import dataclass

import pytest

from bench.errors import BenchRuntimeError
from bench.orchestrators.method_profile import bind_method_profile


@dataclass(frozen=True)
class _PlainSpec:
    value: int = 1


def test_profile_binding_keeps_campaign_identity_out_of_library_specs() -> None:
    spec = _PlainSpec()
    bound = bind_method_profile(
        spec,
        profile="paper:canonical",
        params={"value": 2},
    )
    assert bound is spec
    assert bind_method_profile(_PlainSpec(), profile="standardized", params={}) == _PlainSpec()
    assert bind_method_profile(None, profile="standardized", params={}) is None


def test_profile_binding_rejects_ambiguous_params_profile() -> None:
    with pytest.raises(BenchRuntimeError, match="method.params.profile is ambiguous") as error:
        bind_method_profile(
            _PlainSpec(),
            profile="paper:canonical",
            params={"profile": "legacy-alias"},
        )
    assert error.value.code == "E_BENCH_METHOD_PROFILE_DUPLICATE"


def test_profile_binding_rejects_ambiguous_profile_even_without_a_spec() -> None:
    with pytest.raises(BenchRuntimeError, match="method.params.profile is ambiguous") as error:
        bind_method_profile(None, profile="standardized", params={"profile": "paper:hidden"})
    assert error.value.code == "E_BENCH_METHOD_PROFILE_DUPLICATE"
