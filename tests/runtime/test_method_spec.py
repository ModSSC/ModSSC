from __future__ import annotations

from dataclasses import dataclass

import pytest

from modssc.runtime.method_spec import (
    MethodSpecError,
    build_method_spec,
    method_spec_has_field,
)


@dataclass(frozen=True)
class _Spec:
    value: int = 1


class _Method:
    def __init__(self) -> None:
        self.spec = _Spec()


def test_build_method_spec_is_regime_independent() -> None:
    assert build_method_spec(_Method, {"value": 2}) == _Spec(value=2)
    assert build_method_spec(_Method, require_spec=True) == _Spec()
    assert build_method_spec(_Method) is None


def test_build_method_spec_reports_invalid_params() -> None:
    with pytest.raises(MethodSpecError, match="invalid method params"):
        build_method_spec(_Method, {"unknown": 2})


def test_build_method_spec_reports_strict_introspection_failure() -> None:
    class BrokenMethod:
        def __init__(self) -> None:
            raise RuntimeError("broken")

    with pytest.raises(MethodSpecError, match="failed to instantiate method") as exc_info:
        build_method_spec(BrokenMethod, {"value": 2}, strict=True)

    assert exc_info.value.kind == "method_introspection"


def test_method_spec_field_introspection_is_native() -> None:
    assert method_spec_has_field(_Method, "value", strict=True)
    assert not method_spec_has_field(_Method, "backend", strict=True)


def test_method_spec_field_introspection_preserves_strict_error_taxonomy() -> None:
    class BrokenMethod:
        def __init__(self) -> None:
            raise RuntimeError("broken")

    assert not method_spec_has_field(BrokenMethod, "backend")
    with pytest.raises(MethodSpecError) as exc_info:
        method_spec_has_field(BrokenMethod, "backend", strict=True)
    assert exc_info.value.kind == "method_introspection"
