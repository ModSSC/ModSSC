from __future__ import annotations

from typing import Any

from ..errors import BenchRuntimeError


def bind_method_profile(
    spec: Any,
    *,
    profile: str,
    params: dict[str, Any],
) -> Any:
    """Keep a campaign profile at the bench boundary.

    ``method.profile`` identifies a reproduction card; it is deliberately not
    copied into a public ModSSC method specification.  Executable behaviour is
    described exclusively by generic fields in ``method.params``.  This keeps
    article identities in :mod:`bench` while allowing the library algorithms
    to remain useful independently of the reproduction catalogue.

    The function name is retained for compatibility with the two orchestrators
    while callers migrate to the clearer boundary semantics.
    """

    if "profile" in params:
        raise BenchRuntimeError(
            "E_BENCH_METHOD_PROFILE_DUPLICATE",
            "declare the algorithm profile only as method.profile; "
            "method.params.profile is ambiguous",
        )
    del profile
    return spec


__all__ = ["bind_method_profile"]
