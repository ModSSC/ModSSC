from __future__ import annotations

import modssc.runtime as runtime
from modssc.runtime import (
    build_resume_identity,
    effective_config_sha256,
    protocol_identity_payload,
    protocol_sha256,
)


def test_protocol_identity_is_exported_by_the_runtime_public_facade() -> None:
    expected = {
        "build_resume_identity": build_resume_identity,
        "effective_config_sha256": effective_config_sha256,
        "protocol_identity_payload": protocol_identity_payload,
        "protocol_sha256": protocol_sha256,
    }

    assert set(expected) <= set(runtime.__all__)
    for name, function in expected.items():
        assert getattr(runtime, name) is function
