from __future__ import annotations

import copy

import pytest

from modssc.runtime.software import (
    SoftwareManifest,
    SoftwareProvenanceError,
    attach_software_manifest,
    collect_software_manifest,
    resolve_required_distributions,
    software_identity_payload,
    software_sha256,
)


def _runtime_versions() -> dict[str, object]:
    return {
        "python": "3.11.13",
        "python_implementation": "CPython",
        "platform": "macOS-arm64",
        "executable": "/Users/researcher/.venv/bin/python",
        "modssc": "0.1.0",
        "distribution_sha256": "a" * 64,
        "git_sha": "b" * 40,
        "git_dirty": False,
        "git_diff_sha256": "c" * 64,
        "cuda": None,
        "cudnn": None,
    }


def _with_versions(*required: str) -> dict[str, object]:
    installed = {
        "numpy": "2.3.1",
        "torchvision": "0.23.0",
        "faiss-cpu": "1.11.0",
    }
    return attach_software_manifest(
        _runtime_versions(),
        required_distributions=required,
        version_getter=installed.get,
        require_complete=True,
    )


def test_selected_extra_expansion_is_stable_and_does_not_include_other_extras() -> None:
    groups = {
        "vision": ["torch>=2", "torchvision>=0.20", "Pillow; python_version >= '3.11'"],
        "graph-faiss": ["faiss-cpu; platform_system != 'Darwin'"],
        "hf": ["datasets"],
    }

    selected = resolve_required_distributions(
        extras=["vision"],
        optional_dependencies=groups,
        explicit=["custom_runtime"],
    )

    assert selected == (
        "custom-runtime",
        "numpy",
        "pillow",
        "scipy",
        "torch",
        "torchvision",
    )
    assert "datasets" not in selected
    assert "faiss-cpu" not in selected


def test_torchvision_and_faiss_versions_invalidate_identity_when_declared() -> None:
    reference = _with_versions("numpy", "torchvision", "faiss-cpu")

    changed_torchvision = copy.deepcopy(reference)
    changed_torchvision["software_manifest"]["versions"]["torchvision"] = "0.24.0"  # type: ignore[index]
    changed_faiss = copy.deepcopy(reference)
    changed_faiss["software_manifest"]["versions"]["faiss-cpu"] = "1.12.0"  # type: ignore[index]

    assert software_sha256(changed_torchvision) != software_sha256(reference)
    assert software_sha256(changed_faiss) != software_sha256(reference)


def test_unrequired_torchvision_and_faiss_do_not_block_portable_resume() -> None:
    reference = _with_versions("numpy")
    another_host = copy.deepcopy(reference)
    another_host.update(
        {
            "platform": "Linux-x86_64",
            "executable": "/lustre/venv/bin/python",
            "cuda": "12.6",
            "torchvision": "99.0",
            "faiss-cpu": "99.0",
        }
    )

    assert software_sha256(another_host) == software_sha256(reference)
    payload = software_identity_payload(reference)
    assert payload["software_manifest"]["required_distributions"] == ["numpy"]


def test_manifest_round_trip_and_strict_missing_version_contract() -> None:
    manifest = collect_software_manifest(
        ["NumPy", "torch_vision"],
        version_getter={"numpy": "2.3.1", "torch-vision": None}.get,
    )
    assert SoftwareManifest.from_dict(manifest.to_dict()) == manifest
    assert manifest.missing_versions == ("torch-vision",)

    with pytest.raises(SoftwareProvenanceError, match="torch-vision"):
        manifest.require_complete()
    with pytest.raises(SoftwareProvenanceError, match="torch-vision"):
        collect_software_manifest(
            ["numpy", "torch-vision"],
            version_getter={"numpy": "2.3.1", "torch-vision": None}.get,
            require_complete=True,
        )


@pytest.mark.parametrize(
    "value",
    [
        {"schema_version": 99, "required_distributions": [], "versions": {}},
        {"schema_version": 1, "required_distributions": ["numpy"], "versions": {}},
        {
            "schema_version": 1,
            "required_distributions": ["numpy", "NumPy"],
            "versions": {"numpy": "1"},
        },
    ],
)
def test_manifest_rejects_ambiguous_or_incomplete_payloads(value: dict[str, object]) -> None:
    with pytest.raises(SoftwareProvenanceError):
        SoftwareManifest.from_dict(value)
