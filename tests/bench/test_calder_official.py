from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from bench.campaign.protocols.calder import official

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = REPO_ROOT / "bench" / "assets" / "calder2020" / "protocol_inputs"


def test_official_bundle_reproduces_archived_table1_statistics() -> None:
    evidence = official.verify_calder_official_assets(OFFICIAL_ROOT)

    assert evidence["repository"] == official.OFFICIAL_REPOSITORY
    assert evidence["commit"] == official.OFFICIAL_COMMIT
    assert evidence["knn_sha256"] == official.OFFICIAL_KNN_SHA256
    assert evidence["permutations_sha256"] == official.OFFICIAL_PERMUTATIONS_SHA256
    assert evidence["permutations_artifact_sha256"] == official.PERMUTATIONS_ARTIFACT_SHA256
    assert evidence["labels_content_sha256"] == official.MNIST_LABELS_CONTENT_SHA256
    assert evidence["paper_k_includes_self"] is True
    for method_id, targets in official.TABLE1_TARGETS.items():
        for budget, (target_mean, target_std) in targets.items():
            stats = evidence["results"][method_id][str(budget)]
            assert round(stats["mean"], 1) == target_mean
            assert round(stats["std"], 1) == target_std


def test_official_bundle_rejects_a_different_mnist_label_order() -> None:
    labels = np.arange(70_000, dtype=np.int64) % 10
    with pytest.raises(
        official.CalderOfficialArtifactError,
        match="ordering differs",
    ):
        official.verify_calder_official_assets(
            OFFICIAL_ROOT,
            dataset_labels=labels,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest.update(schema_version=1), "provenance differs"),
        (
            lambda manifest: manifest["provenance"].update(commit="other"),
            "provenance differs",
        ),
        (
            lambda manifest: manifest["files"].pop("graph/mnist-vae-knn30.npz"),
            "file pins differ",
        ),
        (
            lambda manifest: manifest["table1"].update(trials=99),
            "Table 1 contract differs",
        ),
    ],
)
def test_official_manifest_is_not_self_certifying(monkeypatch, mutate, message) -> None:
    manifest = json.loads((OFFICIAL_ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    changed = copy.deepcopy(manifest)
    mutate(changed)
    monkeypatch.setattr(official, "_read_mapping", lambda _path: changed)

    with pytest.raises(official.CalderOfficialArtifactError, match=message):
        official._verify_manifest(OFFICIAL_ROOT)


def test_official_source_is_pinned_as_provenance_but_not_distributed() -> None:
    assert official.OFFICIAL_SOURCE_SHA256 == (
        "e2d16b74ac7d9ba3daab1c2d020e97b268e26bc378fba1f1077bbfd8707a3372"
    )
    assert not (REPO_ROOT / "bench/assets/calder2020/GraphLearningOld-04bece45").exists()
    assert not any(OFFICIAL_ROOT.rglob("*.py"))


def test_official_bundle_rejects_missing_roots_and_symlink_roots(tmp_path) -> None:
    with pytest.raises(official.CalderOfficialArtifactError, match="root is missing"):
        official.verify_calder_official_assets(tmp_path / "missing")

    link = tmp_path / "official-link"
    link.symlink_to(OFFICIAL_ROOT, target_is_directory=True)
    with pytest.raises(official.CalderOfficialArtifactError, match="must not be a symlink"):
        official.verify_calder_official_assets(link)
