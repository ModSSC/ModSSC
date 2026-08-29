from __future__ import annotations

from types import SimpleNamespace

from bench.main import _preprocess_logged_artifacts
from modssc.preprocess.store import ArtifactStore


def test_preprocess_logging_collects_all_step_info_without_known_step_ids() -> None:
    train_info = {"fingerprint": "train"}
    test_info = {"fingerprint": "test"}
    pre = SimpleNamespace(
        train_artifacts=ArtifactStore(
            {
                "features.future_encoder.info": train_info,
                "features.future_encoder": object(),
                "labels.classes": [0, 1],
            }
        ),
        test_artifacts=ArtifactStore(
            {
                "tokens.future_tokenizer.info": test_info,
                "tokens.input_ids": object(),
            }
        ),
    )

    payload = _preprocess_logged_artifacts(pre)

    assert payload["features.future_encoder.info"] is train_info
    assert payload["by_split"]["train"] == {"features.future_encoder.info": train_info}
    assert payload["by_split"]["test"] == {"tokens.future_tokenizer.info": test_info}
    assert "features.future_encoder" not in payload
