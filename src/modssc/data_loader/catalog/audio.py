from __future__ import annotations

from modssc.data_loader.types import DatasetSpec

AUDIO_CATALOG: dict[str, DatasetSpec] = {
    "speechcommands": DatasetSpec(
        key="speechcommands",
        provider="torchaudio",
        uri="torchaudio:SPEECHCOMMANDS",
        modality="audio",
        task="classification",
        description="SpeechCommands (torchaudio) with official training/testing subsets.",
        required_extra="audio",
        source_kwargs={},
    ),
    "yesno": DatasetSpec(
        key="yesno",
        provider="torchaudio",
        uri="torchaudio:YESNO",
        modality="audio",
        task="classification",
        description="YESNO (torchaudio). No official split (data_loader returns train only).",
        required_extra="audio",
        source_kwargs={},
    ),
}
