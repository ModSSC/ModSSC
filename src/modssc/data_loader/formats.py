from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OutputFormat:
    modality: str
    X: str
    y: str
    edges: str | None = None
    masks: str | None = None


OUTPUT_FORMATS: dict[str, OutputFormat] = {
    "tabular": OutputFormat(
        "tabular", "np.ndarray (n_samples, n_features)", "np.ndarray (n_samples,)"
    ),
    "text": OutputFormat(
        "text", "np.ndarray dtype=object (n_samples,) of str", "np.ndarray (n_samples,)"
    ),
    "vision": OutputFormat("vision", "np.ndarray or object array", "np.ndarray (n_samples,)"),
    "audio": OutputFormat(
        "audio", "object array (waveforms or paths)", "np.ndarray or object array"
    ),
    "graph": OutputFormat(
        "graph",
        "np.ndarray (n_nodes, n_features)",
        "np.ndarray (n_nodes,)",
        edges="np.ndarray (2, n_edges)",
        masks="dict[str, np.ndarray bool] (train/val/test) when provided officially",
    ),
    "unknown": OutputFormat("unknown", "provider dependent", "provider dependent"),
}


def get_output_format(modality: str) -> OutputFormat:
    try:
        return OUTPUT_FORMATS[modality]
    except KeyError as e:
        raise KeyError(f"Unknown modality: {modality!r}. Known: {sorted(OUTPUT_FORMATS)}") from e
