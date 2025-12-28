from __future__ import annotations

import numpy as np

from ...errors import GraphValidationError
from ...specs import GraphWeightsSpec, Metric


def compute_edge_weights(
    *,
    distances: np.ndarray,
    metric: Metric,
    weights: GraphWeightsSpec,
) -> np.ndarray:
    """Compute edge weights from a distance array.

    Parameters
    ----------
    distances:
        1D array of distances for each edge.
        For cosine metric, this must be cosine distance in [0, 2].
    metric:
        Distance metric used by the builder.
    weights:
        Weight specification.

    Returns
    -------
    np.ndarray
        float32 weights array.
    """
    d = np.asarray(distances, dtype=np.float32)
    if d.ndim != 1:
        raise GraphValidationError("distances must be 1D")

    if weights.kind == "binary":
        return np.ones_like(d, dtype=np.float32)

    if weights.kind == "heat":
        sigma = float(weights.sigma or 0.0)
        if sigma <= 0:
            raise GraphValidationError("sigma must be > 0 for heat weights")
        return np.exp(-(d * d) / (2.0 * sigma * sigma)).astype(np.float32)

    if weights.kind == "cosine":
        if metric != "cosine":
            raise GraphValidationError("cosine weights require metric='cosine'")
        # cosine distance -> similarity in [-1, 1] roughly, but for normalized vectors it is [0,2]
        return (1.0 - d).astype(np.float32)

    raise GraphValidationError(f"Unknown weight kind: {weights.kind!r}")
