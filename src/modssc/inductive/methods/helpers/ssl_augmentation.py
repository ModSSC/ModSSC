from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from modssc.inductive.errors import InductiveValidationError

from .torch_support import slice_data

ONLINE_AUGMENTATION_META_KEY = "online_augmentation"


def _meta(data: Any) -> Mapping[str, Any]:
    meta = getattr(data, "meta", None)
    return meta if isinstance(meta, Mapping) else {}


def ssl_batch_views(
    data: Any,
    *,
    X_l: Any,
    X_u_w: Any,
    X_u_s: Any,
    idx_l: Any,
    idx_u: Any,
    optimization_step: int,
) -> tuple[Any, Any, Any]:
    """Materialize one labeled weak view and one unlabeled weak/strong pair.

    Configurations without an online augmenter retain the historical fixed-view
    behavior, which keeps all standardized configurations backward compatible.
    """

    meta = _meta(data)
    augmenter = meta.get(ONLINE_AUGMENTATION_META_KEY)
    if augmenter is None:
        return (
            slice_data(X_l, idx_l),
            slice_data(X_u_w, idx_u),
            slice_data(X_u_s, idx_u),
        )

    X_u = getattr(data, "X_u", None)
    if X_u is None:
        raise InductiveValidationError("Online augmentation requires X_u.")
    idx_l_all = meta.get("source_idx_l", meta.get("idx_l"))
    idx_u_all = meta.get("source_idx_u", meta.get("idx_u"))
    if idx_l_all is None or idx_u_all is None:
        raise InductiveValidationError("Online augmentation requires meta.idx_l and meta.idx_u.")

    try:
        x_lb = augmenter.weak_batch(
            X_l,
            indices=idx_l,
            sample_ids=idx_l_all[idx_l],
            step=int(optimization_step),
        )
        x_uw, x_us = augmenter.pair_batch(
            X_u,
            indices=idx_u,
            sample_ids=idx_u_all[idx_u],
            step=int(optimization_step),
        )
    except (IndexError, TypeError, ValueError) as exc:
        raise InductiveValidationError(f"Online augmentation failed: {exc}") from exc
    return x_lb, x_uw, x_us
