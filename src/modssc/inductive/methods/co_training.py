from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.utils import (
    BaseClassifierSpec,
    build_classifier,
    detect_backend,
    ensure_1d_labels,
    ensure_1d_labels_torch,
    ensure_classifier_backend,
    ensure_cpu_device,
    flatten_if_numpy,
    predict_scores,
    select_top_per_class,
    select_top_per_class_torch,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoTrainingSpec(BaseClassifierSpec):
    view_keys: tuple[str, str] | None = None
    max_iter: int = 20
    k_per_class: int = 1
    confidence_threshold: float | None = None


def _view_payload_numpy(value: Any, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(value, Mapping):
        if "X_l" not in value or "X_u" not in value:
            raise InductiveValidationError(f"views[{name!r}] must contain keys 'X_l' and 'X_u'.")
        X_l = value["X_l"]
        X_u = value["X_u"]
    elif isinstance(value, tuple) and len(value) == 2:
        X_l, X_u = value
    else:
        raise InductiveValidationError(
            f"views[{name!r}] must be a mapping with X_l/X_u or a tuple (X_l, X_u)."
        )

    if not isinstance(X_l, np.ndarray) or not isinstance(X_u, np.ndarray):
        raise InductiveValidationError(
            f"views[{name!r}] X_l/X_u must be numpy arrays. Use preprocess core.to_numpy."
        )
    if X_l.ndim < 2 or X_u.ndim < 2:
        raise InductiveValidationError(f"views[{name!r}] X_l/X_u must be at least 2D arrays.")
    return X_l, X_u


def _view_predict_payload_numpy(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, Mapping):
        if "X" in value:
            X = value["X"]
        elif "X_u" in value:
            X = value["X_u"]
        elif "X_l" in value:
            X = value["X_l"]
        else:
            raise InductiveValidationError(
                f"views[{name!r}] must contain key 'X', 'X_u', or 'X_l' for prediction."
            )
    else:
        X = value
    if not isinstance(X, np.ndarray):
        raise InductiveValidationError(f"views[{name!r}] must be a numpy array for prediction.")
    if X.ndim < 2:
        raise InductiveValidationError(f"views[{name!r}] must be at least 2D for prediction.")
    return X


def _get_torch_tensor(obj: Any) -> Any:
    if isinstance(obj, dict) and "x" in obj:
        return obj["x"]
    return obj


def _is_valid_torch(obj: Any, torch: Any) -> bool:
    return isinstance(obj, torch.Tensor) or (isinstance(obj, dict) and "x" in obj)


def _get_torch_len(obj: Any) -> int:
    if isinstance(obj, dict) and "x" in obj:
        return int(obj["x"].shape[0])
    return int(obj.shape[0])


def _get_torch_device(obj: Any) -> Any:
    if isinstance(obj, dict) and "x" in obj:
        return obj["x"].device
    return obj.device


def _same_device(a: Any, b: Any) -> bool:
    return (a == b) or (
        getattr(a, "type", None) == getattr(b, "type", None)
        and (getattr(a, "index", None) is None or getattr(b, "index", None) is None)
    )


def _view_payload_torch(value: Any, *, name: str):
    torch = optional_import("torch", extra="inductive-torch")
    if isinstance(value, Mapping):
        if "X_l" not in value or "X_u" not in value:
            raise InductiveValidationError(f"views[{name!r}] must contain keys 'X_l' and 'X_u'.")
        X_l = value["X_l"]
        X_u = value["X_u"]
    elif isinstance(value, tuple) and len(value) == 2:
        X_l, X_u = value
    else:
        raise InductiveValidationError(
            f"views[{name!r}] must be a mapping with X_l/X_u or a tuple (X_l, X_u)."
        )

    if not _is_valid_torch(X_l, torch) or not _is_valid_torch(X_u, torch):
        raise InductiveValidationError(
            f"views[{name!r}] X_l/X_u must be torch tensors. Use preprocess core.to_torch."
        )

    tl = _get_torch_tensor(X_l)
    tu = _get_torch_tensor(X_u)

    if tl.ndim < 2 or tu.ndim < 2:
        raise InductiveValidationError(f"views[{name!r}] X_l/X_u must be at least 2D tensors.")
    if tl.device != tu.device:
        raise InductiveValidationError(f"views[{name!r}] X_l/X_u must share device.")
    return X_l, X_u


def _view_predict_payload_torch(value: Any, *, name: str):
    torch = optional_import("torch", extra="inductive-torch")
    if isinstance(value, Mapping):
        if "X" in value:
            X = value["X"]
        elif "X_u" in value:
            X = value["X_u"]
        elif "X_l" in value:
            X = value["X_l"]
        else:
            raise InductiveValidationError(
                f"views[{name!r}] must contain key 'X', 'X_u', or 'X_l' for prediction."
            )
    else:
        X = value
    if not _is_valid_torch(X, torch):
        raise InductiveValidationError(f"views[{name!r}] must be a torch tensor for prediction.")

    Xt = _get_torch_tensor(X)
    if Xt.ndim < 2:
        raise InductiveValidationError(f"views[{name!r}] must be at least 2D for prediction.")
    return X


def _index_torch(obj: Any, idx: Any) -> Any:
    if isinstance(obj, dict) and "x" in obj:
        torch = optional_import("torch", extra="inductive-torch")
        res = obj.copy()
        res["x"] = obj["x"][idx]

        if "edge_index" in obj:
            try:
                from torch_geometric.utils import subgraph
            except ImportError as exc:
                raise InductiveValidationError(
                    "PyG is required to slice edge_index for graph inputs. "
                    "Install torch_geometric or remove graph preprocessing."
                ) from exc
            else:
                edge_index = obj["edge_index"]
                num_nodes = obj["x"].shape[0]

                subset = idx
                if isinstance(subset, slice):
                    subset = torch.arange(num_nodes, device=obj["x"].device)[subset]

                new_ei, _ = subgraph(subset, edge_index, relabel_nodes=True, num_nodes=num_nodes)
                res["edge_index"] = new_ei

        return res
    return obj[idx]


def _cat_torch(objs: list[Any], dim: int = 0) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    if isinstance(objs[0], dict) and "x" in objs[0]:
        res = objs[0].copy()
        res["x"] = torch.cat([o["x"] for o in objs], dim=dim)

        if "edge_index" in objs[0]:
            edge_indices = []
            offset = 0
            for o in objs:
                ei = o["edge_index"]
                edge_indices.append(ei + offset)
                offset += o["x"].shape[0]
            res["edge_index"] = torch.cat(edge_indices, dim=1)

        return res
    return torch.cat(objs, dim=dim)


class CoTrainingMethod(InductiveMethod):
    """Co-training with two views (CPU/GPU)."""

    info = MethodInfo(
        method_id="co_training",
        name="Co-Training",
        year=1998,
        family="classic",
        supports_gpu=True,
        paper_title="Combining labeled and unlabeled data with co-training",
        paper_pdf="https://www.cs.cmu.edu/~avrim/Papers/co-training.pdf",
        official_code="",
    )

    def __init__(self, spec: CoTrainingSpec | None = None) -> None:
        self.spec = spec or CoTrainingSpec()
        self._clf1: Any | None = None
        self._clf2: Any | None = None
        self._view_keys: tuple[str, str] | None = None
        self._backend: str | None = None

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> CoTrainingMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        if data is None:
            raise InductiveValidationError("data must not be None.")
        if data.views is None:
            raise InductiveValidationError("CoTraining requires data.views with two views.")

        backend = detect_backend(data.X_l)
        ensure_classifier_backend(self.spec, backend=backend)
        logger.debug("backend=%s", backend)
        if backend == "numpy":
            ensure_cpu_device(device)
            if not isinstance(data.X_l, np.ndarray):
                raise InductiveValidationError(
                    "X_l must be a numpy array. Use preprocess core.to_numpy."
                )
            if not isinstance(data.y_l, np.ndarray):
                raise InductiveValidationError(
                    "y_l must be a numpy array. Use preprocess labels.to_numpy."
                )
            y_l = ensure_1d_labels(data.y_l, name="y_l")
        else:
            torch = optional_import("torch", extra="inductive-torch")
            if not _is_valid_torch(data.X_l, torch):
                raise InductiveValidationError(
                    "X_l must be a torch tensor. Use preprocess core.to_torch."
                )
            if not isinstance(data.y_l, torch.Tensor):
                raise InductiveValidationError(
                    "y_l must be a torch tensor. Use preprocess labels.to_torch."
                )
            y_l = ensure_1d_labels_torch(data.y_l, name="y_l")

        keys = self.spec.view_keys
        if keys is None:
            keys = tuple(sorted(data.views.keys()))[:2]
        if len(keys) != 2:
            raise InductiveValidationError("CoTraining requires exactly two view keys.")

        if backend == "numpy":
            v1_l, v1_u = _view_payload_numpy(data.views[keys[0]], name=keys[0])
            v2_l, v2_u = _view_payload_numpy(data.views[keys[1]], name=keys[1])
            # Ensure flattening for standard classifiers
            v1_l = flatten_if_numpy(v1_l)
            v1_u = flatten_if_numpy(v1_u)
            v2_l = flatten_if_numpy(v2_l)
            v2_u = flatten_if_numpy(v2_u)
        else:
            v1_l, v1_u = _view_payload_torch(data.views[keys[0]], name=keys[0])
            v2_l, v2_u = _view_payload_torch(data.views[keys[1]], name=keys[1])

        if backend == "torch":
            d1 = _get_torch_device(v1_l)
            d2 = _get_torch_device(v2_l)
            if d1 != d2:
                raise InductiveValidationError("views must be on the same device.")
            if not _same_device(y_l.device, d1):
                try:
                    y_l = y_l.to(d1)
                except Exception as exc:
                    raise InductiveValidationError(
                        "y_l must be on the same device as the view tensors."
                    ) from exc
            if not _same_device(y_l.device, d1) or not _same_device(y_l.device, d2):
                raise InductiveValidationError(
                    "y_l must be on the same device as the view tensors."
                )

        l1 = v1_l.shape[0] if backend == "numpy" else _get_torch_len(v1_l)
        l2 = v2_l.shape[0] if backend == "numpy" else _get_torch_len(v2_l)
        u1 = v1_u.shape[0] if backend == "numpy" else _get_torch_len(v1_u)
        u2 = v2_u.shape[0] if backend == "numpy" else _get_torch_len(v2_u)

        if l1 != y_l.shape[0] or l2 != y_l.shape[0]:
            raise InductiveValidationError("View X_l must align with y_l length.")
        if u1 != u2:
            raise InductiveValidationError("View X_u must have the same number of rows.")
        logger.info(
            "Co-training sizes: n_labeled=%s n_unlabeled=%s",
            int(l1),
            int(u1),
        )

        clf1 = build_classifier(self.spec, seed=seed)
        clf2 = build_classifier(self.spec, seed=seed)

        X1_l = v1_l
        X2_l = v2_l
        y1_l = y_l
        y2_l = y_l

        X1_u = v1_u
        X2_u = v2_u

        iter_count = 0
        while iter_count < int(self.spec.max_iter):
            clf1.fit(X1_l, y1_l)
            clf2.fit(X2_l, y2_l)

            u1_len = X1_u.shape[0] if backend == "numpy" else _get_torch_len(X1_u)
            if u1_len == 0:
                break

            scores1 = predict_scores(clf1, X1_u, backend=backend)
            scores2 = predict_scores(clf2, X2_u, backend=backend)

            if backend == "numpy":
                idx1 = select_top_per_class(
                    scores1,
                    k_per_class=int(self.spec.k_per_class),
                    threshold=self.spec.confidence_threshold,
                )
                idx2 = select_top_per_class(
                    scores2,
                    k_per_class=int(self.spec.k_per_class),
                    threshold=self.spec.confidence_threshold,
                )
            else:
                idx1 = select_top_per_class_torch(
                    scores1,
                    k_per_class=int(self.spec.k_per_class),
                    threshold=self.spec.confidence_threshold,
                )
                idx2 = select_top_per_class_torch(
                    scores2,
                    k_per_class=int(self.spec.k_per_class),
                    threshold=self.spec.confidence_threshold,
                )

            sel1 = int(idx1.numel()) if backend == "torch" else int(idx1.size)
            sel2 = int(idx2.numel()) if backend == "torch" else int(idx2.size)
            if sel1 == 0 and sel2 == 0:
                logger.debug("Co-training iter=%s no new labels; stopping.", iter_count)
                break

            if backend == "numpy":
                if idx1.size:
                    y_from_1 = np.asarray(clf1.predict(X1_u[idx1]))
                    X2_l = np.concatenate([X2_l, X2_u[idx1]], axis=0)
                    y2_l = np.concatenate([y2_l, y_from_1], axis=0)
                if idx2.size:
                    y_from_2 = np.asarray(clf2.predict(X2_u[idx2]))
                    X1_l = np.concatenate([X1_l, X1_u[idx2]], axis=0)
                    y1_l = np.concatenate([y1_l, y_from_2], axis=0)

                mask = np.ones((X1_u.shape[0],), dtype=bool)
                if idx1.size:
                    mask[idx1] = False
                if idx2.size:
                    mask[idx2] = False
                X1_u = X1_u[mask]
                X2_u = X2_u[mask]
            else:
                if idx1.numel():
                    y_from_1 = clf1.predict(_index_torch(X1_u, idx1))
                    X2_l = _cat_torch([X2_l, _index_torch(X2_u, idx1)], dim=0)
                    y2_l = torch.cat([y2_l, y_from_1], dim=0)
                if idx2.numel():
                    y_from_2 = clf2.predict(_index_torch(X2_u, idx2))
                    X1_l = _cat_torch([X1_l, _index_torch(X1_u, idx2)], dim=0)
                    y1_l = torch.cat([y1_l, y_from_2], dim=0)

                d_u = _get_torch_device(X1_u)
                l_u = _get_torch_len(X1_u)
                mask = torch.ones((l_u,), dtype=torch.bool, device=d_u)
                if idx1.numel():
                    mask[idx1] = False
                if idx2.numel():
                    mask[idx2] = False
                X1_u = _index_torch(X1_u, mask)
                X2_u = _index_torch(X2_u, mask)

            logger.debug(
                "Co-training iter=%s selected_view1=%s selected_view2=%s remaining=%s",
                iter_count,
                sel1,
                sel2,
                _get_torch_len(X1_u),
            )
            iter_count += 1

        clf1.fit(X1_l, y1_l)
        clf2.fit(X2_l, y2_l)

        self._clf1 = clf1
        self._clf2 = clf2
        self._view_keys = keys
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def _predict_scores_pair(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self._clf1 is None or self._clf2 is None:
            raise RuntimeError("CoTrainingMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X1)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict input backend mismatch.")
        s1 = predict_scores(self._clf1, X1, backend=backend)
        s2 = predict_scores(self._clf2, X2, backend=backend)
        if s1.shape[1] != s2.shape[1]:
            raise InductiveValidationError("CoTraining classifiers must agree on class count.")
        c1 = getattr(self._clf1, "classes_", None)
        c2 = getattr(self._clf2, "classes_", None)
        if c1 is not None and c2 is not None and not np.array_equal(c1, c2):
            raise InductiveValidationError("CoTraining classifiers disagree on class labels.")
        return (s1 + s2) / 2.0

    def predict_proba(self, data: Any) -> np.ndarray:
        if data is None or data.views is None:
            raise InductiveValidationError("CoTraining requires data.views at prediction time.")
        if self._view_keys is None:
            raise RuntimeError("CoTrainingMethod missing view keys; fit() was not called.")
        v1 = data.views.get(self._view_keys[0])
        v2 = data.views.get(self._view_keys[1])
        if v1 is None or v2 is None:
            raise InductiveValidationError("Missing required views for prediction.")
        backend = self._backend or detect_backend(data.X_l)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict_proba input backend mismatch.")
        if backend == "numpy":
            X1 = _view_predict_payload_numpy(v1, name=self._view_keys[0])
            X2 = _view_predict_payload_numpy(v2, name=self._view_keys[1])
            X1 = flatten_if_numpy(X1)
            X2 = flatten_if_numpy(X2)
        else:
            X1 = _view_predict_payload_torch(v1, name=self._view_keys[0])
            X2 = _view_predict_payload_torch(v2, name=self._view_keys[1])
        scores = self._predict_scores_pair(X1, X2)
        if backend == "numpy":
            row_sum = scores.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0.0] = 1.0
            return (scores / row_sum).astype(np.float32, copy=False)
        torch = optional_import("torch", extra="inductive-torch")
        row_sum = scores.sum(dim=1, keepdim=True)
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        return scores / row_sum

    def predict(self, data: Any) -> np.ndarray:
        if self._clf1 is None:
            raise RuntimeError("CoTrainingMethod is not fitted yet. Call fit() first.")
        scores = self.predict_proba(data)
        backend = self._backend or detect_backend(data.X_l)
        if backend == "numpy":
            idx = scores.argmax(axis=1)
            classes = getattr(self._clf1, "classes_", None)
            if classes is None:
                return idx
            return np.asarray(classes)[idx]
        idx = scores.argmax(dim=1)
        classes_t = getattr(self._clf1, "classes_t_", None)
        if classes_t is None:
            return idx
        return classes_t[idx]
