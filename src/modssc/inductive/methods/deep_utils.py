from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from modssc.inductive.deep import TorchModelBundle, validate_torch_model_bundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.optional import optional_import


def _torch():
    return optional_import("torch", extra="inductive-torch")


def ensure_model_bundle(bundle: TorchModelBundle) -> TorchModelBundle:
    return validate_torch_model_bundle(bundle)


def ensure_model_device(model: Any, *, device: Any) -> None:
    params = list(model.parameters())
    if not params:
        raise InductiveValidationError("model must have parameters.")
    dev = params[0].device
    for p in params:
        if p.device != dev:
            raise InductiveValidationError("model parameters must share the same device.")
    if dev != device:
        raise InductiveValidationError("model parameters must be on the same device as data.")


def extract_logits(output: Any):
    torch = _torch()
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping) and "logits" in output:
        logits = output["logits"]
        if not isinstance(logits, torch.Tensor):
            raise InductiveValidationError("model output 'logits' must be a torch.Tensor.")
        return logits
    if isinstance(output, tuple) and output:
        logits = output[0]
        if not isinstance(logits, torch.Tensor):
            raise InductiveValidationError("model output tuple[0] must be a torch.Tensor.")
        return logits
    raise InductiveValidationError(
        "model output must be a torch.Tensor or a mapping with key 'logits'."
    )


def extract_features(output: Any):
    torch = _torch()
    if isinstance(output, Mapping) and "feat" in output:
        feat = output["feat"]
        if not isinstance(feat, torch.Tensor):
            raise InductiveValidationError("model output 'feat' must be a torch.Tensor.")
        return feat
    raise InductiveValidationError(
        "model output must be a mapping with key 'feat' when mixup_manifold is enabled."
    )


def ensure_float_tensor(x: Any, *, name: str) -> None:
    torch = _torch()
    if isinstance(x, dict) and any(isinstance(v, torch.Tensor) for v in x.values()):
        # Optionally check dtype of 'x' key if present
        if (
            "x" in x
            and isinstance(x["x"], torch.Tensor)
            and x["x"].dtype
            not in (
                torch.float32,
                torch.float64,
            )
        ):
            raise InductiveValidationError(f"{name}['x'] must be float32 or float64.")
        return
    if not isinstance(x, torch.Tensor):
        raise InductiveValidationError(f"{name} must be a torch.Tensor.")
    if x.dtype not in (torch.float32, torch.float64):
        raise InductiveValidationError(f"{name} must be float32 or float64.")


def get_torch_len(x: Any) -> int:
    if isinstance(x, dict) and "x" in x:
        return int(x["x"].shape[0])
    return int(x.shape[0])


def get_torch_device(x: Any) -> Any:
    if isinstance(x, dict) and "x" in x:
        return x["x"].device
    return x.device


def get_torch_feature_dim(x: Any) -> int:
    if isinstance(x, dict) and "x" in x:
        return int(x["x"].shape[1])
    return int(x.shape[1])


def get_torch_ndim(x: Any) -> int:
    if isinstance(x, dict) and "x" in x:
        return int(x["x"].ndim)
    return int(x.ndim)


def concat_data(items: list[Any]) -> Any:
    """Concatenate tensors or graph dicts along the batch dimension.

    For dict inputs with 'x' and 'edge_index', this builds a disjoint union graph
    by offsetting edge indices for each block.
    """
    if not items:
        return items
    if not any(isinstance(x, dict) for x in items):
        torch = _torch()
        return torch.cat(items, dim=0)

    torch = _torch()
    graphs = []
    for x in items:
        if isinstance(x, dict):
            graphs.append(x)
        else:
            graphs.append({"x": x})

    xs = [torch.as_tensor(g["x"]) for g in graphs]
    x_cat = torch.cat(xs, dim=0)
    out: dict[str, Any] = {"x": x_cat}

    if all(isinstance(g, dict) and "edge_index" in g for g in graphs):
        edge_indices = []
        edge_weights = []
        offset = 0
        for g in graphs:
            ei = g["edge_index"]
            if not isinstance(ei, torch.Tensor):
                ei = torch.as_tensor(ei, device=x_cat.device, dtype=torch.long)
            ei = ei + offset
            edge_indices.append(ei)
            if "edge_weight" in g:
                ew = g["edge_weight"]
                if not isinstance(ew, torch.Tensor):
                    ew = torch.as_tensor(ew, device=x_cat.device)
                edge_weights.append(ew)
            offset += int(g["x"].shape[0])
        out["edge_index"] = torch.cat(edge_indices, dim=1)
        if edge_weights:
            out["edge_weight"] = torch.cat(edge_weights, dim=0)

    # Concatenate per-node tensors for shared keys.
    shared_keys = set.intersection(*(set(g.keys()) for g in graphs)) if graphs else set()
    for key in shared_keys:
        if key in {"x", "edge_index", "edge_weight"}:
            continue
        vals = [g[key] for g in graphs]
        if all(isinstance(v, torch.Tensor) for v in vals):
            # If node-wise (matching each graph's node count), concatenate
            if all(v.shape[0] == g["x"].shape[0] for v, g in zip(vals, graphs, strict=False)):
                out[key] = torch.cat(vals, dim=0)
            else:
                out[key] = vals[0]
        else:
            out[key] = vals[0]

    return out


@contextmanager
def freeze_batchnorm(model: Any, *, enabled: bool):
    if not enabled:
        yield
        return
    torch = _torch()
    bns = []
    states = []
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            bns.append(m)
            states.append(m.training)
            m.eval()
    try:
        yield
    finally:
        for m, state in zip(bns, states, strict=False):
            m.train(state)


def num_batches(n: int, batch_size: int) -> int:
    return max(1, int(math.ceil(float(n) / float(batch_size))))


def _iter_batch_indices(n: int, *, batch_size: int, generator: Any, device: Any) -> Iterator[Any]:
    torch = _torch()
    idx = torch.randperm(int(n), generator=generator, device="cpu")
    if device is not None and getattr(device, "type", "cpu") != "cpu":
        idx = idx.to(device)
    for start in range(0, int(n), int(batch_size)):
        yield idx[start : start + int(batch_size)]


def cycle_batch_indices(
    n: int, *, batch_size: int, generator: Any, device: Any, steps: int
) -> Iterator[Any]:
    it = _iter_batch_indices(n, batch_size=batch_size, generator=generator, device=device)
    for _ in range(int(steps)):
        try:
            idx = next(it)
        except StopIteration:
            it = _iter_batch_indices(n, batch_size=batch_size, generator=generator, device=device)
            idx = next(it)
        yield idx


def slice_data(X: Any, idx: Any) -> Any:
    torch = _torch()
    if isinstance(X, dict):
        n = int(X["x"].shape[0]) if "x" in X else 0
        batch_data = {}
        # Handle features 'x'
        if "x" in X:
            batch_data["x"] = X["x"][idx]

        # Handle graph 'edge_index'
        if "edge_index" in X:
            try:
                from torch_geometric.utils import subgraph

                subgraph_idx = idx
                if isinstance(idx, slice):
                    start, stop, step = idx.indices(n)
                    device = (
                        X["edge_index"].device
                        if isinstance(X["edge_index"], torch.Tensor)
                        else None
                    )
                    subgraph_idx = torch.arange(start, stop, step, device=device)

                # relabel_nodes=True ensures indices map to 0..len(idx)
                edge_index, _ = subgraph(
                    subgraph_idx, X["edge_index"], relabel_nodes=True, num_nodes=n
                )
                batch_data["edge_index"] = edge_index
            except ImportError:
                pass

        for k, v in X.items():
            if k in ("x", "edge_index"):
                continue
            if isinstance(v, torch.Tensor) and v.shape[0] == n:
                batch_data[k] = v[idx]
            else:
                batch_data[k] = v
        return batch_data

    return X[idx]


def cat_data(data_list: list[Any]) -> Any:
    torch = _torch()
    if not data_list:
        return None
    sample = data_list[0]

    if isinstance(sample, dict):
        # Check if it looks like graph data
        if "edge_index" in sample:
            try:
                from torch_geometric.data import Batch, Data

                # Convert dicts to Data objects for batching
                # We need to ensure we only pass fields that Data supports or handle arbitrarily
                # For simplicity, convert x and edge_index
                objs = []
                for d in data_list:
                    # kwargs construction
                    kwargs = {k: v for k, v in d.items() if k not in ("batch", "ptr")}
                    objs.append(Data(**kwargs))

                batch = Batch.from_data_list(objs)
                # Convert back to dict
                return batch.to_dict()  # or dict(batch) ? batch.to_dict() returns dict of tensors
            except ImportError:
                pass

        # Fallback for non-graph dicts: cat tensors key-wise
        out = {}
        for k in sample:
            if isinstance(sample[k], torch.Tensor):
                out[k] = torch.cat([d[k] for d in data_list], dim=0)
            else:
                out[k] = sample[k]  # duplicate/last wins?
        return out

    return torch.cat(data_list, dim=0)


def cycle_batches(
    X: Any,
    y: Any | None,
    *,
    batch_size: int,
    generator: Any,
    steps: int,
) -> Iterator[tuple[Any, Any | None]]:
    torch = _torch()

    if isinstance(X, dict) and "x" in X:
        n = int(X["x"].shape[0])
        device = X["x"].device
    else:
        n = int(X.shape[0])
        device = X.device

    if n <= 0:
        raise InductiveValidationError("Batching requires non-empty tensors.")

    if not isinstance(X, torch.Tensor) and not isinstance(X, dict):
        raise InductiveValidationError("Batching expects torch.Tensor inputs (or dict).")

    it = cycle_batch_indices(
        n, batch_size=batch_size, generator=generator, device=device, steps=steps
    )

    for idx in it:
        batch_x = slice_data(X, idx)
        if y is not None:
            yield batch_x, y[idx]
        else:
            yield batch_x, None
