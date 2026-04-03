from __future__ import annotations

from collections.abc import Callable

from modssc.backends.torch_common import make_dtype_from_spec, make_resolve_device, make_to_tensor

from ..errors import OptionalDependencyError
from ..optional import optional_import


def _torch():
    return optional_import("torch", extra="transductive-torch")


def _torch_getter():
    return _torch()


resolve_device = make_resolve_device(
    torch_getter=_torch_getter,
    optional_dependency_error_cls=OptionalDependencyError,
    extra="transductive-torch",
)
dtype_from_spec = make_dtype_from_spec(torch_getter=_torch_getter)
to_tensor = make_to_tensor(torch_getter=_torch_getter)


def spmm(
    *,
    n_nodes: int,
    edge_index,
    edge_weight,
    X,
    device,
    dtype,
):
    """Sparse adjacency times dense features using torch sparse COO."""
    torch = _torch()
    edge_index_t = to_tensor(edge_index, device=device, dtype=torch.long)
    E = int(edge_index_t.shape[1])
    if E == 0:
        out_shape = (n_nodes,) + tuple(X.shape[1:])
        return torch.zeros(out_shape, device=device, dtype=dtype)

    if edge_weight is None:
        w = torch.ones((E,), device=device, dtype=dtype)
    else:
        w = to_tensor(edge_weight, device=device, dtype=dtype).reshape(-1)
        if int(w.numel()) != E:
            raise ValueError("edge_weight must have shape (E,)")

    src = edge_index_t[0]
    dst = edge_index_t[1]

    # COO matrix with shape (n, n): A[dst, src] = w
    A = torch.sparse_coo_tensor(
        torch.stack([dst, src], dim=0),
        w,
        size=(n_nodes, n_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()

    out = torch.sparse.mm(A, X.view(-1, 1)).view(-1) if X.ndim == 1 else torch.sparse.mm(A, X)
    return out


def normalize_edges(
    *,
    n_nodes: int,
    edge_index,
    edge_weight,
    mode: str,
    device,
    dtype,
    eps: float = 1e-12,
):
    torch = _torch()
    edge_index_t = to_tensor(edge_index, device=device, dtype=torch.long)
    E = int(edge_index_t.shape[1])
    if edge_weight is None:
        w = torch.ones((E,), device=device, dtype=dtype)
    else:
        w = to_tensor(edge_weight, device=device, dtype=dtype).reshape(-1)
        if int(w.numel()) != E:
            raise ValueError("edge_weight must have shape (E,)")

    if E == 0:
        return w

    src = edge_index_t[0]
    dst = edge_index_t[1]

    deg = torch.zeros((n_nodes,), device=device, dtype=dtype)
    deg.scatter_add_(0, dst, w)

    if mode == "none":
        out = w
    elif mode == "rw":
        out = w / torch.clamp(deg[dst], min=eps)
    elif mode == "sym":
        out = w / torch.sqrt(torch.clamp(deg[dst], min=eps) * torch.clamp(deg[src], min=eps))
    else:
        raise ValueError(f"Unknown normalization mode: {mode!r}")

    return out


def cg_solve(
    *,
    matvec: Callable,
    b,
    x0=None,
    tol: float = 1e-6,
    max_iter: int = 1000,
):
    """Conjugate gradient in torch."""
    torch = _torch()
    x = torch.zeros_like(b) if x0 is None else x0.clone()

    r = b - matvec(x)
    p = r.clone()
    rs_old = torch.dot(r.view(-1), r.view(-1))

    info = {"n_iter": 0, "converged": False, "residual_norm": float(torch.sqrt(rs_old).item())}
    if rs_old.item() == 0.0:
        info["converged"] = True
        info["residual_norm"] = 0.0
        return x, info

    for k in range(int(max_iter)):
        Ap = matvec(p)
        denom = torch.dot(p.view(-1), Ap.view(-1))
        if denom.item() == 0.0:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r.view(-1), r.view(-1))
        info["n_iter"] = k + 1
        info["residual_norm"] = float(torch.sqrt(rs_new).item())
        if info["residual_norm"] <= tol:
            info["converged"] = True
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, info
