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
    if not isinstance(x, torch.Tensor):
        raise InductiveValidationError(f"{name} must be a torch.Tensor.")
    if x.dtype not in (torch.float32, torch.float64):
        raise InductiveValidationError(f"{name} must be float32 or float64.")


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


def cycle_batches(
    X: Any,
    y: Any | None,
    *,
    batch_size: int,
    generator: Any,
    steps: int,
) -> Iterator[tuple[Any, Any | None]]:
    torch = _torch()
    n = int(X.shape[0])
    if n <= 0:
        raise InductiveValidationError("Batching requires non-empty tensors.")
    if not isinstance(X, torch.Tensor):
        raise InductiveValidationError("Batching expects torch.Tensor inputs.")
    it = cycle_batch_indices(
        n, batch_size=batch_size, generator=generator, device=X.device, steps=steps
    )
    for idx in it:
        if y is None:
            yield X[idx], None
        else:
            yield X[idx], y[idx]
