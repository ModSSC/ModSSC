from __future__ import annotations

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.optional import optional_import

from .types import TorchModelBundle


def validate_torch_model_bundle(bundle: TorchModelBundle) -> TorchModelBundle:
    """Validate a torch model bundle (model + optimizer)."""
    torch = optional_import("torch", extra="inductive-torch")
    if not isinstance(bundle, TorchModelBundle):
        raise InductiveValidationError("model_bundle must be a TorchModelBundle.")
    if not isinstance(bundle.model, torch.nn.Module):
        raise InductiveValidationError("model_bundle.model must be a torch.nn.Module.")
    if not isinstance(bundle.optimizer, torch.optim.Optimizer):
        raise InductiveValidationError("model_bundle.optimizer must be a torch.optim.Optimizer.")

    params = [p for p in bundle.model.parameters() if p.requires_grad]
    if not params:
        raise InductiveValidationError("model_bundle.model must have trainable parameters.")

    model_ids = {id(p) for p in bundle.model.parameters()}
    for group in bundle.optimizer.param_groups:
        for p in group.get("params", []):
            if id(p) not in model_ids:
                raise InductiveValidationError(
                    "model_bundle.optimizer params must come from model parameters."
                )

    if bundle.ema_model is not None and not isinstance(bundle.ema_model, torch.nn.Module):
        raise InductiveValidationError("model_bundle.ema_model must be a torch.nn.Module.")

    return bundle
