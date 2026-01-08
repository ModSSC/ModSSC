from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.deep_utils import (
    cycle_batch_indices,
    cycle_batches,
    ensure_float_tensor,
    ensure_model_bundle,
    ensure_model_device,
    extract_logits,
    freeze_batchnorm,
    num_batches,
)
from modssc.inductive.methods.utils import (
    detect_backend,
    ensure_1d_labels_torch,
    ensure_torch_data,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec

logger = logging.getLogger(__name__)


def _l2_normalize(tensor: Any, *, eps: float = 1e-12) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    if not isinstance(tensor, torch.Tensor):
        raise InductiveValidationError("VAT perturbations require torch.Tensor inputs.")
    if int(tensor.shape[0]) == 0:
        return tensor
    flat = tensor.reshape(int(tensor.shape[0]), -1)
    norm = torch.norm(flat, dim=1, keepdim=True).clamp_min(float(eps))
    view_shape = (int(tensor.shape[0]),) + (1,) * (int(tensor.ndim) - 1)
    return tensor / norm.view(view_shape)


def _kl_divergence(probs: Any, log_probs: Any) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    return (probs * (torch.log(probs + 1e-12) - log_probs)).sum(dim=1).mean()


@dataclass(frozen=True)
class VATSpec:
    """Specification for VAT (torch-only)."""

    model_bundle: TorchModelBundle | None = None
    lambda_u: float = 1.0
    xi: float = 1e-6
    eps: float = 2.5
    num_iters: int = 1
    unsup_warm_up: float = 0.4
    batch_size: int = 64
    max_epochs: int = 1
    freeze_bn: bool = True
    detach_target: bool = True


class VATMethod(InductiveMethod):
    """Virtual adversarial training (torch-only)."""

    info = MethodInfo(
        method_id="vat",
        name="VAT",
        year=2017,
        family="consistency",
        supports_gpu=True,
        paper_title=(
            "Virtual Adversarial Training: A Regularization Method for Supervised and "
            "Semi-Supervised Learning"
        ),
        paper_pdf="https://arxiv.org/abs/1704.03976",
        official_code="",
    )

    def __init__(self, spec: VATSpec | None = None) -> None:
        self.spec = spec or VATSpec()
        self._bundle: TorchModelBundle | None = None
        self._backend: str | None = None

    def _vat_loss(
        self,
        model: Any,
        x_u: Any,
        *,
        xi: float,
        eps: float,
        num_iters: int,
        freeze_bn: bool,
        detach_target: bool,
    ) -> Any:
        torch = optional_import("torch", extra="inductive-torch")
        if int(num_iters) <= 0:
            raise InductiveValidationError("num_iters must be >= 1.")

        if detach_target:
            with torch.no_grad(), freeze_batchnorm(model, enabled=bool(freeze_bn)):
                logits_u = extract_logits(model(x_u))
                if int(logits_u.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                probs = torch.softmax(logits_u, dim=1)
        else:
            with freeze_batchnorm(model, enabled=bool(freeze_bn)):
                logits_u = extract_logits(model(x_u))
            if int(logits_u.ndim) != 2:
                raise InductiveValidationError("Model logits must be 2D (batch, classes).")
            probs = torch.softmax(logits_u, dim=1)

        d = torch.randn_like(x_u)
        for _ in range(int(num_iters)):
            d = _l2_normalize(d) * float(xi)
            d = d.detach()
            d.requires_grad_(True)
            with freeze_batchnorm(model, enabled=bool(freeze_bn)):
                logits_d = extract_logits(model(x_u + d))
            if int(logits_d.ndim) != 2:
                raise InductiveValidationError("Model logits must be 2D (batch, classes).")
            log_probs_d = torch.nn.functional.log_softmax(logits_d, dim=1)
            kl = _kl_divergence(probs, log_probs_d)
            grad = torch.autograd.grad(kl, d, retain_graph=False, create_graph=False)[0]
            d = grad.detach()

        r_adv = _l2_normalize(d) * float(eps)
        with freeze_batchnorm(model, enabled=bool(freeze_bn)):
            logits_adv = extract_logits(model(x_u + r_adv))
        if int(logits_adv.ndim) != 2:
            raise InductiveValidationError("Model logits must be 2D (batch, classes).")
        log_probs_adv = torch.nn.functional.log_softmax(logits_adv, dim=1)
        return _kl_divergence(probs, log_probs_adv)

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> VATMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "params lambda_u=%s xi=%s eps=%s num_iters=%s unsup_warm_up=%s batch_size=%s "
            "max_epochs=%s freeze_bn=%s detach_target=%s has_model_bundle=%s device=%s seed=%s",
            self.spec.lambda_u,
            self.spec.xi,
            self.spec.eps,
            self.spec.num_iters,
            self.spec.unsup_warm_up,
            self.spec.batch_size,
            self.spec.max_epochs,
            self.spec.freeze_bn,
            self.spec.detach_target,
            bool(self.spec.model_bundle),
            device,
            seed,
        )
        if data is None:
            raise InductiveValidationError("data must not be None.")

        backend = detect_backend(data.X_l)
        if backend != "torch":
            raise InductiveValidationError("VAT requires torch tensors (torch backend).")

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        X_u = ds.X_u if ds.X_u is not None else ds.X_u_w
        if X_u is None:
            raise InductiveValidationError("VAT requires X_u (unlabeled data).")

        X_l = ds.X_l
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        logger.info(
            "VAT sizes: n_labeled=%s n_unlabeled=%s",
            int(X_l.shape[0]),
            int(X_u.shape[0]),
        )

        if int(X_l.shape[0]) == 0:
            raise InductiveValidationError("X_l must be non-empty.")
        if int(X_u.shape[0]) == 0:
            raise InductiveValidationError("X_u must be non-empty.")

        ensure_float_tensor(X_l, name="X_l")
        ensure_float_tensor(X_u, name="X_u")

        if y_l.dtype != torch.int64:
            raise InductiveValidationError("y_l must be int64 for torch cross entropy.")

        if self.spec.model_bundle is None:
            raise InductiveValidationError("model_bundle must be provided for VAT.")

        bundle = ensure_model_bundle(self.spec.model_bundle)
        model = bundle.model
        optimizer = bundle.optimizer
        ensure_model_device(model, device=X_l.device)

        if int(self.spec.batch_size) <= 0:
            raise InductiveValidationError("batch_size must be >= 1.")
        if int(self.spec.max_epochs) <= 0:
            raise InductiveValidationError("max_epochs must be >= 1.")
        if float(self.spec.lambda_u) < 0:
            raise InductiveValidationError("lambda_u must be >= 0.")
        if float(self.spec.xi) <= 0:
            raise InductiveValidationError("xi must be > 0.")
        if float(self.spec.eps) <= 0:
            raise InductiveValidationError("eps must be > 0.")
        if int(self.spec.num_iters) <= 0:
            raise InductiveValidationError("num_iters must be >= 1.")
        if float(self.spec.unsup_warm_up) < 0:
            raise InductiveValidationError("unsup_warm_up must be >= 0.")

        steps_l = num_batches(int(X_l.shape[0]), int(self.spec.batch_size))
        steps_u = num_batches(int(X_u.shape[0]), int(self.spec.batch_size))
        steps_per_epoch = max(int(steps_l), int(steps_u))
        total_steps = int(self.spec.max_epochs) * steps_per_epoch
        if float(self.spec.unsup_warm_up) <= 0:
            warmup_steps = 0
        else:
            warmup_steps = int(max(1, round(float(self.spec.unsup_warm_up) * total_steps)))

        gen_l = torch.Generator().manual_seed(int(seed))
        gen_u = torch.Generator().manual_seed(int(seed) + 1)

        step_idx = 0
        model.train()
        for epoch in range(int(self.spec.max_epochs)):
            iter_l = cycle_batches(
                X_l,
                y_l,
                batch_size=int(self.spec.batch_size),
                generator=gen_l,
                steps=steps_per_epoch,
            )
            iter_u_idx = cycle_batch_indices(
                int(X_u.shape[0]),
                batch_size=int(self.spec.batch_size),
                generator=gen_u,
                device=X_u.device,
                steps=steps_per_epoch,
            )
            for step, ((x_lb, y_lb), idx_u) in enumerate(zip(iter_l, iter_u_idx, strict=False)):
                x_u = X_u[idx_u]

                logits_l = extract_logits(model(x_lb))
                if int(logits_l.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                if y_lb.min().item() < 0 or y_lb.max().item() >= int(logits_l.shape[1]):
                    raise InductiveValidationError("y_l labels must be within [0, n_classes).")

                sup_loss = torch.nn.functional.cross_entropy(logits_l, y_lb)

                vat_loss = self._vat_loss(
                    model,
                    x_u,
                    xi=float(self.spec.xi),
                    eps=float(self.spec.eps),
                    num_iters=int(self.spec.num_iters),
                    freeze_bn=bool(self.spec.freeze_bn),
                    detach_target=bool(self.spec.detach_target),
                )

                warm = 1.0 if warmup_steps <= 0 else min(float(step_idx) / float(warmup_steps), 1.0)
                loss = sup_loss + float(self.spec.lambda_u) * vat_loss * float(warm)

                if step == 0:
                    logger.debug(
                        "VAT epoch=%s warm=%.3f sup_loss=%.4f vat_loss=%.4f eps=%s xi=%s",
                        epoch,
                        float(warm),
                        float(sup_loss.item()),
                        float(vat_loss.item()),
                        self.spec.eps,
                        self.spec.xi,
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_idx += 1

        self._bundle = bundle
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, X: Any) -> Any:
        if self._bundle is None:
            raise RuntimeError("VATMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if backend != "torch":
            raise InductiveValidationError("VAT predict_proba requires torch tensors.")
        torch = optional_import("torch", extra="inductive-torch")
        if not isinstance(X, torch.Tensor):
            raise InductiveValidationError("predict_proba requires torch.Tensor inputs.")

        model = self._bundle.model
        was_training = model.training
        model.eval()
        with torch.no_grad():
            logits = extract_logits(model(X))
            if int(logits.ndim) != 2:
                raise InductiveValidationError("Model logits must be 2D (batch, classes).")
            proba = torch.softmax(logits, dim=1)
        if was_training:
            model.train()
        return proba

    def predict(self, X: Any) -> Any:
        proba = self.predict_proba(X)
        return proba.argmax(dim=1)
