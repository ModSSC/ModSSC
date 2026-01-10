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


def _entropy(probs: Any) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    return -(probs * torch.log(probs + 1e-12)).sum(dim=1)


def _js_divergence(p1: Any, p2: Any) -> Any:
    p_avg = 0.5 * (p1 + p2)
    return (_entropy(p_avg) - 0.5 * _entropy(p1) - 0.5 * _entropy(p2)).mean()


def _soft_cross_entropy(target_probs: Any, logits: Any) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    if target_probs.shape != log_probs.shape:
        raise InductiveValidationError("Target distribution shape mismatch.")
    return -(target_probs * log_probs).sum(dim=1).mean()


def _fgsm_adversarial(
    model: Any,
    x_l: Any,
    y_l: Any,
    x_u: Any | None,
    *,
    epsilon: float,
    freeze_bn: bool,
    clip_min: float | None,
    clip_max: float | None,
) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    if x_u is None or int(x_u.shape[0]) == 0:
        x_all = x_l
        n_l = int(x_l.shape[0])
    else:
        x_all = torch.cat([x_l, x_u], dim=0)
        n_l = int(x_l.shape[0])

    x_adv = x_all.detach().clone().requires_grad_(True)
    with freeze_batchnorm(model, enabled=bool(freeze_bn)):
        logits = extract_logits(model(x_adv))
    if int(logits.ndim) != 2:
        raise InductiveValidationError("Model logits must be 2D (batch, classes).")
    if int(logits.shape[0]) != int(x_adv.shape[0]):
        raise InductiveValidationError("Model logits batch size does not match inputs.")

    if n_l > 0:
        if int(y_l.shape[0]) != n_l:
            raise InductiveValidationError("Labeled batch size mismatch.")
        targets = torch.empty((int(logits.shape[0]),), dtype=torch.int64, device=logits.device)
        targets[:n_l] = y_l
        if int(logits.shape[0]) > n_l:
            pseudo = logits.detach().argmax(dim=1)
            targets[n_l:] = pseudo[n_l:]
    else:
        targets = logits.detach().argmax(dim=1)

    loss = torch.nn.functional.cross_entropy(logits, targets)
    grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
    adv = x_adv + float(epsilon) * grad.sign()

    if clip_min is not None and clip_max is not None:
        adv = torch.clamp(adv, min=float(clip_min), max=float(clip_max))
    elif clip_min is not None:
        adv = torch.clamp(adv, min=float(clip_min))
    elif clip_max is not None:
        adv = torch.clamp(adv, max=float(clip_max))

    return adv.detach()


@dataclass(frozen=True)
class DeepCoTrainingSpec:
    """Specification for Deep Co-Training (torch-only)."""

    model_bundle_1: TorchModelBundle | None = None
    model_bundle_2: TorchModelBundle | None = None
    lambda_cot: float = 1.0
    lambda_dif: float = 1.0
    adv_eps: float = 0.03
    batch_size: int = 64
    max_epochs: int = 1
    freeze_bn: bool = True
    detach_target: bool = True
    adv_clip_min: float | None = None
    adv_clip_max: float | None = None


class DeepCoTrainingMethod(InductiveMethod):
    """Deep Co-Training with adversarial view difference (torch-only)."""

    info = MethodInfo(
        method_id="deep_co_training",
        name="Deep Co-Training",
        year=2018,
        family="agreement",
        supports_gpu=True,
        paper_title="Deep Co-Training for Semi-Supervised Image Recognition",
        paper_pdf="https://arxiv.org/abs/1803.05984",
        official_code="",
    )

    def __init__(self, spec: DeepCoTrainingSpec | None = None) -> None:
        self.spec = spec or DeepCoTrainingSpec()
        self._bundle1: TorchModelBundle | None = None
        self._bundle2: TorchModelBundle | None = None
        self._backend: str | None = None

    def _check_models(self, model1: Any, model2: Any) -> None:
        if model1 is model2:
            raise InductiveValidationError(
                "model_bundle_1 and model_bundle_2 must wrap distinct models."
            )
        params1 = list(model1.parameters())
        params2 = list(model2.parameters())
        ids1 = {id(p) for p in params1}
        for p in params2:
            if id(p) in ids1:
                raise InductiveValidationError(
                    "model_bundle_1 and model_bundle_2 must not share parameters."
                )

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> DeepCoTrainingMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "params lambda_cot=%s lambda_dif=%s adv_eps=%s batch_size=%s max_epochs=%s "
            "freeze_bn=%s detach_target=%s adv_clip_min=%s adv_clip_max=%s "
            "has_bundle_1=%s has_bundle_2=%s device=%s seed=%s",
            self.spec.lambda_cot,
            self.spec.lambda_dif,
            self.spec.adv_eps,
            self.spec.batch_size,
            self.spec.max_epochs,
            self.spec.freeze_bn,
            self.spec.detach_target,
            self.spec.adv_clip_min,
            self.spec.adv_clip_max,
            bool(self.spec.model_bundle_1),
            bool(self.spec.model_bundle_2),
            device,
            seed,
        )
        if data is None:
            raise InductiveValidationError("data must not be None.")

        backend = detect_backend(data.X_l)
        if backend != "torch":
            raise InductiveValidationError("Deep Co-Training requires torch tensors.")

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        X_u = ds.X_u if ds.X_u is not None else ds.X_u_w
        if X_u is None:
            raise InductiveValidationError("Deep Co-Training requires X_u (unlabeled data).")

        X_l = ds.X_l
        if int(X_l.shape[0]) == 0:
            raise InductiveValidationError("X_l must be non-empty.")
        if int(X_u.shape[0]) == 0:
            raise InductiveValidationError("X_u must be non-empty.")

        y_l = ds.y_l
        if not isinstance(y_l, torch.Tensor):
            raise InductiveValidationError("y_l must be a torch.Tensor.")
        if y_l.dtype != torch.int64:
            raise InductiveValidationError("y_l must be int64 for torch cross entropy.")
        y_l = ensure_1d_labels_torch(y_l, name="y_l")

        logger.info(
            "Deep Co-Training sizes: n_labeled=%s n_unlabeled=%s",
            int(X_l.shape[0]),
            int(X_u.shape[0]),
        )

        ensure_float_tensor(X_l, name="X_l")
        ensure_float_tensor(X_u, name="X_u")

        if self.spec.model_bundle_1 is None or self.spec.model_bundle_2 is None:
            raise InductiveValidationError(
                "model_bundle_1 and model_bundle_2 must be provided for Deep Co-Training."
            )
        bundle1 = ensure_model_bundle(self.spec.model_bundle_1)
        bundle2 = ensure_model_bundle(self.spec.model_bundle_2)
        model1 = bundle1.model
        model2 = bundle2.model
        optimizer1 = bundle1.optimizer
        optimizer2 = bundle2.optimizer

        ensure_model_device(model1, device=X_l.device)
        ensure_model_device(model2, device=X_l.device)
        self._check_models(model1, model2)

        if int(self.spec.batch_size) <= 0:
            raise InductiveValidationError("batch_size must be >= 1.")
        if int(self.spec.max_epochs) <= 0:
            raise InductiveValidationError("max_epochs must be >= 1.")
        if float(self.spec.lambda_cot) < 0:
            raise InductiveValidationError("lambda_cot must be >= 0.")
        if float(self.spec.lambda_dif) < 0:
            raise InductiveValidationError("lambda_dif must be >= 0.")
        if float(self.spec.adv_eps) < 0:
            raise InductiveValidationError("adv_eps must be >= 0.")
        if (
            self.spec.adv_clip_min is not None
            and self.spec.adv_clip_max is not None
            and float(self.spec.adv_clip_min) > float(self.spec.adv_clip_max)
        ):
            raise InductiveValidationError("adv_clip_min must be <= adv_clip_max.")

        steps_l = num_batches(int(X_l.shape[0]), int(self.spec.batch_size))
        steps_u = num_batches(int(X_u.shape[0]), int(self.spec.batch_size))
        steps_per_epoch = max(int(steps_l), int(steps_u))

        gen_l1 = torch.Generator().manual_seed(int(seed))
        gen_l2 = torch.Generator().manual_seed(int(seed) + 1)
        gen_u = torch.Generator().manual_seed(int(seed) + 2)

        model1.train()
        model2.train()
        for epoch in range(int(self.spec.max_epochs)):
            iter_l1 = cycle_batches(
                X_l,
                y_l,
                batch_size=int(self.spec.batch_size),
                generator=gen_l1,
                steps=steps_per_epoch,
            )
            iter_l2 = cycle_batches(
                X_l,
                y_l,
                batch_size=int(self.spec.batch_size),
                generator=gen_l2,
                steps=steps_per_epoch,
            )
            iter_u_idx = cycle_batch_indices(
                int(X_u.shape[0]),
                batch_size=int(self.spec.batch_size),
                generator=gen_u,
                device=X_u.device,
                steps=steps_per_epoch,
            )
            for step, ((x_lb1, y_lb1), (x_lb2, y_lb2), idx_u) in enumerate(
                zip(iter_l1, iter_l2, iter_u_idx, strict=False)
            ):
                x_u = X_u[idx_u]
                if int(x_lb1.shape[0]) != int(x_lb2.shape[0]):
                    raise InductiveValidationError("Labeled batch sizes must match.")

                x1_adv = _fgsm_adversarial(
                    model1,
                    x_lb1,
                    y_lb1,
                    x_u,
                    epsilon=float(self.spec.adv_eps),
                    freeze_bn=bool(self.spec.freeze_bn),
                    clip_min=self.spec.adv_clip_min,
                    clip_max=self.spec.adv_clip_max,
                )
                x2_adv = _fgsm_adversarial(
                    model2,
                    x_lb2,
                    y_lb2,
                    x_u,
                    epsilon=float(self.spec.adv_eps),
                    freeze_bn=bool(self.spec.freeze_bn),
                    clip_min=self.spec.adv_clip_min,
                    clip_max=self.spec.adv_clip_max,
                )

                x1_all = torch.cat([x_lb1, x_u], dim=0)
                x2_all = torch.cat([x_lb2, x_u], dim=0)

                logits1_all = extract_logits(model1(x1_all))
                logits2_all = extract_logits(model2(x2_all))
                if int(logits1_all.ndim) != 2 or int(logits2_all.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                if int(logits1_all.shape[0]) != int(x1_all.shape[0]):
                    raise InductiveValidationError("Model1 logits batch size mismatch.")
                if int(logits2_all.shape[0]) != int(x2_all.shape[0]):
                    raise InductiveValidationError("Model2 logits batch size mismatch.")
                if int(logits1_all.shape[1]) != int(logits2_all.shape[1]):
                    raise InductiveValidationError("Models must agree on class count.")

                n_l1 = int(x_lb1.shape[0])
                n_l2 = int(x_lb2.shape[0])
                logits1_l = logits1_all[:n_l1]
                logits2_l = logits2_all[:n_l2]
                logits1_u = logits1_all[n_l1:]
                logits2_u = logits2_all[n_l2:]

                loss_sup = torch.nn.functional.cross_entropy(logits1_l, y_lb1)
                loss_sup = loss_sup + torch.nn.functional.cross_entropy(logits2_l, y_lb2)

                probs1_u = torch.softmax(logits1_u, dim=1)
                probs2_u = torch.softmax(logits2_u, dim=1)
                loss_cot = _js_divergence(probs1_u, probs2_u)

                probs1_all = torch.softmax(logits1_all, dim=1)
                probs2_all = torch.softmax(logits2_all, dim=1)
                if bool(self.spec.detach_target):
                    probs1_all = probs1_all.detach()
                    probs2_all = probs2_all.detach()

                logits2_adv = extract_logits(model2(x1_adv))
                logits1_adv = extract_logits(model1(x2_adv))
                if int(logits2_adv.ndim) != 2 or int(logits1_adv.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                if int(logits2_adv.shape[0]) != int(x1_adv.shape[0]):
                    raise InductiveValidationError("Model2 adversarial logits batch mismatch.")
                if int(logits1_adv.shape[0]) != int(x2_adv.shape[0]):
                    raise InductiveValidationError("Model1 adversarial logits batch mismatch.")
                if int(logits2_adv.shape[1]) != int(probs1_all.shape[1]):
                    raise InductiveValidationError("Model2 logits class mismatch.")
                if int(logits1_adv.shape[1]) != int(probs2_all.shape[1]):
                    raise InductiveValidationError("Model1 logits class mismatch.")

                loss_dif = _soft_cross_entropy(probs1_all, logits2_adv)
                loss_dif = loss_dif + _soft_cross_entropy(probs2_all, logits1_adv)

                loss = loss_sup
                if float(self.spec.lambda_cot) != 0.0:
                    loss = loss + float(self.spec.lambda_cot) * loss_cot
                if float(self.spec.lambda_dif) != 0.0:
                    loss = loss + float(self.spec.lambda_dif) * loss_dif

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()

                logger.debug(
                    "Deep Co-Training epoch=%s step=%s loss=%.4f sup=%.4f cot=%.4f dif=%.4f",
                    epoch,
                    step,
                    float(loss.item()),
                    float(loss_sup.item()),
                    float(loss_cot.item()),
                    float(loss_dif.item()),
                )

        self._bundle1 = bundle1
        self._bundle2 = bundle2
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, X: Any) -> Any:
        if self._bundle1 is None or self._bundle2 is None:
            raise RuntimeError("DeepCoTrainingMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if backend != "torch":
            raise InductiveValidationError("DeepCoTraining predict_proba requires torch tensors.")
        torch = optional_import("torch", extra="inductive-torch")
        if not isinstance(X, torch.Tensor):
            raise InductiveValidationError("predict_proba requires torch.Tensor inputs.")

        model1 = self._bundle1.model
        model2 = self._bundle2.model
        was_training1 = model1.training
        was_training2 = model2.training
        model1.eval()
        model2.eval()
        with torch.no_grad():
            logits1 = extract_logits(model1(X))
            logits2 = extract_logits(model2(X))
            if int(logits1.ndim) != 2 or int(logits2.ndim) != 2:
                raise InductiveValidationError("Model logits must be 2D (batch, classes).")
            if int(logits1.shape[1]) != int(logits2.shape[1]):
                raise InductiveValidationError("Models must agree on class count.")
            probs = torch.softmax(logits1, dim=1) + torch.softmax(logits2, dim=1)
            probs = probs * 0.5
        if was_training1:
            model1.train()
        if was_training2:
            model2.train()
        return probs

    def predict(self, X: Any) -> Any:
        proba = self.predict_proba(X)
        return proba.argmax(dim=1)
