from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

import numpy as np

from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.deep_utils import (
    concat_data,
    get_torch_device,
    get_torch_len,
    slice_data,
)
from modssc.inductive.methods.utils import (
    BaseClassifierSpec,
    build_classifier,
    detect_backend,
    ensure_1d_labels,
    ensure_1d_labels_torch,
    ensure_classifier_backend,
    ensure_cpu_device,
    ensure_numpy_data,
    ensure_torch_data,
    predict_in_batches,
    predict_scores_in_batches,
    select_confident,
    select_confident_torch,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec

logger = logging.getLogger(__name__)

_TRAINING_MODES = frozenset({"iterative_threshold", "joint_mlp"})


def _lee2013_alpha(
    epoch: int,
    *,
    alpha_final: float = 3.0,
    start_epoch: int = 100,
    end_epoch: int = 600,
) -> float:
    """Equation (16) from Lee (2013), for the +PL run without DAE."""

    t = int(epoch)
    if t < int(start_epoch):
        return 0.0
    if t < int(end_epoch):
        return float(alpha_final) * (t - int(start_epoch)) / (int(end_epoch) - int(start_epoch))
    return float(alpha_final)


def _lee2013_learning_rate(epoch: int, *, initial: float = 1.5, decay: float = 0.998) -> float:
    """Exponentially decaying learning rate from equation (12)."""

    return float(initial) * float(decay) ** int(epoch)


def _lee2013_momentum(
    epoch: int,
    *,
    initial: float = 0.5,
    final: float = 0.99,
    ramp_epochs: int = 500,
) -> float:
    """Linearly increasing momentum from equation (13)."""

    t = int(epoch)
    if t >= int(ramp_epochs):
        return float(final)
    ratio = float(t) / float(ramp_epochs)
    return ratio * float(final) + (1.0 - ratio) * float(initial)


def _lee2013_joint_loss(
    logits_l: Any,
    y_l: Any,
    logits_u: Any,
    *,
    alpha: float,
    n_classes: int,
) -> tuple[Any, Any, Any, Any]:
    """Equation (15) with hard pseudo-labels from equation (14)."""

    torch = optional_import("torch", extra="inductive-torch")
    functional = torch.nn.functional
    y_l_one_hot = functional.one_hot(y_l.to(dtype=torch.int64), num_classes=int(n_classes)).to(
        dtype=logits_l.dtype
    )
    pseudo = torch.argmax(torch.sigmoid(logits_u.detach()), dim=1)
    y_u_one_hot = functional.one_hot(pseudo, num_classes=int(n_classes)).to(dtype=logits_u.dtype)

    supervised = (
        functional.binary_cross_entropy_with_logits(logits_l, y_l_one_hot, reduction="none")
        .sum(dim=1)
        .mean()
    )
    unsupervised = (
        functional.binary_cross_entropy_with_logits(logits_u, y_u_one_hot, reduction="none")
        .sum(dim=1)
        .mean()
    )
    total = supervised + float(alpha) * unsupervised
    return total, supervised, unsupervised, pseudo


def _lee2013_sgd_step(
    parameters: Iterable[Any],
    momentum_buffers: dict[int, Any],
    *,
    learning_rate: float,
    momentum: float,
) -> None:
    """Apply equations (10)-(11), including Lee's ``(1 - p(t))`` factor."""

    torch = optional_import("torch", extra="inductive-torch")
    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is None:
                continue
            key = id(parameter)
            update = momentum_buffers.get(key)
            if update is None:
                update = torch.zeros_like(parameter)
                momentum_buffers[key] = update
            update.mul_(float(momentum)).add_(
                parameter.grad,
                alpha=-(1.0 - float(momentum)) * float(learning_rate),
            )
            parameter.add_(update)


def _lee2013_flatten_input(
    value: Any,
    *,
    torch: Any,
    name: str,
    input_dim: int,
) -> Any:
    if isinstance(value, Mapping):
        value = value.get("x")
    if not isinstance(value, torch.Tensor):
        raise InductiveValidationError(f"{name} must be a torch.Tensor for lee2013_mnist.")
    if int(value.ndim) < 2:
        raise InductiveValidationError(f"{name} must have shape (n, ...).")
    feature_dim = math.prod(int(size) for size in value.shape[1:])
    if int(feature_dim) != int(input_dim):
        raise InductiveValidationError(
            f"{name} must flatten to {int(input_dim)} MNIST pixels (got {int(feature_dim)})."
        )
    # An explicit second dimension keeps the empty-batch case well-defined;
    # torch cannot infer ``-1`` when the tensor contains zero elements.
    flattened = value.reshape(int(value.shape[0]), int(input_dim)).to(dtype=torch.float32)
    if not bool(torch.isfinite(flattened).all().item()):
        raise InductiveValidationError(f"{name} must contain only finite values.")
    if int(flattened.numel()) > 0:
        minimum = float(flattened.min().item())
        maximum = float(flattened.max().item())
        if minimum < 0.0 or maximum > 1.0:
            raise InductiveValidationError(
                f"{name} must be scaled to [0, 1] for lee2013_mnist "
                f"(observed [{minimum}, {maximum}])."
            )
    return flattened


def _lee2013_epoch_batches(
    *,
    torch: Any,
    n_samples: int,
    batch_size: int,
    n_steps: int,
    generator: Any,
    device: Any,
) -> list[Any]:
    batches: list[Any] = []
    order = torch.randperm(int(n_samples), generator=generator)
    cursor = 0
    for _ in range(int(n_steps)):
        parts: list[Any] = []
        remaining = int(batch_size)
        while remaining > 0:
            available = int(n_samples) - cursor
            take = min(available, remaining)
            parts.append(order[cursor : cursor + take])
            cursor += take
            remaining -= take
            if cursor == int(n_samples):
                order = torch.randperm(int(n_samples), generator=generator)
                cursor = 0
        batches.append(torch.cat(parts).to(device=device))
    return batches


def _build_lee2013_mlp(
    *,
    torch: Any,
    input_dim: int,
    hidden_units: int,
    n_classes: int,
    hidden_dropout: float,
    input_dropout: float,
) -> Any:
    class _Lee2013Dropout(torch.nn.Module):
        """Original (non-inverted) dropout convention used in equation (9)."""

        def __init__(self, probability: float) -> None:
            super().__init__()
            self.p = float(probability)

        def forward(self, inputs: Any) -> Any:
            keep_probability = 1.0 - self.p
            if self.training:
                if self.p == 0.0:
                    return inputs
                mask = torch.rand_like(inputs) >= self.p
                return inputs * mask.to(dtype=inputs.dtype)
            return inputs * keep_probability

    class _Lee2013MLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dropout = _Lee2013Dropout(float(input_dropout))
            self.hidden = torch.nn.Linear(int(input_dim), int(hidden_units))
            self.hidden_dropout = _Lee2013Dropout(float(hidden_dropout))
            self.output = torch.nn.Linear(int(hidden_units), int(n_classes))

        def forward(self, inputs: Any) -> Any:
            hidden = self.input_dropout(inputs)
            hidden = torch.relu(self.hidden(hidden))
            hidden = self.hidden_dropout(hidden)
            # Lee uses sigmoid output units. Returning logits lets the training
            # loss use the numerically stable BCE-with-logits formulation.
            return self.output(hidden)

    return _Lee2013MLP()


class _Lee2013Classifier:
    def __init__(self, model: Any, *, input_dim: int, n_classes: int) -> None:
        self.model = model
        self.input_dim = int(input_dim)
        self.n_classes = int(n_classes)
        self.classes_ = np.arange(self.n_classes, dtype=np.int64)

    def predict_scores(self, X: Any) -> Any:
        torch = optional_import("torch", extra="inductive-torch")
        inputs = _lee2013_flatten_input(
            X,
            torch=torch,
            name="X",
            input_dim=self.input_dim,
        )
        model_device = next(self.model.parameters()).device
        if inputs.device != model_device:
            raise InductiveValidationError(
                f"X must be on the fitted model device {model_device} (got {inputs.device})."
            )
        was_training = bool(self.model.training)
        self.model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(self.model(inputs))
        if was_training:
            self.model.train()
        return scores

    def predict(self, X: Any) -> Any:
        return self.predict_scores(X).argmax(dim=1)


@dataclass(frozen=True)
class PseudoLabelSpec(BaseClassifierSpec):
    max_iter: int = 10
    confidence_threshold: float = 0.95
    max_new_labels: int | None = None
    min_new_labels: int = 1
    paper_input_dim: int = field(default=784, kw_only=True)
    paper_hidden_units: int = field(default=5000, kw_only=True)
    paper_num_classes: int = field(default=10, kw_only=True)
    paper_epochs: int = field(default=601, kw_only=True)
    paper_labeled_batch_size: int = field(default=32, kw_only=True)
    paper_unlabeled_batch_size: int = field(default=256, kw_only=True)
    paper_hidden_dropout: float = field(default=0.5, kw_only=True)
    paper_input_dropout: float = field(default=0.0, kw_only=True)
    paper_initial_learning_rate: float = field(default=1.5, kw_only=True)
    paper_learning_rate_decay: float = field(default=0.998, kw_only=True)
    paper_momentum_initial: float = field(default=0.5, kw_only=True)
    paper_momentum_final: float = field(default=0.99, kw_only=True)
    paper_momentum_ramp_epochs: int = field(default=500, kw_only=True)
    paper_alpha_final: float = field(default=3.0, kw_only=True)
    paper_alpha_start_epoch: int = field(default=100, kw_only=True)
    paper_alpha_end_epoch: int = field(default=600, kw_only=True)
    training_mode: Literal["iterative_threshold", "joint_mlp"] = field(
        default="iterative_threshold", kw_only=True
    )


class PseudoLabelMethod(InductiveMethod):
    """Classic pseudo-labeling with iterative refinement (CPU/GPU)."""

    info = MethodInfo(
        method_id="pseudo_label",
        name="Pseudo-Label",
        year=2013,
        family="classic",
        supports_gpu=True,
        paper_title="Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks",
        paper_pdf="docs/article_code/inductive/2013-Pseudo Label/8_pseudo_label.pdf",
        official_code=None,
    )

    def __init__(self, spec: PseudoLabelSpec | None = None) -> None:
        self.spec = spec or PseudoLabelSpec()
        self._clf: Any | None = None
        self._backend: str | None = None
        self._classifier_backend: str | None = None
        self.diagnostics_: dict[str, Any] = {}

    def _validate_training_mode(self) -> str:
        mode = str(self.spec.training_mode)
        if mode not in _TRAINING_MODES:
            raise InductiveValidationError(
                f"training_mode must be one of {sorted(_TRAINING_MODES)!r}."
            )
        if mode == "iterative_threshold":
            return mode
        if self.spec.classifier_backend != "torch":
            raise InductiveValidationError(
                "training_mode='joint_mlp' requires classifier_backend='torch'."
            )
        positive_ints = {
            "paper_input_dim": self.spec.paper_input_dim,
            "paper_hidden_units": self.spec.paper_hidden_units,
            "paper_num_classes": self.spec.paper_num_classes,
            "paper_epochs": self.spec.paper_epochs,
            "paper_labeled_batch_size": self.spec.paper_labeled_batch_size,
            "paper_unlabeled_batch_size": self.spec.paper_unlabeled_batch_size,
            "paper_momentum_ramp_epochs": self.spec.paper_momentum_ramp_epochs,
        }
        for name, value in positive_ints.items():
            if int(value) <= 0:
                raise InductiveValidationError(f"{name} must be > 0.")
        if int(self.spec.paper_num_classes) < 2:
            raise InductiveValidationError("paper_num_classes must be >= 2.")
        for name, value in (
            ("paper_hidden_dropout", self.spec.paper_hidden_dropout),
            ("paper_input_dropout", self.spec.paper_input_dropout),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise InductiveValidationError(f"{name} must be in [0, 1).")
        if float(self.spec.paper_initial_learning_rate) <= 0.0:
            raise InductiveValidationError("paper_initial_learning_rate must be > 0.")
        if not 0.0 < float(self.spec.paper_learning_rate_decay) <= 1.0:
            raise InductiveValidationError("paper_learning_rate_decay must be in (0, 1].")
        if not 0.0 <= float(self.spec.paper_momentum_initial) < 1.0:
            raise InductiveValidationError("paper_momentum_initial must be in [0, 1).")
        if not 0.0 <= float(self.spec.paper_momentum_final) < 1.0:
            raise InductiveValidationError("paper_momentum_final must be in [0, 1).")
        if float(self.spec.paper_momentum_initial) > float(self.spec.paper_momentum_final):
            raise InductiveValidationError(
                "paper_momentum_initial must not exceed paper_momentum_final."
            )
        if float(self.spec.paper_alpha_final) < 0.0:
            raise InductiveValidationError("paper_alpha_final must be >= 0.")
        if int(self.spec.paper_alpha_start_epoch) < 0:
            raise InductiveValidationError("paper_alpha_start_epoch must be >= 0.")
        if int(self.spec.paper_alpha_end_epoch) <= int(self.spec.paper_alpha_start_epoch):
            raise InductiveValidationError(
                "paper_alpha_end_epoch must be greater than paper_alpha_start_epoch."
            )
        return mode

    def _fit_joint_mlp(
        self,
        data: Any,
        *,
        device: DeviceSpec,
        seed: int,
        started_at: float,
    ) -> PseudoLabelMethod:
        torch = optional_import("torch", extra="inductive-torch")
        ds = ensure_torch_data(data, device=device)
        if ds.X_u is None or int(get_torch_len(ds.X_u)) == 0:
            raise InductiveValidationError(
                "lee2013_mnist +PL requires a non-empty unlabeled partition."
            )

        X_l = _lee2013_flatten_input(
            ds.X_l,
            torch=torch,
            name="X_l",
            input_dim=int(self.spec.paper_input_dim),
        )
        X_u = _lee2013_flatten_input(
            ds.X_u,
            torch=torch,
            name="X_u",
            input_dim=int(self.spec.paper_input_dim),
        )
        if int(X_l.shape[0]) == 0:
            raise InductiveValidationError("X_l must be non-empty.")
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l").to(dtype=torch.int64)
        if int(y_l.min().item()) < 0 or int(y_l.max().item()) >= int(self.spec.paper_num_classes):
            raise InductiveValidationError(
                "y_l values must be in [0, paper_num_classes) for lee2013_mnist."
            )

        cuda_devices: list[int] = []
        if X_l.device.type == "cuda":
            cuda_devices = [
                int(X_l.device.index)
                if X_l.device.index is not None
                else int(torch.cuda.current_device())
            ]

        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(int(seed))
            if cuda_devices:
                torch.cuda.manual_seed_all(int(seed))

            model = _build_lee2013_mlp(
                torch=torch,
                input_dim=int(self.spec.paper_input_dim),
                hidden_units=int(self.spec.paper_hidden_units),
                n_classes=int(self.spec.paper_num_classes),
                hidden_dropout=float(self.spec.paper_hidden_dropout),
                input_dropout=float(self.spec.paper_input_dropout),
            ).to(device=X_l.device, dtype=torch.float32)

            n_labeled = int(X_l.shape[0])
            n_unlabeled = int(X_u.shape[0])
            steps_per_epoch = max(
                math.ceil(n_labeled / int(self.spec.paper_labeled_batch_size)),
                math.ceil(n_unlabeled / int(self.spec.paper_unlabeled_batch_size)),
            )
            index_generator = torch.Generator(device="cpu")
            index_generator.manual_seed(int(seed))
            momentum_buffers: dict[int, Any] = {}
            alpha_history: list[float] = []
            pseudo_counts = torch.zeros(
                (int(self.spec.paper_num_classes),),
                dtype=torch.int64,
                device=X_l.device,
            )
            pseudo_labels_assigned = 0
            pseudo_label_updates = 0

            for epoch_index in range(int(self.spec.paper_epochs)):
                alpha = _lee2013_alpha(
                    epoch_index,
                    alpha_final=float(self.spec.paper_alpha_final),
                    start_epoch=int(self.spec.paper_alpha_start_epoch),
                    end_epoch=int(self.spec.paper_alpha_end_epoch),
                )
                learning_rate = _lee2013_learning_rate(
                    epoch_index,
                    initial=float(self.spec.paper_initial_learning_rate),
                    decay=float(self.spec.paper_learning_rate_decay),
                )
                momentum = _lee2013_momentum(
                    epoch_index,
                    initial=float(self.spec.paper_momentum_initial),
                    final=float(self.spec.paper_momentum_final),
                    ramp_epochs=int(self.spec.paper_momentum_ramp_epochs),
                )
                alpha_history.append(float(alpha))
                labeled_batches = _lee2013_epoch_batches(
                    torch=torch,
                    n_samples=n_labeled,
                    batch_size=int(self.spec.paper_labeled_batch_size),
                    n_steps=steps_per_epoch,
                    generator=index_generator,
                    device=X_l.device,
                )
                unlabeled_batches = _lee2013_epoch_batches(
                    torch=torch,
                    n_samples=n_unlabeled,
                    batch_size=int(self.spec.paper_unlabeled_batch_size),
                    n_steps=steps_per_epoch,
                    generator=index_generator,
                    device=X_l.device,
                )

                model.train()
                for labeled_idx, unlabeled_idx in zip(
                    labeled_batches, unlabeled_batches, strict=True
                ):
                    for parameter in model.parameters():
                        parameter.grad = None
                    logits_l = model(X_l[labeled_idx])
                    logits_u = model(X_u[unlabeled_idx])
                    total, _, _, pseudo = _lee2013_joint_loss(
                        logits_l,
                        y_l[labeled_idx],
                        logits_u,
                        alpha=alpha,
                        n_classes=int(self.spec.paper_num_classes),
                    )
                    if not bool(torch.isfinite(total).item()):
                        raise InductiveValidationError(
                            "lee2013_mnist training produced a non-finite loss."
                        )
                    total.backward()
                    _lee2013_sgd_step(
                        model.parameters(),
                        momentum_buffers,
                        learning_rate=learning_rate,
                        momentum=momentum,
                    )
                    pseudo_counts.add_(
                        torch.bincount(
                            pseudo,
                            minlength=int(self.spec.paper_num_classes),
                        )
                    )
                    pseudo_labels_assigned += int(pseudo.numel())
                    pseudo_label_updates += 1

        model.eval()
        final_pseudo_counts = torch.zeros(
            (int(self.spec.paper_num_classes),),
            dtype=torch.int64,
            device=X_u.device,
        )
        with torch.no_grad():
            for start in range(0, n_unlabeled, int(self.spec.paper_unlabeled_batch_size)):
                logits_u = model(X_u[start : start + int(self.spec.paper_unlabeled_batch_size)])
                final_pseudo = torch.argmax(torch.sigmoid(logits_u), dim=1)
                final_pseudo_counts.add_(
                    torch.bincount(
                        final_pseudo,
                        minlength=int(self.spec.paper_num_classes),
                    )
                )
        self._clf = _Lee2013Classifier(
            model,
            input_dim=int(self.spec.paper_input_dim),
            n_classes=int(self.spec.paper_num_classes),
        )
        self._backend = "torch"
        self._classifier_backend = "torch"
        self.diagnostics_ = {
            "training_mode": self.spec.training_mode,
            "dae_pretraining": False,
            "dropout_convention": "lee2013_non_inverted",
            "hidden_dropout_probability": float(self.spec.paper_hidden_dropout),
            "input_dropout_probability": float(self.spec.paper_input_dropout),
            "n_labeled": int(n_labeled),
            "n_unlabeled": int(n_unlabeled),
            "labeled_batch_size": int(self.spec.paper_labeled_batch_size),
            "unlabeled_batch_size": int(self.spec.paper_unlabeled_batch_size),
            "epochs_completed": int(self.spec.paper_epochs),
            "steps_per_epoch": int(steps_per_epoch),
            "parameter_updates": int(pseudo_label_updates),
            "alpha_history": alpha_history,
            "alpha_first": float(alpha_history[0]),
            "alpha_last": float(alpha_history[-1]),
            "alpha_final": float(self.spec.paper_alpha_final),
            "alpha_start_epoch": int(self.spec.paper_alpha_start_epoch),
            "alpha_end_epoch": int(self.spec.paper_alpha_end_epoch),
            "schedule_epoch_first": 0,
            "schedule_epoch_last": int(self.spec.paper_epochs) - 1,
            "alpha_reached_final": bool(
                int(self.spec.paper_epochs) - 1 >= int(self.spec.paper_alpha_end_epoch)
            ),
            "learning_rate_first": _lee2013_learning_rate(
                0,
                initial=float(self.spec.paper_initial_learning_rate),
                decay=float(self.spec.paper_learning_rate_decay),
            ),
            "learning_rate_last": _lee2013_learning_rate(
                int(self.spec.paper_epochs) - 1,
                initial=float(self.spec.paper_initial_learning_rate),
                decay=float(self.spec.paper_learning_rate_decay),
            ),
            "momentum_first": _lee2013_momentum(
                0,
                initial=float(self.spec.paper_momentum_initial),
                final=float(self.spec.paper_momentum_final),
                ramp_epochs=int(self.spec.paper_momentum_ramp_epochs),
            ),
            "momentum_last": _lee2013_momentum(
                int(self.spec.paper_epochs) - 1,
                initial=float(self.spec.paper_momentum_initial),
                final=float(self.spec.paper_momentum_final),
                ramp_epochs=int(self.spec.paper_momentum_ramp_epochs),
            ),
            "pseudo_label_updates": int(pseudo_label_updates),
            "pseudo_labels_assigned_total": int(pseudo_labels_assigned),
            "pseudo_label_class_counts": [
                int(value) for value in pseudo_counts.detach().cpu().tolist()
            ],
            "final_pseudo_labels_assigned": int(n_unlabeled),
            "final_pseudo_label_class_counts": [
                int(value) for value in final_pseudo_counts.detach().cpu().tolist()
            ],
            "confidence_threshold_applied": False,
        }
        logger.info(
            "Finished %s.fit training_mode=%s updates=%s pseudo_labels=%s in %.3fs",
            self.info.method_id,
            self.spec.training_mode,
            pseudo_label_updates,
            pseudo_labels_assigned,
            perf_counter() - started_at,
        )
        return self

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> PseudoLabelMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        training_mode = self._validate_training_mode()
        if training_mode == "joint_mlp":
            return self._fit_joint_mlp(data, device=device, seed=seed, started_at=start)
        backend = detect_backend(data.X_l)
        ensure_classifier_backend(self.spec, backend=backend)
        logger.debug("backend=%s", backend)

        if backend == "numpy":
            ensure_cpu_device(device)
            ds = ensure_numpy_data(data)
            y_l = ensure_1d_labels(ds.y_l, name="y_l")

            if ds.X_u is None or np.asarray(ds.X_u).size == 0:
                clf = build_classifier(self.spec, seed=seed)
                clf.fit(ds.X_l, y_l)
                self._clf = clf
                self._backend = backend
                logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
                return self

            X_l = np.asarray(ds.X_l)
            X_u = np.asarray(ds.X_u)
            y_l = np.asarray(y_l)
            logger.info(
                "Pseudo-label sizes: n_labeled=%s n_unlabeled=%s",
                int(X_l.shape[0]),
                int(X_u.shape[0]),
            )

            if X_l.shape[0] == 0:
                raise InductiveValidationError("X_l must be non-empty.")

            clf = build_classifier(self.spec, seed=seed)

            X_u_curr = X_u
            iter_count = 0
            while iter_count < int(self.spec.max_iter):
                clf.fit(X_l, y_l)

                if X_u_curr.shape[0] == 0:
                    break

                scores = predict_scores_in_batches(clf, X_u_curr, backend=backend)
                idx = select_confident(
                    scores,
                    threshold=float(self.spec.confidence_threshold),
                    max_new=self.spec.max_new_labels,
                )
                logger.debug(
                    "Pseudo-label iter=%s accepted=%s remaining=%s",
                    iter_count,
                    int(idx.size),
                    int(X_u_curr.shape[0]),
                )
                if idx.size < int(self.spec.min_new_labels):
                    break

                y_u = np.asarray(predict_in_batches(clf, X_u_curr[idx], backend=backend))
                X_l = np.concatenate([X_l, X_u_curr[idx]], axis=0)
                y_l = np.concatenate([y_l, y_u], axis=0)

                keep = np.ones((X_u_curr.shape[0],), dtype=bool)
                keep[idx] = False
                X_u_curr = X_u_curr[keep]

                iter_count += 1

            clf.fit(X_l, y_l)
            self._clf = clf
            self._backend = backend
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        ds = ensure_torch_data(data, device=device)
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        torch = optional_import("torch", extra="inductive-torch")
        if ds.X_u is None or int(get_torch_len(ds.X_u)) == 0:
            clf = build_classifier(self.spec, seed=seed)
            clf.fit(ds.X_l, y_l)
            self._clf = clf
            self._backend = backend
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        X_l = ds.X_l
        X_u = ds.X_u
        if int(get_torch_len(X_l)) == 0:
            raise InductiveValidationError("X_l must be non-empty.")
        logger.info(
            "Pseudo-label sizes: n_labeled=%s n_unlabeled=%s",
            int(get_torch_len(X_l)),
            int(get_torch_len(X_u)),
        )

        clf = build_classifier(self.spec, seed=seed)

        X_u_curr = X_u
        iter_count = 0
        while iter_count < int(self.spec.max_iter):
            clf.fit(X_l, y_l)

            if int(get_torch_len(X_u_curr)) == 0:
                break

            scores = predict_scores_in_batches(clf, X_u_curr, backend=backend)
            idx = select_confident_torch(
                scores,
                threshold=float(self.spec.confidence_threshold),
                max_new=self.spec.max_new_labels,
            )
            logger.debug(
                "Pseudo-label iter=%s accepted=%s remaining=%s",
                iter_count,
                int(idx.numel()),
                int(get_torch_len(X_u_curr)),
            )
            if int(idx.numel()) < int(self.spec.min_new_labels):
                break

            x_u_sel = slice_data(X_u_curr, idx)
            y_u = predict_in_batches(clf, x_u_sel, backend=backend)
            X_l = concat_data([X_l, x_u_sel])
            y_l = torch.cat([y_l, y_u], dim=0)

            mask = torch.ones(
                (int(get_torch_len(X_u_curr)),),
                dtype=torch.bool,
                device=get_torch_device(X_u_curr),
            )
            mask[idx] = False
            X_u_curr = slice_data(X_u_curr, mask)

            iter_count += 1

        clf.fit(X_l, y_l)
        self._clf = clf
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("PseudoLabelMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict_proba input backend mismatch.")
        scores = predict_scores_in_batches(self._clf, X, backend=backend)
        if backend == "numpy":
            row_sum = scores.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0.0] = 1.0
            return (scores / row_sum).astype(np.float32, copy=False)
        torch = optional_import("torch", extra="inductive-torch")
        row_sum = scores.sum(dim=1, keepdim=True)
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        return scores / row_sum

    def predict(self, X: Any) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("PseudoLabelMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict input backend mismatch.")
        return predict_in_batches(self._clf, X, backend=backend)
