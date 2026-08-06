from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.optional import optional_import

from .common import TwoLayerMLP as _MLP
from .common import normalize_device_name, prepare_data_cached, set_torch_seed, spmm, torch

logger = logging.getLogger(__name__)

_GRAND_OFFICIAL_COMMIT = "7a2fd6e7c3f20ca2c84b06ec1c5dc7f227dbfe2b"
_TRAINING_MODES = frozenset({"legacy", "random_propagation_consistency"})


def _propagate(
    x: Any,
    edge_index: Any,
    edge_weight: Any,
    *,
    n_nodes: int,
    steps: int,
) -> Any:
    """Historical ModSSC ``A^K X`` propagation used by standardized runs."""

    out = x
    for _ in range(int(steps)):
        out = spmm(edge_index, edge_weight, out, n_nodes=n_nodes)
    return out


def _mixed_order_propagate(
    x: Any,
    edge_index: Any,
    edge_weight: Any,
    *,
    n_nodes: int,
    steps: int,
) -> Any:
    """Return ``(I + A + ... + A^K) X / (K + 1)``.

    GRAND separates propagation from the MLP.  The original ModSSC baseline
    only returned ``A^K X``; that is a different model and over-smooths much
    more aggressively.
    """

    if int(steps) < 0:
        raise ValueError("steps must be >= 0")
    current = x
    total = x.clone()
    for _ in range(int(steps)):
        current = spmm(edge_index, edge_weight, current, n_nodes=n_nodes).detach()
        total = total + current
    return (total / float(int(steps) + 1)).detach()


def _dropnode(x: Any, *, drop_probability: float, training: bool) -> Any:
    """Apply GRAND DropNode, dropping complete feature rows.

    Match the official implementation: training masks complete rows without
    inverted-dropout rescaling, while inference multiplies every feature by
    the keep probability so both paths have the same expected scale.
    """

    probability = float(drop_probability)
    if not (0.0 <= probability < 1.0):
        raise ValueError("drop_probability must be in [0, 1)")
    if probability == 0.0:
        return x
    keep_probability = 1.0 - probability
    if not training:
        return x * keep_probability
    # The reference implementation deliberately samples DropNode on the CPU
    # and only then moves the mask to CUDA. Keeping that separate RNG stream
    # is material: MLP dropout samples use the CUDA generator.
    keep_probabilities = torch.full(
        (int(x.shape[0]),),
        keep_probability,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    mask = torch.bernoulli(keep_probabilities).unsqueeze(1)
    return x * mask.to(device=x.device, dtype=x.dtype)


def _sharpen(probabilities: Any, *, temperature: float) -> Any:
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be > 0")
    powered = probabilities.pow(1.0 / float(temperature))
    return powered / powered.sum(dim=1, keepdim=True)


def _consistency_loss(log_probabilities: list[Any], *, temperature: float) -> Any:
    """Official GRAND squared-distance consistency objective (Eq. 2--3)."""

    if not log_probabilities:
        raise ValueError("log_probabilities must contain at least one augmentation")
    probabilities = [item.exp() for item in log_probabilities]
    center = sum(probabilities) / len(probabilities)
    target = _sharpen(center, temperature=float(temperature)).detach()
    loss = sum((probability - target).pow(2).sum(dim=1).mean() for probability in probabilities)
    return loss / len(probabilities)


def _grand_objective(
    logits: list[Any],
    labels: Any,
    train_mask: Any,
    *,
    temperature: float,
    consistency_weight: float,
) -> tuple[Any, Any, Any]:
    """Return the official mean NLL, consistency term, and combined loss."""

    if not logits:
        raise ValueError("logits must contain at least one augmentation")
    log_probabilities = [torch.log_softmax(item, dim=-1) for item in logits]
    supervised_loss = sum(
        torch.nn.functional.nll_loss(item[train_mask], labels[train_mask])
        for item in log_probabilities
    ) / len(log_probabilities)
    consistency_loss = _consistency_loss(
        log_probabilities,
        temperature=float(temperature),
    )
    loss = supervised_loss + float(consistency_weight) * consistency_loss
    return supervised_loss, consistency_loss, loss


def _sigmoid_rampup(epoch: int, rampup_epochs: int) -> float:
    """Optional Mean-Teacher-style ramp used by some GRAND reproductions."""

    if int(rampup_epochs) <= 0:
        return 1.0
    clipped = min(max(float(epoch), 0.0), float(rampup_epochs))
    phase = 1.0 - clipped / float(rampup_epochs)
    return float(math.exp(-5.0 * phase * phase))


def _initialize_mlp(model: Any, *, seed: int | None = None) -> None:
    """Reproduce the pinned code's ``MLPLayer.reset_parameters`` trajectory.

    The official layer stores weights as ``(in_features, out_features)`` and
    calls ``normal_(mean=-1/sqrt(out_features), std=1/sqrt(out_features))``.
    ``torch.nn.Linear`` stores the transpose, so sampling in the official
    layout before copying is necessary for seed-level parity.
    """

    if seed is not None:
        # ``nn.Linear`` initialized itself before this function was called.
        # Reset both CPU and CUDA generators so those discarded draws do not
        # shift either the CPU DropNode or CUDA dropout trajectories.
        set_torch_seed(int(seed))
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            stdv = 1.0 / math.sqrt(int(module.out_features))
            official_weight = torch.empty(
                (int(module.in_features), int(module.out_features)),
                dtype=module.weight.dtype,
                device=torch.device("cpu"),
            )
            official_weight.normal_(mean=-stdv, std=stdv)
            official_bias = None
            if module.bias is not None:
                official_bias = torch.empty(
                    (int(module.out_features),),
                    dtype=module.bias.dtype,
                    device=torch.device("cpu"),
                )
                official_bias.normal_(mean=-stdv, std=stdv)
            with torch.no_grad():
                module.weight.copy_(official_weight.transpose(0, 1).to(device=module.weight.device))
                if module.bias is not None and official_bias is not None:
                    module.bias.copy_(official_bias.to(device=module.bias.device))


@dataclass(frozen=True)
class _CheckpointUpdate:
    running_min_loss: float
    running_max_accuracy: float
    bad_epochs: int
    save_checkpoint: bool


def _official_checkpoint_step(
    *,
    val_loss: float,
    val_accuracy: float,
    running_min_loss: float,
    running_max_accuracy: float,
    best_val_loss: float,
    bad_epochs: int,
) -> _CheckpointUpdate:
    """Apply the exact nested checkpoint/patience rule in ``train_grand.py``."""

    resets_patience = val_loss <= running_min_loss or val_accuracy >= running_max_accuracy
    if not resets_patience:
        return _CheckpointUpdate(
            running_min_loss=running_min_loss,
            running_max_accuracy=running_max_accuracy,
            bad_epochs=int(bad_epochs) + 1,
            save_checkpoint=False,
        )
    return _CheckpointUpdate(
        running_min_loss=min(running_min_loss, val_loss),
        running_max_accuracy=max(running_max_accuracy, val_accuracy),
        bad_epochs=0,
        save_checkpoint=val_loss <= best_val_loss,
    )


@dataclass(frozen=True)
class GRANDSpec:
    """GRAND parameters with isolated generic training modes."""

    # Keep the original public order and defaults byte-for-byte compatible with
    # the pre-replication standardized implementation.
    hidden_dim: int = 64
    mlp_dropout: float = 0.5
    prop_steps: int = 8
    dropnode: float = 0.5
    num_samples: int = 4
    lambda_consistency: float = 1.0
    lr: float = 0.01
    weight_decay: float = 5e-4
    max_epochs: int = 200
    patience: int = 50
    add_self_loops: bool = True
    # Advanced extensions are keyword-only so historical positional calls do
    # not silently change meaning.
    training_mode: str = field(default="legacy", kw_only=True)
    input_dropout: float | None = field(default=None, kw_only=True)
    hidden_dropout: float | None = field(default=None, kw_only=True)
    use_batch_norm: bool = field(default=False, kw_only=True)
    temperature: float = field(default=0.5, kw_only=True)
    consistency_rampup_epochs: int = field(default=0, kw_only=True)


class GRANDMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="grand",
        name="GRAND",
        year=2020,
        family="gnn",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="Graph Random Neural Networks for Semi-Supervised Learning on Graphs",
        paper_pdf=(
            "https://papers.nips.cc/paper/2020/file/fb4c835feb0a65cc39739320d7a51c02-Paper.pdf"
        ),
        official_code=f"https://github.com/THUDM/GRAND/tree/{_GRAND_OFFICIAL_COMMIT}",
    )

    def __init__(self, spec: GRANDSpec | None = None) -> None:
        self.spec = spec or GRANDSpec()
        self._device: Any | None = None
        self._model: Any | None = None
        self._edge_index: Any | None = None
        self._edge_weight: Any | None = None
        self._n_nodes: int | None = None
        self._prep_cache: dict[str, Any] = {}
        self.diagnostics_: dict[str, Any] = {}

    def _validate_spec(self) -> None:
        if int(self.spec.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be > 0")
        if int(self.spec.prop_steps) < 0:
            raise ValueError("prop_steps must be >= 0")
        if int(self.spec.num_samples) <= 0:
            raise ValueError("num_samples must be > 0")
        if int(self.spec.max_epochs) <= 0:
            raise ValueError("max_epochs must be > 0")
        if int(self.spec.patience) <= 0:
            raise ValueError("patience must be > 0")
        if int(self.spec.consistency_rampup_epochs) < 0:
            raise ValueError("consistency_rampup_epochs must be >= 0")
        if float(self.spec.temperature) <= 0.0:
            raise ValueError("temperature must be > 0")
        if float(self.spec.lambda_consistency) < 0.0:
            raise ValueError("lambda_consistency must be >= 0")
        _dropnode(torch.zeros((1, 1)), drop_probability=self.spec.dropnode, training=False)
        for name, value in (
            ("mlp_dropout", self.spec.mlp_dropout),
            ("input_dropout", self.spec.input_dropout),
            ("hidden_dropout", self.spec.hidden_dropout),
        ):
            if value is not None and not (0.0 <= float(value) < 1.0):
                raise ValueError(f"{name} must be in [0, 1)")

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> GRANDMethod:
        if self.spec.training_mode not in _TRAINING_MODES:
            raise ValueError(f"training_mode must be one of {sorted(_TRAINING_MODES)!r}")
        if self.spec.training_mode == "legacy":
            return self._fit_standardized(data, device=device, seed=seed)
        return self._fit_random_propagation_consistency(data, device=device, seed=seed)

    def _fit_standardized(
        self,
        data: Any,
        *,
        device: str | None = None,
        seed: int = 0,
    ) -> GRANDMethod:
        """Run the historical ModSSC GRAND-style standardized baseline."""

        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        optional_import("torch", extra="transductive-torch")

        self._device = normalize_device_name(device)
        prep = prepare_data_cached(
            data,
            device=self._device,
            add_self_loops=self.spec.add_self_loops,
            norm_mode="rw",
            cache=self._prep_cache,
        )
        val_count = int(prep.val_mask.sum()) if prep.val_mask is not None else None
        logger.info(
            "GRAND sizes: n_nodes=%s n_classes=%s train=%s val=%s",
            prep.n_nodes,
            prep.n_classes,
            int(prep.train_mask.sum()),
            val_count if val_count is not None else "none",
        )
        self._n_nodes = prep.n_nodes
        self._edge_index = prep.edge_index
        self._edge_weight = prep.edge_weight

        model = _MLP(
            prep.X.shape[1],
            self.spec.hidden_dim,
            prep.n_classes,
            dropout=self.spec.mlp_dropout,
        ).to(torch.device(self._device))
        self._model = model
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.spec.lr,
            weight_decay=self.spec.weight_decay,
        )

        best_state: dict[str, Any] | None = None
        best_val = float("inf")
        bad_epochs = 0
        torch.manual_seed(int(seed))

        for epoch in range(int(self.spec.max_epochs)):
            model.train()
            probabilities = []
            supervised_loss: Any = 0.0
            for _ in range(int(self.spec.num_samples)):
                x_aug = torch.nn.functional.dropout(
                    prep.X,
                    p=self.spec.dropnode,
                    training=True,
                )
                x_prop = _propagate(
                    x_aug,
                    prep.edge_index,
                    prep.edge_weight,
                    n_nodes=prep.n_nodes,
                    steps=self.spec.prop_steps,
                )
                logits = model(x_prop)
                supervised_loss = supervised_loss + torch.nn.functional.cross_entropy(
                    logits[prep.train_mask],
                    prep.y[prep.train_mask],
                )
                probabilities.append(torch.softmax(logits, dim=1))

            supervised_loss = supervised_loss / float(self.spec.num_samples)
            stacked = torch.stack(probabilities, dim=0)
            center = stacked.mean(dim=0).clamp_min(1e-12)
            consistency = (
                (
                    stacked.clamp_min(1e-12)
                    * (torch.log(stacked.clamp_min(1e-12)) - torch.log(center))
                )
                .sum(dim=2)
                .mean()
            )
            loss = supervised_loss + self.spec.lambda_consistency * consistency
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if prep.val_mask is not None and prep.val_mask.any():
                model.eval()
                with torch.no_grad():
                    x_prop = _propagate(
                        prep.X,
                        prep.edge_index,
                        prep.edge_weight,
                        n_nodes=prep.n_nodes,
                        steps=self.spec.prop_steps,
                    )
                    logits = model(x_prop)
                    val_loss = torch.nn.functional.cross_entropy(
                        logits[prep.val_mask],
                        prep.y[prep.val_mask],
                    ).item()
                if val_loss < best_val - 1e-9:
                    best_val = val_loss
                    best_state = {
                        key: value.detach().clone() for key, value in model.state_dict().items()
                    }
                    bad_epochs = 0
                    logger.debug("GRAND epoch=%s val_loss=%.4f best updated", epoch, val_loss)
                else:
                    bad_epochs += 1
                    logger.debug(
                        "GRAND epoch=%s val_loss=%.4f bad_epochs=%s/%s",
                        epoch,
                        val_loss,
                        bad_epochs,
                        self.spec.patience,
                    )
                    if bad_epochs >= int(self.spec.patience):
                        logger.debug(
                            "GRAND early_stop epoch=%s best_val=%.4f",
                            epoch,
                            best_val,
                        )
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.diagnostics_ = {}
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def _fit_random_propagation_consistency(
        self,
        data: Any,
        *,
        device: str | None = None,
        seed: int = 0,
    ) -> GRANDMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        optional_import("torch", extra="transductive-torch")
        self._validate_spec()
        set_torch_seed(int(seed))

        self._device = normalize_device_name(device)
        prep = prepare_data_cached(
            data,
            device=self._device,
            add_self_loops=self.spec.add_self_loops,
            norm_mode="sym",
            cache=self._prep_cache,
        )
        val_count = int(prep.val_mask.sum()) if prep.val_mask is not None else None
        logger.info(
            "GRAND sizes: n_nodes=%s n_classes=%s train=%s val=%s",
            prep.n_nodes,
            prep.n_classes,
            int(prep.train_mask.sum()),
            val_count if val_count is not None else "none",
        )
        self._n_nodes = prep.n_nodes
        self._edge_index = prep.edge_index
        self._edge_weight = prep.edge_weight

        input_dropout = (
            self.spec.mlp_dropout
            if self.spec.input_dropout is None
            else float(self.spec.input_dropout)
        )
        hidden_dropout = (
            self.spec.mlp_dropout
            if self.spec.hidden_dropout is None
            else float(self.spec.hidden_dropout)
        )
        # The pinned implementation initializes its custom MLP on CPU before
        # moving it to CUDA. Build in the same order and reset the generators
        # inside ``_initialize_mlp`` to discard ``nn.Linear``'s own defaults.
        model = _MLP(
            prep.X.shape[1],
            self.spec.hidden_dim,
            prep.n_classes,
            dropout=self.spec.mlp_dropout,
            input_dropout=input_dropout,
            hidden_dropout=hidden_dropout,
            batch_norm=self.spec.use_batch_norm,
        )
        _initialize_mlp(model, seed=int(seed))
        model = model.to(torch.device(self._device))
        self._model = model

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.spec.lr, weight_decay=self.spec.weight_decay
        )

        best_state: dict[str, Any] | None = None
        best_val_loss = float("inf")
        best_val_accuracy = float("-inf")
        running_min_loss = float("inf")
        running_max_accuracy = float("-inf")
        bad_epochs = 0
        best_epoch: int | None = None
        epochs_completed = 0
        stopped_early = False
        last_supervised_loss = float("nan")
        last_consistency_loss = float("nan")

        for epoch in range(int(self.spec.max_epochs)):
            model.train()
            optimizer.zero_grad()

            propagated_samples: list[Any] = []
            for _ in range(int(self.spec.num_samples)):
                x_aug = _dropnode(
                    prep.X,
                    drop_probability=self.spec.dropnode,
                    training=True,
                )
                x_prop = _mixed_order_propagate(
                    x_aug,
                    prep.edge_index,
                    prep.edge_weight,
                    n_nodes=prep.n_nodes,
                    steps=self.spec.prop_steps,
                )
                propagated_samples.append(x_prop)

            # The official loop samples and propagates every view before any
            # MLP forward pass. This order also preserves exact CPU behavior
            # when a parity fixture runs without CUDA.
            logits_samples = [model(x_prop) for x_prop in propagated_samples]
            consistency_weight = float(self.spec.lambda_consistency) * _sigmoid_rampup(
                epoch,
                int(self.spec.consistency_rampup_epochs),
            )
            supervised_loss, consistency_loss, loss = _grand_objective(
                logits_samples,
                prep.y,
                prep.train_mask,
                temperature=float(self.spec.temperature),
                consistency_weight=consistency_weight,
            )

            loss.backward()
            optimizer.step()
            epochs_completed = epoch + 1
            last_supervised_loss = float(supervised_loss.detach().item())
            last_consistency_loss = float(consistency_loss.detach().item())

            if prep.val_mask is None or not bool(prep.val_mask.any()):
                continue

            model.eval()
            with torch.no_grad():
                x_eval = _dropnode(
                    prep.X,
                    drop_probability=self.spec.dropnode,
                    training=False,
                )
                x_prop = _mixed_order_propagate(
                    x_eval,
                    prep.edge_index,
                    prep.edge_weight,
                    n_nodes=prep.n_nodes,
                    steps=self.spec.prop_steps,
                )
                val_log_probabilities = torch.log_softmax(model(x_prop), dim=-1)
                val_loss = float(
                    torch.nn.functional.nll_loss(
                        val_log_probabilities[prep.val_mask],
                        prep.y[prep.val_mask],
                    ).item()
                )
                val_predictions = val_log_probabilities[prep.val_mask].argmax(dim=1)
                val_correct = (val_predictions == prep.y[prep.val_mask]).to(torch.float64)
                val_accuracy = float((val_correct.sum() / len(val_correct)).item())

            checkpoint_update = _official_checkpoint_step(
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                running_min_loss=running_min_loss,
                running_max_accuracy=running_max_accuracy,
                best_val_loss=best_val_loss,
                bad_epochs=bad_epochs,
            )
            running_min_loss = checkpoint_update.running_min_loss
            running_max_accuracy = checkpoint_update.running_max_accuracy
            bad_epochs = checkpoint_update.bad_epochs

            # In the reference code this save is nested inside the rule that
            # resets patience on either a loss or accuracy envelope update.
            if checkpoint_update.save_checkpoint:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                best_state = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }

            if bad_epochs >= int(self.spec.patience):
                stopped_early = True
                logger.debug(
                    "GRAND early_stop epoch=%s best_val_loss=%.4f",
                    epoch,
                    best_val_loss,
                )
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        self.diagnostics_ = {
            # Names the implemented protocol without asserting empirical paper match.
            "algorithm": "grand_feng2020_protocol",
            "training_mode": self.spec.training_mode,
            "epochs_completed": int(epochs_completed),
            "best_epoch": best_epoch,
            "best_val_loss": None if best_epoch is None else float(best_val_loss),
            "best_val_accuracy": None if best_epoch is None else float(best_val_accuracy),
            "stopped_early": bool(stopped_early),
            "last_supervised_loss": float(last_supervised_loss),
            "last_consistency_loss": float(last_consistency_loss),
            "propagation": "mixed_order_mean",
            "perturbation": "dropnode",
            "dropnode_scaling": "official_train_mask_eval_keep_probability",
            "dropnode_rng": "official_cpu_bernoulli",
            "initialization": "official_mlplayer_normal",
            "checkpoint_policy": "official_loss_checkpoint_patience_loss_or_accuracy",
            "model_seed": int(seed),
            "temperature": float(self.spec.temperature),
        }
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, data: Any) -> np.ndarray:
        if self.spec.training_mode == "legacy":
            return self._predict_proba_standardized(data)
        return self._predict_proba_random_propagation_consistency(data)

    def _predict_proba_standardized(self, data: Any) -> np.ndarray:
        if (
            self._model is None
            or self._edge_index is None
            or self._edge_weight is None
            or self._n_nodes is None
        ):
            raise RuntimeError("GRANDMethod is not fitted yet. Call fit() first.")

        prep = prepare_data_cached(
            data,
            device=self._device or "cpu",
            add_self_loops=self.spec.add_self_loops,
            norm_mode="rw",
            cache=self._prep_cache,
        )
        if prep.n_nodes != self._n_nodes:
            raise ValueError(f"GRAND was fitted on n={self._n_nodes} nodes, got n={prep.n_nodes}")
        self._model.eval()
        with torch.no_grad():
            x_prop = _propagate(
                prep.X,
                prep.edge_index,
                prep.edge_weight,
                n_nodes=prep.n_nodes,
                steps=self.spec.prop_steps,
            )
            logits = self._model(x_prop)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.detach().cpu().numpy()

    def _predict_proba_random_propagation_consistency(self, data: Any) -> np.ndarray:
        if (
            self._model is None
            or self._edge_index is None
            or self._edge_weight is None
            or self._n_nodes is None
        ):
            raise RuntimeError("GRANDMethod is not fitted yet. Call fit() first.")

        prep = prepare_data_cached(
            data,
            device=self._device or "cpu",
            add_self_loops=self.spec.add_self_loops,
            norm_mode="sym",
            cache=self._prep_cache,
        )
        if prep.n_nodes != self._n_nodes:
            raise ValueError(f"GRAND was fitted on n={self._n_nodes} nodes, got n={prep.n_nodes}")

        self._model.eval()
        with torch.no_grad():
            x_eval = _dropnode(
                prep.X,
                drop_probability=self.spec.dropnode,
                training=False,
            )
            x_prop = _mixed_order_propagate(
                x_eval,
                prep.edge_index,
                prep.edge_weight,
                n_nodes=prep.n_nodes,
                steps=self.spec.prop_steps,
            )
            logits = self._model(x_prop)
            proba = torch.softmax(logits, dim=1)
        return proba.detach().cpu().numpy()
