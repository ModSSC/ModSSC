from __future__ import annotations

import math

import numpy as np


def alpha(epoch: int) -> float:
    if epoch < 100:
        return 0.0
    if epoch < 600:
        return 3.0 * float(epoch - 100) / 500.0
    return 3.0


def learning_rate(epoch: int) -> float:
    return 1.5 * 0.998**epoch


def momentum(epoch: int) -> float:
    if epoch >= 500:
        return 0.99
    return (float(epoch) / 500.0) * 0.99 + (1.0 - float(epoch) / 500.0) * 0.5


def _binary_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    losses = np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    return float(losses.sum(axis=1).mean())


def joint_loss(
    logits_l: np.ndarray,
    y_l: np.ndarray,
    logits_u: np.ndarray,
    coefficient: float,
) -> tuple[float, float, float, np.ndarray]:
    n_classes = int(logits_l.shape[1])
    labeled_targets = np.eye(n_classes, dtype=np.float64)[y_l]
    pseudo = np.argmax(1.0 / (1.0 + np.exp(-logits_u)), axis=1)
    unlabeled_targets = np.eye(n_classes, dtype=np.float64)[pseudo]
    supervised = _binary_cross_entropy(logits_l, labeled_targets)
    unsupervised = _binary_cross_entropy(logits_u, unlabeled_targets)
    return supervised + coefficient * unsupervised, supervised, unsupervised, pseudo


def sgd_trajectory(
    initial_parameter: float,
    gradients: list[float],
    *,
    rate: float,
    inertia: float,
) -> list[float]:
    parameter = float(initial_parameter)
    update = 0.0
    trajectory: list[float] = []
    for gradient in gradients:
        update = inertia * update - (1.0 - inertia) * rate * float(gradient)
        parameter += update
        trajectory.append(parameter)
    return trajectory


def dropout_eval(values: np.ndarray, probability: float) -> np.ndarray:
    return np.asarray(values) * (1.0 - float(probability))


def dropout_train(
    values: np.ndarray,
    probability: float,
    uniform_draws: np.ndarray,
) -> np.ndarray:
    values_array = np.asarray(values)
    draws_array = np.asarray(uniform_draws)
    if values_array.shape != draws_array.shape:
        raise ValueError("values and uniform_draws must have the same shape")
    return values_array * (draws_array >= float(probability))


def assert_finite(values: list[float]) -> None:
    if not all(math.isfinite(value) for value in values):
        raise AssertionError("oracle produced a non-finite value")
