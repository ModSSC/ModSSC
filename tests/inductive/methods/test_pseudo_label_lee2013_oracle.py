from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from modssc.inductive.methods.pseudo_label import (
    _build_lee2013_mlp,
    _lee2013_alpha,
    _lee2013_joint_loss,
    _lee2013_learning_rate,
    _lee2013_momentum,
    _lee2013_sgd_step,
)
from tests.inductive.methods import _pseudo_label_lee2013_oracle as oracle


def test_lee2013_independent_oracle_matches_published_schedules() -> None:
    epochs = [0, 99, 100, 250, 350, 499, 500, 599, 600, 700]
    expected_alpha = [oracle.alpha(epoch) for epoch in epochs]
    expected_rates = [oracle.learning_rate(epoch) for epoch in epochs]
    expected_momentum = [oracle.momentum(epoch) for epoch in epochs]
    oracle.assert_finite(expected_alpha + expected_rates + expected_momentum)

    assert [_lee2013_alpha(epoch) for epoch in epochs] == pytest.approx(expected_alpha)
    assert [_lee2013_learning_rate(epoch) for epoch in epochs] == pytest.approx(expected_rates)
    assert [_lee2013_momentum(epoch) for epoch in epochs] == pytest.approx(expected_momentum)


def test_lee2013_independent_oracle_matches_joint_loss_and_sgd_transition() -> None:
    logits_l_np = np.array([[0.2, -0.3], [-0.7, 1.1]], dtype=np.float64)
    labels_np = np.array([0, 1], dtype=np.int64)
    logits_u_np = np.array([[-1.0, 0.4], [2.0, -0.2]], dtype=np.float64)
    expected_total, expected_l, expected_u, expected_pseudo = oracle.joint_loss(
        logits_l_np,
        labels_np,
        logits_u_np,
        0.75,
    )

    total, supervised, unsupervised, pseudo = _lee2013_joint_loss(
        torch.as_tensor(logits_l_np),
        torch.as_tensor(labels_np),
        torch.as_tensor(logits_u_np),
        alpha=0.75,
        n_classes=2,
    )
    assert float(total.item()) == pytest.approx(expected_total)
    assert float(supervised.item()) == pytest.approx(expected_l)
    assert float(unsupervised.item()) == pytest.approx(expected_u)
    np.testing.assert_array_equal(pseudo.numpy(), expected_pseudo)

    expected_trajectory = oracle.sgd_trajectory(
        1.0,
        [2.0, 1.0],
        rate=1.5,
        inertia=0.5,
    )
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    buffers: dict[int, torch.Tensor] = {}
    observed: list[float] = []
    for gradient in [2.0, 1.0]:
        parameter.grad = torch.tensor([gradient])
        _lee2013_sgd_step(
            [parameter],
            buffers,
            learning_rate=1.5,
            momentum=0.5,
        )
        observed.append(float(parameter.item()))
    assert observed == pytest.approx(expected_trajectory)


def test_lee2013_independent_oracle_matches_equation_9_dropout() -> None:
    model = _build_lee2013_mlp(
        torch=torch,
        input_dim=4,
        hidden_units=3,
        n_classes=2,
        hidden_dropout=0.25,
        input_dropout=0.0,
    )
    values = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    model.hidden_dropout.eval()
    expected_eval = oracle.dropout_eval(values.numpy(), 0.25)
    np.testing.assert_allclose(model.hidden_dropout(values).numpy(), expected_eval)

    model.hidden_dropout.train()
    draws = torch.tensor(
        [
            [0.10, 0.25, 0.80],
            [0.24, 0.26, 0.99],
        ]
    )
    expected_train = oracle.dropout_train(values.numpy(), 0.25, draws.numpy())
    with patch.object(torch, "rand_like", return_value=draws):
        observed_train = model.hidden_dropout(values)
    np.testing.assert_array_equal(observed_train.numpy(), expected_train)
