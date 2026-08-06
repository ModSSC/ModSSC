from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from modssc.transductive.methods.gnn.common import TwoLayerMLP  # noqa: E402
from modssc.transductive.methods.gnn.grand import (  # noqa: E402
    GRANDMethod,
    GRANDSpec,
    _consistency_loss,
    _dropnode,
    _grand_objective,
    _initialize_mlp,
    _mixed_order_propagate,
    _sharpen,
    _sigmoid_rampup,
)


def test_grand_training_mode_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="training_mode must be one of"):
        GRANDMethod(GRANDSpec(training_mode="unknown")).fit(object())


def test_mixed_order_propagation_averages_every_power() -> None:
    features = torch.tensor([[1.0], [3.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.ones(2)

    propagated = _mixed_order_propagate(
        features,
        edge_index,
        edge_weight,
        n_nodes=2,
        steps=1,
    )

    torch.testing.assert_close(propagated, torch.tensor([[2.0], [2.0]]))
    torch.testing.assert_close(
        _mixed_order_propagate(
            features,
            edge_index,
            edge_weight,
            n_nodes=2,
            steps=0,
        ),
        features,
    )


def test_dropnode_matches_official_training_and_inference_scaling(monkeypatch) -> None:
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([1.0, 0.0])
    monkeypatch.setattr(torch, "bernoulli", lambda *_args, **_kwargs: mask)

    augmented = _dropnode(features, drop_probability=0.5, training=True)

    torch.testing.assert_close(augmented, torch.tensor([[1.0, 2.0], [0.0, 0.0]]))
    torch.testing.assert_close(
        _dropnode(features, drop_probability=0.5, training=False),
        features * 0.5,
    )


def test_dropnode_zero_probability_is_identity_without_random_draw(monkeypatch) -> None:
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("torch.bernoulli must not be called when DropNode is disabled")

    monkeypatch.setattr(torch, "bernoulli", fail_if_called)

    assert _dropnode(features, drop_probability=0.0, training=True) is features
    assert _dropnode(features, drop_probability=0.0, training=False) is features


def test_predict_proba_applies_official_dropnode_inference_scaling(monkeypatch) -> None:
    import modssc.transductive.methods.gnn.grand as grand_module

    features = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    prep = SimpleNamespace(
        X=features,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weight=torch.empty((0,)),
        n_nodes=2,
    )
    monkeypatch.setattr(grand_module, "prepare_data_cached", lambda *_args, **_kwargs: prep)

    class IdentityModel:
        def eval(self) -> None:
            return None

        def __call__(self, values):
            return values

    method = GRANDMethod(
        GRANDSpec(
            dropnode=0.5,
            prop_steps=0,
            training_mode="random_propagation_consistency",
        )
    )
    method._model = IdentityModel()
    method._edge_index = prep.edge_index
    method._edge_weight = prep.edge_weight
    method._n_nodes = prep.n_nodes

    probabilities = torch.from_numpy(method.predict_proba(object()))

    torch.testing.assert_close(probabilities, torch.softmax(features * 0.5, dim=1))


def test_sharpen_and_consistency_match_grand_equations() -> None:
    probabilities = torch.tensor([[0.25, 0.75]])
    sharpened = _sharpen(probabilities, temperature=0.5)
    torch.testing.assert_close(sharpened, torch.tensor([[0.1, 0.9]]))

    log_probabilities = [torch.log(probabilities), torch.log(probabilities)]
    assert float(_consistency_loss(log_probabilities, temperature=1.0)) == pytest.approx(
        0.0, abs=1e-12
    )

    logits_1 = torch.log(torch.tensor([[0.8, 0.2], [0.4, 0.6]]))
    logits_2 = torch.log(torch.tensor([[0.6, 0.4], [0.2, 0.8]]))
    actual = _consistency_loss([logits_1, logits_2], temperature=1.0)
    center = (torch.exp(logits_1) + torch.exp(logits_2)) / 2.0
    expected = torch.stack(
        [
            (torch.exp(logits_1) - center).pow(2).sum(dim=1).mean(),
            (torch.exp(logits_2) - center).pow(2).sum(dim=1).mean(),
        ]
    ).mean()
    torch.testing.assert_close(actual, expected)


def test_optional_consistency_ramp_is_bounded_and_finishes_at_one() -> None:
    assert _sigmoid_rampup(0, 0) == 1.0
    assert 0.0 < _sigmoid_rampup(0, 10) < _sigmoid_rampup(5, 10) < 1.0
    assert _sigmoid_rampup(10, 10) == 1.0
    assert _sigmoid_rampup(50, 10) == 1.0


@pytest.mark.parametrize("probability", [-0.1, 1.0])
def test_dropnode_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="drop_probability"):
        _dropnode(torch.ones((2, 2)), drop_probability=probability, training=True)


def test_mixed_order_propagation_rejects_negative_steps() -> None:
    with pytest.raises(ValueError, match="steps"):
        _mixed_order_propagate(
            torch.ones((1, 1)),
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,)),
            n_nodes=1,
            steps=-1,
        )


def test_grand_equation_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="temperature"):
        _sharpen(torch.ones((1, 2)), temperature=0.0)
    with pytest.raises(ValueError, match="log_probabilities"):
        _consistency_loss([], temperature=1.0)
    with pytest.raises(ValueError, match="logits"):
        _grand_objective(
            [],
            torch.zeros(1, dtype=torch.long),
            torch.ones(1, dtype=torch.bool),
            temperature=1.0,
            consistency_weight=1.0,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"prop_steps": -1}, "prop_steps"),
        ({"num_samples": 0}, "num_samples"),
        ({"max_epochs": 0}, "max_epochs"),
        ({"patience": 0}, "patience"),
        ({"consistency_rampup_epochs": -1}, "rampup"),
        ({"temperature": 0.0}, "temperature"),
        ({"lambda_consistency": -1.0}, "lambda_consistency"),
        ({"mlp_dropout": 1.0}, "mlp_dropout"),
        ({"input_dropout": -0.1}, "input_dropout"),
        ({"hidden_dropout": 1.0}, "hidden_dropout"),
    ],
)
def test_grand_spec_validation_rejects_invalid_values(overrides, message) -> None:
    method = GRANDMethod(GRANDSpec(**overrides))

    with pytest.raises(ValueError, match=message):
        method._validate_spec()


def test_grand_initialization_and_batch_normalized_mlp() -> None:
    model = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(3, 2, bias=False))
    _initialize_mlp(model)
    mlp = TwoLayerMLP(3, 4, 2, dropout=0.0, batch_norm=True)
    mlp.eval()

    output = mlp(torch.ones((2, 3)))

    assert output.shape == (2, 2)


def test_grand_stops_when_validation_loss_and_accuracy_both_worsen(monkeypatch) -> None:
    import modssc.transductive.methods.gnn.grand as grand_module

    prep = SimpleNamespace(
        X=torch.zeros((4, 2)),
        y=torch.tensor([0, 1, 0, 1]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weight=torch.empty((0,)),
        train_mask=torch.tensor([True, True, False, False]),
        val_mask=torch.tensor([False, False, True, True]),
        n_nodes=4,
        n_classes=2,
    )
    monkeypatch.setattr(grand_module, "prepare_data_cached", lambda *_args, **_kwargs: prep)

    class _DeterministicMLP(torch.nn.Module):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))
            self.validation_calls = 0

        def forward(self, values):
            if self.training:
                return torch.zeros((values.shape[0], 2), device=values.device) + self.anchor
            self.validation_calls += 1
            logits = (
                torch.tensor([[0.0, 0.0], [0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
                if self.validation_calls == 1
                else torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 5.0], [5.0, 0.0]])
            )
            return logits.to(values.device) + self.anchor

    monkeypatch.setattr(grand_module, "_MLP", _DeterministicMLP)
    method = GRANDMethod(
        GRANDSpec(
            hidden_dim=2,
            prop_steps=0,
            training_mode="random_propagation_consistency",
            dropnode=0.0,
            num_samples=1,
            max_epochs=5,
            patience=1,
        )
    )

    method.fit(object(), device="cpu", seed=0)

    assert method.diagnostics_["epochs_completed"] == 2
    assert method.diagnostics_["best_epoch"] == 0
    assert method.diagnostics_["stopped_early"] is True
    assert method.diagnostics_["model_seed"] == 0


def test_grand_without_validation_finishes_without_checkpoint(monkeypatch) -> None:
    import modssc.transductive.methods.gnn.grand as grand_module

    prep = SimpleNamespace(
        X=torch.zeros((4, 2)),
        y=torch.tensor([0, 1, 0, 1]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weight=torch.empty((0,)),
        train_mask=torch.tensor([True, True, False, False]),
        val_mask=torch.zeros(4, dtype=torch.bool),
        n_nodes=4,
        n_classes=2,
    )
    monkeypatch.setattr(grand_module, "prepare_data_cached", lambda *_args, **_kwargs: prep)
    method = GRANDMethod(
        GRANDSpec(
            hidden_dim=2,
            prop_steps=0,
            training_mode="random_propagation_consistency",
            dropnode=0.0,
            num_samples=1,
            max_epochs=1,
            patience=1,
        )
    )

    method.fit(object(), device="cpu", seed=0)

    assert method.diagnostics_["epochs_completed"] == 1
    assert method.diagnostics_["best_epoch"] is None
    assert method.diagnostics_["best_val_loss"] is None
    assert method.diagnostics_["stopped_early"] is False


@pytest.mark.parametrize("training_mode", ["legacy", "random_propagation_consistency"])
def test_grand_predict_proba_rejects_unfitted_and_mismatched_graphs(
    monkeypatch, training_mode: str
) -> None:
    import modssc.transductive.methods.gnn.grand as grand_module

    method = GRANDMethod(GRANDSpec(training_mode=training_mode))
    with pytest.raises(RuntimeError, match="not fitted"):
        method.predict_proba(object())

    prep = SimpleNamespace(n_nodes=3)
    monkeypatch.setattr(grand_module, "prepare_data_cached", lambda *_args, **_kwargs: prep)
    method._model = torch.nn.Identity()
    method._edge_index = torch.empty((2, 0), dtype=torch.long)
    method._edge_weight = torch.empty((0,))
    method._n_nodes = 2
    method._device = "cpu"

    with pytest.raises(ValueError, match="fitted on n=2 nodes, got n=3"):
        method.predict_proba(object())
