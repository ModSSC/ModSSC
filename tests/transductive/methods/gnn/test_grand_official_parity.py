from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from modssc.transductive.methods.gnn.common import TwoLayerMLP  # noqa: E402
from modssc.transductive.methods.gnn.grand import (  # noqa: E402
    GRANDMethod,
    _consistency_loss,
    _dropnode,
    _grand_objective,
    _initialize_mlp,
    _mixed_order_propagate,
    _official_checkpoint_step,
    _sharpen,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "grand_official_7a2fd6e7.json"
OFFICIAL_COMMIT = "7a2fd6e7c3f20ca2c84b06ec1c5dc7f227dbfe2b"


@pytest.fixture(scope="module")
def official_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _sparse_edges(adjacency: Any) -> tuple[Any, Any]:
    rows, columns = torch.nonzero(adjacency, as_tuple=True)
    # ModSSC stores A[dst, src] as edge_index=(src, dst).
    return torch.stack((columns, rows)), adjacency[rows, columns]


def test_frozen_oracle_pins_the_reviewed_official_sources(
    official_fixture: dict[str, Any],
) -> None:
    provenance = official_fixture["provenance"]

    assert provenance["repository"] == "https://github.com/THUDM/GRAND"
    assert provenance["commit"] == OFFICIAL_COMMIT
    assert provenance["paper_sha256"] == (
        "b37c0c72d463d8ccd973d59fb85bdf23ab2a18e13bde7d5cde5916564272e7b2"
    )
    assert provenance["supplemental_sha256"] == (
        "aa4c625aa3a074e3440386bf303efe2c3cb85f5eab6b1ecbbf40e793be48a7d2"
    )
    assert GRANDMethod.info.official_code == (
        f"https://github.com/THUDM/GRAND/tree/{OFFICIAL_COMMIT}"
    )
    assert (
        GRANDMethod.info.paper_title
        == "Graph Random Neural Networks for Semi-Supervised Learning on Graphs"
    )
    assert provenance["source_sha256"] == {
        "train_grand.py": "6c6e3162937fcb382172569af7d9ddfa71c677a3a406c17ace7cd3d7a4978443",
        "pygcn/models.py": "970837c2bd448bf21bb6085ff8604d73bd8f7098260696f86fd1f1acd6082cba",
        "pygcn/layers.py": "a2715fd232e3449d76a072b88496938e2182aade355bce8d363f2056476567c4",
        "pygcn/utils.py": "a6f272c9ddcccca29a7c788b0b9c36c0b83c57b137d700f02e765d810c4a90d6",
        "run100_cora.sh": "cc94cfc7eb6194c6a6cb4a2438a89a72b518163eee10c1514542299420dc684a",
    }
    assert provenance["cora_raw_sha256"] == {
        "ind.cora.allx": "9419ba2f26f5c35243db64aba110e0f35a04851609e0dc5433450676ca6b8543",
        "ind.cora.ally": "2b998f5cc7fedc86e7f97ca2498f47a1ffc1462c29e2d578b23bf3f62b6e7d71",
        "ind.cora.graph": "58f13302f39dde8852dad6fe6d15b89b077b6d7f837626bce671ef80344b383d",
        "ind.cora.test.index": ("297ce89af2b51a6a194181c7dbe1796c6a1cf9cd9349a88383b1b1e227867875"),
        "ind.cora.tx": "b9afbaa4a400df991f6f02ef677e1e44da55ffc04801fd02f9e673987829226a",
        "ind.cora.ty": "41f5ac76596a1699cc33f53084a2419e92961c42bb75f36fc38e616d348532ad",
        "ind.cora.x": "23cfa55d91c6f624233f5eb7b6e3f141f1bd2b2ae39608a63cf5d084bb27baab",
        "ind.cora.y": "94465c14eb53e04ca262198dcbb2521ee8af60fb3f3d546cd6ca24a511b0e7d1",
    }


def test_dropnode_and_propagation_match_the_frozen_official_oracle(
    monkeypatch: pytest.MonkeyPatch,
    official_fixture: dict[str, Any],
) -> None:
    dropnode = official_fixture["dropnode"]
    propagation = official_fixture["propagation"]
    features = torch.tensor(dropnode["features"], dtype=torch.float32)
    frozen_mask = torch.tensor(dropnode["bernoulli_mask"], dtype=torch.float32)
    bernoulli_inputs: list[Any] = []

    def frozen_bernoulli(probabilities: Any) -> Any:
        bernoulli_inputs.append(probabilities.detach().clone())
        return frozen_mask

    monkeypatch.setattr(torch, "bernoulli", frozen_bernoulli)
    train_augmented = _dropnode(
        features,
        drop_probability=float(dropnode["drop_probability"]),
        training=True,
    )
    inference_augmented = _dropnode(
        features,
        drop_probability=float(dropnode["drop_probability"]),
        training=False,
    )

    assert len(bernoulli_inputs) == 1
    assert bernoulli_inputs[0].device.type == "cpu"
    torch.testing.assert_close(bernoulli_inputs[0], torch.full((4,), 0.75))
    torch.testing.assert_close(
        train_augmented,
        torch.tensor(dropnode["training_output"], dtype=torch.float32),
    )
    torch.testing.assert_close(
        inference_augmented,
        torch.tensor(dropnode["inference_output"], dtype=torch.float32),
    )

    adjacency = torch.tensor(propagation["adjacency"], dtype=torch.float32)
    edge_index, edge_weight = _sparse_edges(adjacency)
    propagated_train = _mixed_order_propagate(
        train_augmented.clone().requires_grad_(True),
        edge_index,
        edge_weight,
        n_nodes=4,
        steps=int(propagation["order"]),
    )
    propagated_inference = _mixed_order_propagate(
        inference_augmented,
        edge_index,
        edge_weight,
        n_nodes=4,
        steps=int(propagation["order"]),
    )

    assert propagated_train.requires_grad is False
    torch.testing.assert_close(
        propagated_train,
        torch.tensor(propagation["training_output"], dtype=torch.float32),
    )
    torch.testing.assert_close(
        propagated_inference,
        torch.tensor(propagation["inference_output"], dtype=torch.float32),
    )


def test_sharpening_consistency_and_total_loss_match_the_frozen_official_oracle(
    official_fixture: dict[str, Any],
) -> None:
    objective = official_fixture["objective"]
    logits = [torch.tensor(values, dtype=torch.float32) for values in objective["logits"]]
    log_probabilities = [torch.log_softmax(values, dim=-1) for values in logits]
    mean_probability = sum(values.exp() for values in log_probabilities) / len(log_probabilities)
    expected_mean = torch.tensor(objective["mean_probability"], dtype=torch.float32)
    expected_target = torch.tensor(objective["sharpened_target"], dtype=torch.float32)

    torch.testing.assert_close(mean_probability, expected_mean)
    torch.testing.assert_close(
        _sharpen(mean_probability, temperature=float(objective["temperature"])),
        expected_target,
    )
    consistency = _consistency_loss(
        log_probabilities,
        temperature=float(objective["temperature"]),
    )

    labels = torch.tensor(objective["labels"], dtype=torch.long)
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[torch.tensor(objective["train_indices"], dtype=torch.long)] = True
    supervised, objective_consistency, total = _grand_objective(
        logits,
        labels,
        train_mask,
        temperature=float(objective["temperature"]),
        consistency_weight=float(objective["consistency_weight"]),
    )

    assert float(consistency) == pytest.approx(objective["consistency_loss"], abs=1e-7)
    assert float(supervised) == pytest.approx(objective["supervised_loss"], abs=1e-7)
    assert float(objective_consistency) == pytest.approx(objective["consistency_loss"], abs=1e-7)
    assert float(total) == pytest.approx(objective["total_loss"], abs=1e-7)


def test_checkpoint_policy_matches_the_frozen_official_event_sequence(
    official_fixture: dict[str, Any],
) -> None:
    checkpoint = official_fixture["checkpoint"]
    running_min_loss = float("inf")
    running_max_accuracy = float("-inf")
    best_loss = float("inf")
    best_accuracy = float("-inf")
    best_epoch: int | None = None
    bad_epochs = 0

    for epoch, observation in enumerate(checkpoint["observations"]):
        update = _official_checkpoint_step(
            val_loss=float(observation["loss"]),
            val_accuracy=float(observation["accuracy"]),
            running_min_loss=running_min_loss,
            running_max_accuracy=running_max_accuracy,
            best_val_loss=best_loss,
            bad_epochs=bad_epochs,
        )
        assert (update.bad_epochs == 0) is observation["reset"]
        assert update.save_checkpoint is observation["save"]
        assert (update.bad_epochs == checkpoint["patience"]) is observation["stop"]

        running_min_loss = update.running_min_loss
        running_max_accuracy = update.running_max_accuracy
        bad_epochs = update.bad_epochs
        if update.save_checkpoint:
            best_loss = float(observation["loss"])
            best_accuracy = float(observation["accuracy"])
            best_epoch = epoch

    assert best_epoch == checkpoint["best_epoch"]
    assert best_loss == checkpoint["best_loss"]
    assert best_accuracy == checkpoint["best_accuracy"]


class _OfficialMLPLayer(torch.nn.Module):
    """Independent transcription of ``pygcn/layers.py::MLPLayer``."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((in_features, out_features)))
        self.bias = torch.nn.Parameter(torch.empty(out_features))
        stdv = 1.0 / math.sqrt(out_features)
        self.weight.data.normal_(-stdv, stdv)
        self.bias.data.normal_(-stdv, stdv)

    def forward(self, values: Any) -> Any:
        return torch.mm(values, self.weight) + self.bias


class _OfficialMLP(torch.nn.Module):
    """Dropout-free reference composition used only for initializer parity."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = _OfficialMLPLayer(3, 2)
        self.layer2 = _OfficialMLPLayer(2, 2)

    def forward(self, values: Any) -> Any:
        return self.layer2(torch.relu(self.layer1(values)))


def test_initializer_matches_the_pinned_official_layer_layout_and_rng() -> None:
    candidate = TwoLayerMLP(3, 2, 2, dropout=0.0)
    seed = 1
    torch.manual_seed(seed)
    reference = _OfficialMLP()

    _initialize_mlp(candidate, seed=seed)
    candidate.eval()
    reference.eval()
    values = torch.tensor(
        [
            [-2.0, 1.0, -0.5],
            [1.0, -2.0, -1.0],
            [3.0, 3.0, -4.0],
        ]
    )

    torch.testing.assert_close(candidate.lin1.weight, reference.layer1.weight.T)
    torch.testing.assert_close(candidate.lin1.bias, reference.layer1.bias)
    torch.testing.assert_close(candidate.lin2.weight, reference.layer2.weight.T)
    torch.testing.assert_close(candidate.lin2.bias, reference.layer2.bias)
    torch.testing.assert_close(candidate(values), reference(values))
