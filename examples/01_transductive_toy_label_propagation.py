"""Transductive SSL quickstart on a synthetic graph.

Goal
- Show a scikit-learn-like workflow with a transductive method (label propagation).
- No external datasets, no torch required (NumPy backend).

Expected
- Runs after: pip install modssc
"""

from __future__ import annotations

import numpy as np

from modssc.evaluation import accuracy
from modssc.graph import GraphBuilderSpec, build_graph
from modssc.graph.artifacts import NodeDataset
from modssc.transductive import get_method_class
from modssc.transductive.methods.classic.label_propagation import LabelPropagationSpec


def main() -> None:
    rng = np.random.default_rng(0)
    n_per = 100
    d = 16

    X = np.vstack(
        [
            rng.normal(loc=-2.0, scale=0.8, size=(n_per, d)),
            rng.normal(loc=0.0, scale=0.8, size=(n_per, d)),
            rng.normal(loc=2.0, scale=0.8, size=(n_per, d)),
        ]
    ).astype(np.float32)
    y_true = np.array([0] * n_per + [1] * n_per + [2] * n_per, dtype=np.int64)

    # Build a k-NN graph on the feature vectors
    gspec = GraphBuilderSpec(
        scheme="knn",
        metric="cosine",
        k=15,
        symmetrize="mutual",
        self_loops=True,
    )
    graph = build_graph(X, spec=gspec, seed=0, cache=False)

    # Reveal only a few labels (transductive setting)
    train_mask = np.zeros((X.shape[0],), dtype=bool)
    for c in np.unique(y_true):
        idx = np.flatnonzero(y_true == c)
        train_mask[rng.choice(idx, size=5, replace=False)] = True

    y_obs = y_true.copy()
    y_obs[~train_mask] = -1  # unlabeled marker

    data = NodeDataset(X=X, y=y_obs, graph=graph, masks={"train_mask": train_mask})

    # Fit label propagation (hard clamp)
    method_cls = get_method_class("label_propagation")
    method = method_cls(spec=LabelPropagationSpec(max_iter=200, tol=1e-6, norm="rw"))

    method.fit(data)
    proba = method.predict_proba(data)
    pred = proba.argmax(axis=1)

    print(f"accuracy (all):       {accuracy(y_true, pred):.4f}")
    print(f"accuracy (unlabeled): {accuracy(y_true[~train_mask], pred[~train_mask]):.4f}")


if __name__ == "__main__":
    main()
