"""Inductive SSL with two views (Co-Training) on a synthetic dataset.

Goal
- Demonstrate how inductive methods consume `data.views`.
- Runs with core dependencies (NumPy backend).

Notes
- Co-training expects exactly two views, each view provides (X_l, X_u) for fit
  and (X) or (X_u) for predict.

Expected
- Runs after: pip install modssc
"""

from __future__ import annotations

import numpy as np

from modssc.evaluation import accuracy
from modssc.inductive import DeviceSpec, InductiveDataset, get_method_class
from modssc.inductive.methods.co_training import CoTrainingSpec


def main() -> None:
    rng = np.random.default_rng(0)
    n = 300
    d = 20

    X = rng.normal(size=(n, d)).astype(np.float32)
    # Non linear label rule to make the task non trivial
    y = ((X[:, 0] + 0.5 * X[:, 1] - 0.25 * X[:, 2]) > 0).astype(np.int64)

    # Two feature views: split features in half
    X_a = X[:, : d // 2]
    X_b = X[:, d // 2 :]

    # Small labeled set (per class)
    labeled = np.zeros((n,), dtype=bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        labeled[rng.choice(idx, size=5, replace=False)] = True

    # Training payload with views
    train_data = InductiveDataset(
        X_l=X_a[labeled],  # used only for backend detection
        y_l=y[labeled],
        views={
            "view_a": {"X_l": X_a[labeled], "X_u": X_a[~labeled]},
            "view_b": {"X_l": X_b[labeled], "X_u": X_b[~labeled]},
        },
    )

    spec = CoTrainingSpec(
        classifier_id="knn",
        classifier_backend="numpy",
        classifier_params={"k": 7},
        max_iter=15,
        k_per_class=1,
        confidence_threshold=None,
    )
    method_cls = get_method_class("co_training")
    method = method_cls(spec=spec)
    method.fit(train_data, device=DeviceSpec(device="cpu"), seed=0)

    # Prediction payload needs the two views for X
    pred_data = InductiveDataset(
        X_l=X_a[labeled],
        y_l=y[labeled],
        views={
            "view_a": {"X": X_a},
            "view_b": {"X": X_b},
        },
    )
    pred = method.predict(pred_data)

    print(f"accuracy (all): {accuracy(y, pred):.4f}")
    print(f"labeled count:  {int(labeled.sum())} / {n}")


if __name__ == "__main__":
    main()
