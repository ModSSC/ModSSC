"""Inductive SSL quickstart on the built-in 'toy' dataset.

Goal
- Show a scikit-learn-like "fit/predict" workflow with an inductive SSL method.
- No external datasets, no torch, no sklearn required.

Expected
- Runs after: pip install modssc
"""

from __future__ import annotations

import numpy as np

from modssc.data_loader import load_dataset
from modssc.evaluation import accuracy
from modssc.inductive import DeviceSpec, InductiveDataset, get_method_class
from modssc.inductive.methods.pseudo_label import PseudoLabelSpec


def main() -> None:
    # 1) Load deterministic dataset
    ds = load_dataset("toy", download=True)
    X_train = np.asarray(ds.train.X)
    y_train = np.asarray(ds.train.y)

    X_test = np.asarray(ds.test.X) if ds.test is not None else None
    y_test = np.asarray(ds.test.y) if ds.test is not None else None

    # 2) Create a small labeled set (per class) + unlabeled pool
    rng = np.random.default_rng(0)
    labeled_mask = np.zeros((X_train.shape[0],), dtype=bool)
    for c in np.unique(y_train):
        idx = np.flatnonzero(y_train == c)
        labeled_mask[rng.choice(idx, size=3, replace=False)] = True

    X_l = X_train[labeled_mask]
    y_l = y_train[labeled_mask]
    X_u = X_train[~labeled_mask]

    # 3) Configure and instantiate a classic pseudo-labeling method
    spec = PseudoLabelSpec(
        classifier_id="knn",
        classifier_backend="numpy",
        classifier_params={"k": 5},
        confidence_threshold=0.95,
        max_iter=10,
    )
    method_cls = get_method_class("pseudo_label")
    method = method_cls(spec=spec)

    # 4) Fit
    data = InductiveDataset(X_l=X_l, y_l=y_l, X_u=X_u)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    # 5) Evaluate
    pred_train = method.predict(X_train)
    print(f"train accuracy: {accuracy(y_train, pred_train):.4f} (n_train={X_train.shape[0]})")

    if X_test is not None and y_test is not None and X_test.size:
        pred_test = method.predict(X_test)
        print(f"test accuracy:  {accuracy(y_test, pred_test):.4f} (n_test={X_test.shape[0]})")
    else:
        print("no official test split in this dataset")


if __name__ == "__main__":
    main()
