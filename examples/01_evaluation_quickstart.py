from __future__ import annotations

import json

import numpy as np

from modssc.evaluation import evaluate

metrics = ["accuracy", "macro_f1", "balanced_accuracy"]

y_true_labels = np.array([0, 1, 2, 1, 0])
y_true_onehot = np.eye(3)[y_true_labels]

scores = np.array(
    [
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.3, 0.5],
        [0.2, 0.6, 0.2],
        [0.6, 0.3, 0.1],
    ]
)
y_pred_labels = np.array([0, 1, 2, 0, 0])

results = {
    "one_hot_scores": evaluate(y_true_onehot, scores, metrics),
    "labels": evaluate(y_true_labels, y_pred_labels, metrics),
}

print(json.dumps(results, indent=2, sort_keys=True))
