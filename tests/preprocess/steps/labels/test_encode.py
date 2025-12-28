import numpy as np

from modssc.preprocess.steps.labels.encode import EncodeLabelsStep
from modssc.preprocess.store import ArtifactStore


def test_encode_labels_step():
    store = ArtifactStore()
    store.set("raw.y", np.array(["b", "a", "b"]))
    step = EncodeLabelsStep()
    rng = np.random.default_rng(0)

    res = step.transform(store, rng=rng)

    assert res["labels.y"].tolist() == [1, 0, 1]
    assert res["labels.classes"].tolist() == ["a", "b"]
