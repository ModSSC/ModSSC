import numpy as np

from modssc.preprocess.steps.vision.squeeze import SqueezeStep
from modssc.preprocess.store import ArtifactStore


def test_squeeze_step():
    rng = np.random.default_rng(0)
    # (N, H, W, 1) -> (N, H, W)
    store = ArtifactStore({"raw.X": np.zeros((2, 4, 4, 1))})
    step = SqueezeStep(dim=-1, as_list=False)
    out = step.transform(store, rng=rng)
    assert out["raw.X"].shape == (2, 4, 4)

    # as_list=True
    step_list = SqueezeStep(dim=-1, as_list=True)
    out_list = step_list.transform(store, rng=rng)
    assert isinstance(out_list["raw.X"], list)
    assert len(out_list["raw.X"]) == 2
    assert out_list["raw.X"][0].shape == (4, 4)

    # No squeeze needed (channel != 1)
    store2 = ArtifactStore({"raw.X": np.zeros((2, 4, 4, 3))})
    out2 = step.transform(store2, rng=rng)
    assert out2["raw.X"].shape == (2, 4, 4, 3)

    # ndim != 4
    store3 = ArtifactStore({"raw.X": np.zeros((2, 4))})
    out3 = step.transform(store3, rng=rng)
    assert out3["raw.X"].shape == (2, 4)
