from __future__ import annotations

import numpy as np

from modssc.transductive.methods import utils as mutils


class FakeTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def test_to_numpy_detach_cpu_numpy() -> None:
    arr = np.array([1, 2, 3], dtype=np.float32)
    out = mutils.to_numpy(FakeTensor(arr))
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, arr)
