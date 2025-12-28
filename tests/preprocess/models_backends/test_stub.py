import numpy as np

from modssc.preprocess.models_backends.stub import StubEncoder, _sample_bytes


def test_stub_encoder_bytes_input():
    encoder = StubEncoder(dim=4)

    data = [b"hello", b"world"]
    res = encoder.encode(data)
    assert res.shape == (2, 4)

    res2 = encoder.encode(data)
    assert np.allclose(res, res2)


def test_stub_encoder_numpy_input():
    encoder = StubEncoder(dim=4)

    data = np.random.randn(5, 10)
    res = encoder.encode(data)
    assert res.shape == (5, 4)


def test_stub_encoder_list_input():
    encoder = StubEncoder(dim=4)

    data = ["a", "b", "c"]
    res = encoder.encode(data)
    assert res.shape == (3, 4)


def test_stub_encoder_iterable_input():
    encoder = StubEncoder(dim=2)
    data = (x for x in ["a", "b"])
    res = encoder.encode(data)
    assert res.shape == (2, 2)


def test_sample_bytes_coverage():
    assert _sample_bytes(b"test") == b"test"
    assert _sample_bytes(bytearray(b"test")) == b"test"

    assert _sample_bytes("test") == b"test"

    arr = np.array([1, 2, 3], dtype=np.int32)
    b = _sample_bytes(arr)
    assert b"int32" in b
    assert b"|" in b

    large_arr = np.zeros(2000, dtype=np.uint8)
    b_large = _sample_bytes(large_arr)

    parts = b_large.split(b"|")
    assert len(parts) >= 2

    assert len(b_large) < 2000
