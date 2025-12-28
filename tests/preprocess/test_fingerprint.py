from modssc.preprocess.fingerprint import (
    derive_seed,
    fingerprint,
    shallow_mapping,
    stable_json_dumps,
)


def test_stable_json_dumps():
    data = {"b": 2, "a": 1}

    assert stable_json_dumps(data) == '{"a":1,"b":2}'


def test_fingerprint():
    data = {"a": 1}
    fp = fingerprint(data)
    assert isinstance(fp, str)
    assert len(fp) == 64

    fp_prefix = fingerprint(data, prefix="test:")
    assert fp_prefix.startswith("test:")
    assert fp_prefix.endswith(fp)


def test_derive_seed():
    seed = derive_seed(12345, step_id="step1", step_index=0)
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32

    seed2 = derive_seed(12345, step_id="step1", step_index=0)
    assert seed == seed2

    seed3 = derive_seed(12345, step_id="step1", step_index=1)
    assert seed != seed3


def test_shallow_mapping_nested():
    data = {"a": 1, "b": {"x": 10, "y": 20}, "c": [1, 2]}
    result = shallow_mapping(data)
    assert result == data
    assert isinstance(result["b"], dict)

    assert result["b"] is not data["b"] if isinstance(data["b"], dict) else True
