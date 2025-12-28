from unittest.mock import patch

import pytest

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.registry import StepRegistry, _load_object


def test_load_object_invalid_path():
    """Test _load_object with invalid path format."""
    with pytest.raises(ValueError, match="Invalid import path"):
        _load_object("invalid_path_no_colon")


def test_registry_spec_unknown():
    """Test StepRegistry.spec with unknown step_id."""
    registry = StepRegistry(specs={})
    with pytest.raises(PreprocessValidationError, match="Unknown step id"):
        registry.spec("non_existent")


def test_load_object_valid():
    """Test _load_object with valid path."""

    obj = _load_object("os.path:join")
    import os.path

    assert obj is os.path.join


def test_registry_instantiate():
    """Test StepRegistry.instantiate."""

    class DummyStep:
        def __init__(self, a=1):
            self.a = a

    mock_spec = type("Spec", (), {"import_path": "mod:DummyStep"})
    registry = StepRegistry(specs={"dummy": mock_spec})

    with patch("modssc.preprocess.registry._load_object", return_value=DummyStep):
        step = registry.instantiate("dummy", params={"a": 2})
        assert isinstance(step, DummyStep)
        assert step.a == 2
