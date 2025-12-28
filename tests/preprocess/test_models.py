import pytest

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.models import (
    _load_object,
    available_models,
    load_encoder,
    model_info,
    model_spec,
)


def test_load_object_invalid_path():
    with pytest.raises(ValueError, match="Invalid import path"):
        _load_object("invalid_path_no_colon")


def test_load_object_success():
    obj = _load_object("modssc.preprocess.errors:PreprocessError")
    assert obj is not None
    assert obj.__name__ == "PreprocessError"


def test_available_models_filter():
    text_models = available_models(modality="text")
    assert len(text_models) > 0
    for m in text_models:
        spec = model_spec(m)
        assert spec.modality == "text"

    assert available_models(modality="nonexistent") == []


def test_available_models_all():
    all_models = available_models()
    assert len(all_models) > 0
    assert "stub:text" in all_models


def test_model_spec_unknown():
    with pytest.raises(PreprocessValidationError, match="Unknown model id"):
        model_spec("unknown:model")


def test_model_info():
    info = model_info("stub:text")
    assert info["id"] == "stub:text"
    assert info["modality"] == "text"
    assert info["import_path"] == "modssc.preprocess.models_backends.stub:StubEncoder"
    assert info["description"] is not None
    assert isinstance(info["default_kwargs"], dict)


def test_load_encoder():
    encoder = load_encoder("stub:text", dim=16)
    assert encoder is not None
