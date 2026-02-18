from __future__ import annotations

import pytest

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.adsh import ADSHMethod, ADSHSpec
from modssc.inductive.types import DeviceSpec

from ..conftest import (
    make_model_bundle,
    make_numpy_dataset,
    make_torch_dataset,
    make_torch_ssl_dataset,
)


def _make_method(**spec_overrides) -> ADSHMethod:
    kwargs = {"model_bundle": make_model_bundle(), "batch_size": 2, "max_epochs": 1}
    kwargs.update(spec_overrides)
    return ADSHMethod(ADSHSpec(**kwargs))


def test_adsh_fit_data_none():
    with pytest.raises(InductiveValidationError, match="data must not be None"):
        _make_method().fit(None, device=DeviceSpec(device="cpu"), seed=0)  # type: ignore[arg-type]


def test_adsh_fit_requires_torch_backend():
    data = make_numpy_dataset()
    with pytest.raises(InductiveValidationError, match="requires torch tensors"):
        _make_method().fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_adsh_fit_requires_unlabeled_views():
    data = make_torch_dataset()
    with pytest.raises(InductiveValidationError, match="requires X_u_w and X_u_s"):
        _make_method().fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_adsh_invalid_score_warmup_epochs():
    data = make_torch_ssl_dataset()
    with pytest.raises(InductiveValidationError, match="score_warmup_epochs must be >= 0"):
        _make_method(score_warmup_epochs=-1).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_adsh_invalid_p_cutoff():
    data = make_torch_ssl_dataset()
    with pytest.raises(InductiveValidationError, match="p_cutoff must be in \\(0, 1\\]"):
        _make_method(p_cutoff=0.0).fit(data, device=DeviceSpec(device="cpu"), seed=0)
