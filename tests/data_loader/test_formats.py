from __future__ import annotations

import pytest

from modssc.data_loader.formats import get_output_format


def test_get_output_format_known() -> None:
    fmt = get_output_format("tabular")
    assert fmt.modality == "tabular"
    assert "n_samples" in fmt.X
    assert "n_samples" in fmt.y


def test_get_output_format_unknown() -> None:
    with pytest.raises(KeyError):
        get_output_format("unknown_modality")
