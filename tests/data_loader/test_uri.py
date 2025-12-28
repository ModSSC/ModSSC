from __future__ import annotations

import pytest

from modssc.data_loader.errors import InvalidDatasetURIError
from modssc.data_loader.uri import is_uri, parse_uri


def test_is_uri() -> None:
    assert is_uri("openml:61") is True
    assert is_uri("openml:") is False
    assert is_uri(":61") is False
    assert is_uri("toy") is False

    assert is_uri("p:r")
    assert not is_uri("no_colon")
    assert not is_uri(":r")
    assert not is_uri("p:")


def test_parse_uri_ok() -> None:
    p = parse_uri("hf:ag_news")
    assert p.provider == "hf"
    assert p.reference == "ag_news"
    assert p.uri == "hf:ag_news"

    parsed = parse_uri("provider:ref")
    assert parsed.provider == "provider"
    assert parsed.reference == "ref"
    assert parsed.uri == "provider:ref"


def test_parse_uri_invalid() -> None:
    with pytest.raises(InvalidDatasetURIError):
        parse_uri("no_colon")

    with pytest.raises(InvalidDatasetURIError):
        parse_uri("invalid")


def test_parse_uri_empty_provider():
    with pytest.raises(InvalidDatasetURIError):
        parse_uri(":ref")


def test_parse_uri_empty_ref():
    with pytest.raises(InvalidDatasetURIError):
        parse_uri("provider:")


def test_parse_uri_whitespace_only():
    with pytest.raises(InvalidDatasetURIError):
        parse_uri("  :  ")
