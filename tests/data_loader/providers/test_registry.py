from __future__ import annotations

import pytest

from modssc.data_loader.errors import ProviderNotFoundError
from modssc.data_loader.providers import create_provider, get_provider_names
from modssc.data_loader.providers.base import BaseProvider
from modssc.data_loader.uri import ParsedURI


def test_provider_registry_and_factory() -> None:
    names = get_provider_names()
    assert "toy" in names

    p = create_provider("toy")
    assert isinstance(p, BaseProvider)
    assert p.name == "toy"

    with pytest.raises(ProviderNotFoundError):
        create_provider("does_not_exist")


def test_base_provider_methods_raise() -> None:
    class Dummy(BaseProvider):
        name = "dummy"
        required_extra = None

        def resolve(self, parsed: ParsedURI, *, options):
            return super().resolve(parsed, options=options)

        def load_canonical(self, identity, *, raw_dir):
            return super().load_canonical(identity, raw_dir=raw_dir)

    d = Dummy()
    assert d.list() is None

    with pytest.raises(NotImplementedError):
        d.resolve(ParsedURI(provider="dummy", reference="x"), options={})

    with pytest.raises(NotImplementedError):
        d.load_canonical(identity=None, raw_dir=None)  # type: ignore[arg-type]
