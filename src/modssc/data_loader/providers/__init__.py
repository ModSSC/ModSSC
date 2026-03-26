from __future__ import annotations

import logging

from modssc.data_loader.providers.base import BaseProvider
from modssc.utils.imports import load_object

PROVIDERS: dict[str, str] = {
    "toy": "modssc.data_loader.providers.toy:ToyProvider",
    "openml": "modssc.data_loader.providers.openml:OpenMLProvider",
    "hf": "modssc.data_loader.providers.hf:HuggingFaceDatasetsProvider",
    "tfds": "modssc.data_loader.providers.tfds:TFDSProvider",
    "torchvision": "modssc.data_loader.providers.torchvision:TorchvisionProvider",
    "torchaudio": "modssc.data_loader.providers.torchaudio:TorchaudioProvider",
    "pyg": "modssc.data_loader.providers.pyg:PyGProvider",
}

logger = logging.getLogger(__name__)


def get_provider_names() -> list[str]:
    return sorted(PROVIDERS.keys())


def create_provider(name: str) -> BaseProvider:
    try:
        import_path = PROVIDERS[name]
    except KeyError as exc:
        from modssc.data_loader.errors import ProviderNotFoundError

        raise ProviderNotFoundError(name) from exc
    logger.debug("Creating provider: %s", name)
    cls = load_object(import_path)
    return cls()
