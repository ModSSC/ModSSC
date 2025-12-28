from __future__ import annotations

import logging

from modssc.data_loader.providers.base import BaseProvider
from modssc.data_loader.providers.hf import HuggingFaceDatasetsProvider
from modssc.data_loader.providers.openml import OpenMLProvider
from modssc.data_loader.providers.pyg import PyGProvider
from modssc.data_loader.providers.tfds import TFDSProvider
from modssc.data_loader.providers.torchaudio import TorchaudioProvider
from modssc.data_loader.providers.torchvision import TorchvisionProvider
from modssc.data_loader.providers.toy import ToyProvider

PROVIDERS: dict[str, type[BaseProvider]] = {
    "toy": ToyProvider,
    "openml": OpenMLProvider,
    "hf": HuggingFaceDatasetsProvider,
    "tfds": TFDSProvider,
    "torchvision": TorchvisionProvider,
    "torchaudio": TorchaudioProvider,
    "pyg": PyGProvider,
}

logger = logging.getLogger(__name__)


def get_provider_names() -> list[str]:
    return sorted(PROVIDERS.keys())


def create_provider(name: str) -> BaseProvider:
    try:
        cls = PROVIDERS[name]
    except KeyError as e:
        from modssc.data_loader.errors import ProviderNotFoundError

        raise ProviderNotFoundError(name) from e
    logger.debug("Creating provider: %s", name)
    return cls()
