from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.optional import optional_import

SamplerMode = Literal["replacement", "shuffle_repeat"]


def interleave_batch(x: Any, groups: int) -> Any:
    """Interleave a flat SSL batch exactly as the Google FixMatch input path.

    ``groups`` is ``1 + 2 * mu`` for the concatenation of labeled, weak
    unlabeled and strong unlabeled inputs.  For the published CIFAR-10 setting
    this is 15.  The function supports NumPy arrays and torch tensors because
    both expose ``reshape`` and ``transpose`` with the signatures used here.
    """

    groups = _validate_groups(x, groups)
    shape = tuple(int(value) for value in x.shape)
    reshaped = x.reshape((-1, groups, *shape[1:]))
    permuted = reshaped.swapaxes(0, 1) if isinstance(x, np.ndarray) else reshaped.transpose(0, 1)
    return permuted.reshape(shape)


def deinterleave_batch(x: Any, groups: int) -> Any:
    """Invert :func:`interleave_batch` without changing the batch shape."""

    groups = _validate_groups(x, groups)
    shape = tuple(int(value) for value in x.shape)
    reshaped = x.reshape((groups, -1, *shape[1:]))
    permuted = reshaped.swapaxes(0, 1) if isinstance(x, np.ndarray) else reshaped.transpose(0, 1)
    return permuted.reshape(shape)


def _validate_groups(x: Any, groups: int) -> int:
    if not isinstance(groups, int) or isinstance(groups, bool) or groups <= 0:
        raise InductiveValidationError("groups must be a positive integer.")
    if not hasattr(x, "shape") or len(x.shape) == 0:
        raise InductiveValidationError("interleave input must have a batch dimension.")
    batch = int(x.shape[0])
    if batch % groups != 0:
        raise InductiveValidationError("batch size must be divisible by groups.")
    return groups


@dataclass(frozen=True)
class SSLBatchIndices:
    """One fixed-size labeled/unlabeled SSL batch."""

    labeled: np.ndarray
    unlabeled: np.ndarray


class _StatefulIndexStream:
    """Reference index stream with an exactly serializable state.

    Replacement sampling uses the same CPU ``torch.Generator`` and
    ``torch.randint`` primitive as TorchSSL's
    ``RandomSampler(replacement=True)``.  Shuffle-repeat sampling mirrors the
    state transition of TensorFlow 1 ``Dataset.repeat().shuffle(buffer)``; its
    slot generator is deliberately local because the upstream graph does not
    pin a portable TensorFlow/parallel-input RNG stream.
    """

    def __init__(
        self,
        *,
        size: int,
        seed: int,
        mode: SamplerMode,
        shuffle_buffer: int,
    ) -> None:
        self.size = int(size)
        self.mode = mode
        self.shuffle_buffer = int(shuffle_buffer)
        self._numpy_rng: np.random.Generator | None = None
        self._torch: Any | None = None
        self._torch_rng: Any | None = None
        if mode == "replacement":
            torch = optional_import("torch", extra="inductive-torch")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            self._torch = torch
            self._torch_rng = generator
        else:
            self._numpy_rng = np.random.Generator(
                np.random.PCG64(np.random.SeedSequence(int(seed)))
            )
        self._cursor = 0
        self._draws = 0
        self._buffer: np.ndarray | None = None
        if mode == "shuffle_repeat":
            self._buffer = np.arange(self.shuffle_buffer, dtype=np.int64) % self.size
            self._cursor = self.shuffle_buffer % self.size

    def draw(self, count: int) -> np.ndarray:
        count = int(count)
        if self.mode == "replacement":
            assert self._torch is not None
            assert self._torch_rng is not None
            result = (
                self._torch.randint(
                    high=self.size,
                    size=(count,),
                    dtype=self._torch.int64,
                    generator=self._torch_rng,
                    device="cpu",
                )
                .numpy()
                .copy()
            )
        else:
            assert self._buffer is not None
            assert self._numpy_rng is not None
            slots = self._numpy_rng.integers(
                0,
                int(self._buffer.size),
                size=count,
                dtype=np.int64,
            )
            result = np.empty(count, dtype=np.int64)
            for position, slot in enumerate(slots):
                result[position] = self._buffer[int(slot)]
                self._buffer[int(slot)] = self._cursor
                self._cursor = (self._cursor + 1) % self.size
        self._draws += count
        return result

    def state_dict(self) -> dict[str, Any]:
        if self.mode == "replacement":
            assert self._torch_rng is not None
            rng_backend = "torch_cpu"
            rng_state = self._torch_rng.get_state().clone()
        else:
            assert self._numpy_rng is not None
            rng_backend = "numpy_pcg64"
            rng_state = copy.deepcopy(self._numpy_rng.bit_generator.state)
        return {
            "size": self.size,
            "mode": self.mode,
            "shuffle_buffer": self.shuffle_buffer,
            "cursor": self._cursor,
            "draws": self._draws,
            "buffer": None if self._buffer is None else self._buffer.copy(),
            "rng_backend": rng_backend,
            "rng_state": rng_state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "size": self.size,
            "mode": self.mode,
            "shuffle_buffer": self.shuffle_buffer,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise InductiveValidationError(
                    f"Sampler state {key!r} does not match the configured stream."
                )
        cursor = state.get("cursor")
        draws = state.get("draws")
        if not isinstance(cursor, int) or not 0 <= cursor < self.size:
            raise InductiveValidationError("Sampler state cursor is invalid.")
        if not isinstance(draws, int) or draws < 0:
            raise InductiveValidationError("Sampler state draws is invalid.")

        raw_buffer = state.get("buffer")
        if self.mode == "replacement":
            if raw_buffer is not None:
                raise InductiveValidationError(
                    "Replacement sampler state must not contain a buffer."
                )
            buffer = None
        else:
            buffer = np.asarray(raw_buffer, dtype=np.int64)
            if buffer.shape != (self.shuffle_buffer,):
                raise InductiveValidationError("Shuffle-repeat sampler buffer shape is invalid.")
            if bool(((buffer < 0) | (buffer >= self.size)).any()):
                raise InductiveValidationError("Shuffle-repeat sampler buffer values are invalid.")
            buffer = buffer.copy()

        expected_backend = "torch_cpu" if self.mode == "replacement" else "numpy_pcg64"
        if state.get("rng_backend") != expected_backend:
            raise InductiveValidationError("Sampler RNG backend is invalid.")
        numpy_rng: np.random.Generator | None = None
        torch_module: Any | None = None
        torch_rng: Any | None = None
        if self.mode == "replacement":
            torch_module = optional_import("torch", extra="inductive-torch")
            raw_rng_state = state.get("rng_state")
            if (
                not torch_module.is_tensor(raw_rng_state)
                or raw_rng_state.dtype != torch_module.uint8
                or raw_rng_state.ndim != 1
            ):
                raise InductiveValidationError("Sampler RNG state is invalid.")
            torch_rng = torch_module.Generator(device="cpu")
            try:
                torch_rng.set_state(raw_rng_state.detach().cpu().clone())
            except RuntimeError as exc:
                raise InductiveValidationError("Sampler RNG state is invalid.") from exc
        else:
            numpy_rng = np.random.Generator(np.random.PCG64())
            try:
                numpy_rng.bit_generator.state = copy.deepcopy(state["rng_state"])
            except (KeyError, TypeError, ValueError) as exc:
                raise InductiveValidationError("Sampler RNG state is invalid.") from exc

        self._numpy_rng = numpy_rng
        self._torch = torch_module
        self._torch_rng = torch_rng
        self._cursor = cursor
        self._draws = draws
        self._buffer = buffer


def _torchssl_stream_seeds(seed: int) -> tuple[int, int]:
    """Allocate two explicit RandomSampler seeds with PyTorch's own primitive.

    TorchSSL leaves ``generator=None`` and therefore draws an implicit seed
    from PyTorch's process-global generator when each loader iterator starts.
    That seed depends on unrelated model and worker RNG consumption.  ModSSC
    intentionally isolates the two loaders while retaining PyTorch's exact
    seed-allocation primitive and replacement-sampling algorithm.
    """

    torch = optional_import("torch", extra="inductive-torch")
    root = torch.Generator(device="cpu")
    root.manual_seed(int(seed))
    return tuple(
        int(torch.empty((), dtype=torch.int64, device="cpu").random_(generator=root).item())
        for _ in range(2)
    )


class FixedSSLBatchSampler(Iterator[SSLBatchIndices]):
    """Yield fixed 64/448 SSL batches with replayable independent streams.

    ``replacement`` exactly matches TorchSSL's CPU
    ``RandomSampler(replacement=True)`` index primitive after explicit stream
    seed allocation. ``shuffle_repeat`` matches Google FixMatch's
    ``repeat().shuffle(buffer_size)`` buffer transition. Both modes serialize
    their generators, so resuming cannot change a subsequent index.
    """

    _STATE_VERSION = 2

    def __init__(
        self,
        labeled_size: int,
        unlabeled_size: int,
        *,
        labeled_batch_size: int = 64,
        unlabeled_batch_size: int = 448,
        seed: int = 0,
        mode: SamplerMode = "replacement",
        shuffle_buffer: int = 8192,
    ) -> None:
        for name, value in (
            ("labeled_size", labeled_size),
            ("unlabeled_size", unlabeled_size),
            ("labeled_batch_size", labeled_batch_size),
            ("unlabeled_batch_size", unlabeled_batch_size),
            ("shuffle_buffer", shuffle_buffer),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise InductiveValidationError(f"{name} must be a positive integer.")
        if mode not in ("replacement", "shuffle_repeat"):
            raise InductiveValidationError("mode must be either 'replacement' or 'shuffle_repeat'.")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise InductiveValidationError("seed must be an integer.")

        self.labeled_size = labeled_size
        self.unlabeled_size = unlabeled_size
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = unlabeled_batch_size
        self.seed = seed
        self.mode = mode
        self.shuffle_buffer = shuffle_buffer
        self._batches_yielded = 0

        if mode == "replacement":
            labeled_seed, unlabeled_seed = _torchssl_stream_seeds(seed)
        else:
            seed_sequence = np.random.SeedSequence(seed)
            labeled_seed, unlabeled_seed = (
                int(child.generate_state(1, dtype=np.uint64)[0]) for child in seed_sequence.spawn(2)
            )
        self._labeled = _StatefulIndexStream(
            size=labeled_size,
            seed=labeled_seed,
            mode=mode,
            shuffle_buffer=shuffle_buffer,
        )
        self._unlabeled = _StatefulIndexStream(
            size=unlabeled_size,
            seed=unlabeled_seed,
            mode=mode,
            shuffle_buffer=shuffle_buffer,
        )

    def __iter__(self) -> FixedSSLBatchSampler:
        return self

    def __next__(self) -> SSLBatchIndices:
        return self.next_batch()

    def next_batch(self) -> SSLBatchIndices:
        batch = SSLBatchIndices(
            labeled=self._labeled.draw(self.labeled_batch_size),
            unlabeled=self._unlabeled.draw(self.unlabeled_batch_size),
        )
        self._batches_yielded += 1
        return batch

    def contract(self) -> dict[str, str | int | bool]:
        """Return the stable, JSON-serializable scientific sampler contract."""

        if self.mode == "replacement":
            return {
                "mode": "replacement",
                "reference_algorithm": ("torch.utils.data.RandomSampler(replacement=True)"),
                "rng_backend": "torch_cpu_generator",
                "seed_policy": ("explicit_independent_loader_seeds_from_torch_root_generator"),
                "historical_bitstream_claimed": False,
            }
        return {
            "mode": "shuffle_repeat",
            "reference_algorithm": ("tensorflow.data.Dataset.repeat().shuffle(buffer_size)"),
            "rng_backend": "numpy_pcg64",
            "seed_policy": ("explicit_independent_loader_seeds_from_numpy_seedsequence"),
            "shuffle_buffer": self.shuffle_buffer,
            "historical_bitstream_claimed": False,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self._STATE_VERSION,
            "config": {
                "labeled_size": self.labeled_size,
                "unlabeled_size": self.unlabeled_size,
                "labeled_batch_size": self.labeled_batch_size,
                "unlabeled_batch_size": self.unlabeled_batch_size,
                "seed": self.seed,
                "mode": self.mode,
                "shuffle_buffer": self.shuffle_buffer,
            },
            "batches_yielded": self._batches_yielded,
            "labeled": self._labeled.state_dict(),
            "unlabeled": self._unlabeled.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != self._STATE_VERSION:
            raise InductiveValidationError("Unsupported sampler state version.")
        expected_config = self.state_dict()["config"]
        if state.get("config") != expected_config:
            raise InductiveValidationError("Sampler state configuration does not match.")
        batches_yielded = state.get("batches_yielded")
        if not isinstance(batches_yielded, int) or batches_yielded < 0:
            raise InductiveValidationError("Sampler state batches_yielded is invalid.")
        labeled_state = state.get("labeled")
        unlabeled_state = state.get("unlabeled")
        if not isinstance(labeled_state, Mapping) or not isinstance(unlabeled_state, Mapping):
            raise InductiveValidationError("Sampler stream states must be mappings.")

        # Validate into temporary streams so a failed restore cannot partially
        # mutate the live sampler.
        labeled = _StatefulIndexStream(
            size=self.labeled_size,
            seed=0,
            mode=self.mode,
            shuffle_buffer=self.shuffle_buffer,
        )
        unlabeled = _StatefulIndexStream(
            size=self.unlabeled_size,
            seed=0,
            mode=self.mode,
            shuffle_buffer=self.shuffle_buffer,
        )
        labeled.load_state_dict(labeled_state)
        unlabeled.load_state_dict(unlabeled_state)
        self._labeled = labeled
        self._unlabeled = unlabeled
        self._batches_yielded = batches_yielded


__all__ = [
    "FixedSSLBatchSampler",
    "SSLBatchIndices",
    "SamplerMode",
    "deinterleave_batch",
    "interleave_batch",
]
