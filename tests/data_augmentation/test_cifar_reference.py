from __future__ import annotations

import hashlib
import platform
from dataclasses import replace
from importlib import metadata

import numpy as np
import pytest
import torch

from modssc.data_augmentation.cifar_reference import (
    CifarReferenceAugmentation,
    cifar_reference_runtime_identity,
    resolve_cifar_augmentation_profile,
)
from modssc.data_augmentation.errors import DataAugmentationValidationError


def test_profile_resolution_and_seed_validation() -> None:
    assert resolve_cifar_augmentation_profile("google_fixmatch_ra") == "google_fixmatch_ra"
    assert resolve_cifar_augmentation_profile("torchssl_ra") == "torchssl_ra"
    with pytest.raises(DataAugmentationValidationError, match="Unknown CIFAR"):
        resolve_cifar_augmentation_profile("unknown")
    with pytest.raises(DataAugmentationValidationError, match="seed"):
        CifarReferenceAugmentation("google_fixmatch_ra", seed=True)
    with pytest.raises(DataAugmentationValidationError, match="seed"):
        cifar_reference_runtime_identity(profile="google_fixmatch_ra", seed=True)


def test_runtime_identity_is_derived_from_the_executable_policy() -> None:
    google = CifarReferenceAugmentation("google_fixmatch_ra", seed=7).runtime_identity()
    torchssl = CifarReferenceAugmentation("torchssl_ra", seed=7).runtime_identity()

    assert google["augmenter_id"] == "vision.cifar_reference"
    assert google["schema_version"] == 2
    assert google["pixel_backend"] == {
        "pillow_version": metadata.version("Pillow"),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }
    assert google["config"]["profile"] == "google_fixmatch_ra"
    assert google["config"]["seed"] == 7
    assert google["config"]["padding"] == 4
    assert google["config"]["operation_names"] != torchssl["config"]["operation_names"]


def test_generic_online_batch_contract_selects_and_names_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    augmenter = CifarReferenceAugmentation("google_fixmatch_ra", seed=7)
    calls: list[tuple[np.ndarray, dict[str, object]]] = []

    def apply_batch(batch: np.ndarray, **kwargs: object) -> np.ndarray:
        calls.append((batch.copy(), dict(kwargs)))
        return batch + len(calls)

    monkeypatch.setattr(augmenter, "apply_batch", apply_batch)
    X = np.arange(15, dtype=np.float32).reshape(5, 3)
    indices = np.asarray([3, 1], dtype=np.int64)
    sample_ids = np.asarray([103, 101], dtype=np.int64)

    weak = augmenter.weak_batch(
        X,
        indices=indices,
        sample_ids=sample_ids,
        step=4,
    )
    unlabeled_weak, unlabeled_strong = augmenter.pair_batch(
        X,
        indices=indices,
        sample_ids=sample_ids,
        step=5,
    )

    np.testing.assert_array_equal(calls[0][0], X[indices])
    assert [call[1]["view"] for call in calls] == [
        "labeled_weak",
        "unlabeled_weak",
        "unlabeled_strong",
    ]
    assert calls[0][1]["step"] == 4
    assert calls[1][1]["step"] == calls[2][1]["step"] == 5
    assert all(call[1]["sample_ids"] is sample_ids for call in calls)
    np.testing.assert_array_equal(weak, X[indices] + 1)
    np.testing.assert_array_equal(unlabeled_weak, X[indices] + 2)
    np.testing.assert_array_equal(unlabeled_strong, X[indices] + 3)


def test_worker_configuration_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODSSC_AUGMENT_THREADS", raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: None)
    assert CifarReferenceAugmentation("google_fixmatch_ra")._worker_count == 1

    monkeypatch.setenv("MODSSC_AUGMENT_THREADS", "not-an-integer")
    with pytest.raises(DataAugmentationValidationError, match="must be an integer"):
        CifarReferenceAugmentation("google_fixmatch_ra")

    monkeypatch.setenv("MODSSC_AUGMENT_THREADS", "0")
    with pytest.raises(DataAugmentationValidationError, match="must be positive"):
        CifarReferenceAugmentation("google_fixmatch_ra")


@pytest.mark.parametrize(
    ("profile", "num_ops", "fixed_cutout"),
    [
        ("google_fixmatch_ra", 2, True),
        ("torchssl_ra", 3, False),
    ],
)
def test_vectorized_draw_oracle_is_per_occurrence_replayable(
    profile: str,
    num_ops: int,
    fixed_cutout: bool,
) -> None:
    augmenter = CifarReferenceAugmentation(profile, seed=17)
    ids = np.array([91, 3, 44], dtype=np.int64)
    occurrences = np.array([5, 7, 11], dtype=np.int64)
    draws = augmenter.sample_batch(
        ids,
        occurrence_ids=occurrences,
        step=8,
        view="unlabeled_strong",
    )
    reordered = augmenter.sample_batch(
        ids[[2, 0]],
        occurrence_ids=occurrences[[2, 0]],
        step=8,
        view="unlabeled_strong",
    )

    assert draws.operation_indices.shape == (3, num_ops)
    assert draws.magnitudes.shape == (3, num_ops)
    assert draws.apply_operation.shape == (3, num_ops)
    np.testing.assert_array_equal(reordered.crop_top, draws.crop_top[[2, 0]])
    np.testing.assert_array_equal(reordered.operation_indices, draws.operation_indices[[2, 0]])
    np.testing.assert_array_equal(reordered.magnitudes, draws.magnitudes[[2, 0]])
    if fixed_cutout:
        np.testing.assert_array_equal(draws.cutout_size_fraction, np.full(3, 0.5))
        assert bool(((draws.magnitudes >= 1) & (draws.magnitudes <= 9)).all())
    else:
        assert bool((draws.cutout_size_fraction < 0.5).all())
        assert bool(draws.apply_operation.all())


def test_replacement_duplicates_receive_independent_draws() -> None:
    augmenter = CifarReferenceAugmentation("torchssl_ra", seed=5)
    draws = augmenter.sample_batch(
        np.array([3, 3], dtype=np.int64),
        occurrence_ids=np.array([0, 1], dtype=np.int64),
        step=2,
        view="unlabeled_strong",
    )
    assert not np.array_equal(draws.operation_indices[0], draws.operation_indices[1])
    assert not np.array_equal(draws.magnitudes[0], draws.magnitudes[1])


def test_three_view_domains_are_independent_and_weak_has_no_policy() -> None:
    augmenter = CifarReferenceAugmentation("torchssl_ra", seed=2)
    ids = np.arange(8, dtype=np.int64)
    labeled = augmenter.sample_batch(ids, step=0, view="labeled_weak")
    unlabeled = augmenter.sample_batch(ids, step=0, view="unlabeled_weak")
    strong = augmenter.sample_batch(ids, step=0, view="unlabeled_strong")

    assert labeled.operation_indices.shape == unlabeled.operation_indices.shape == (8, 0)
    assert strong.operation_indices.shape == (8, 3)
    assert not np.array_equal(labeled.crop_top, unlabeled.crop_top)
    assert not np.array_equal(unlabeled.crop_top, strong.crop_top)
    assert not labeled.cutout_size_fraction.any()
    assert not unlabeled.cutout_size_fraction.any()


def test_draw_oracle_has_stable_known_values() -> None:
    draws = CifarReferenceAugmentation("google_fixmatch_ra", seed=7).sample_batch(
        np.array([0, 10], dtype=np.int64),
        step=5,
        view="unlabeled_strong",
    )
    np.testing.assert_array_equal(draws.crop_top, np.array([4, 6]))
    np.testing.assert_array_equal(draws.crop_left, np.array([5, 8]))
    np.testing.assert_array_equal(draws.flip, np.array([True, False]))
    np.testing.assert_array_equal(draws.operation_indices, np.array([[10, 13], [8, 13]]))
    np.testing.assert_array_equal(draws.magnitudes, np.array([[5.0, 6.0], [9.0, 3.0]]))


@pytest.mark.parametrize("profile", ["google_fixmatch_ra", "torchssl_ra"])
@pytest.mark.parametrize("channels_last", [False, True])
def test_apply_batch_is_deterministic_preserves_layout_and_batch_partition(
    profile: str,
    channels_last: bool,
) -> None:
    augmenter = CifarReferenceAugmentation(profile, seed=13)
    batch = torch.arange(4 * 3 * 32 * 32, dtype=torch.int64).remainder(256).to(torch.uint8)
    batch = batch.reshape(4, 3, 32, 32)
    if channels_last:
        batch = batch.permute(0, 2, 3, 1)
    ids = torch.tensor([11, 5, 8, 2])

    whole = augmenter.apply_batch(
        batch,
        sample_ids=ids,
        occurrence_ids=np.arange(4),
        step=4,
        view="unlabeled_strong",
    )
    repeated = augmenter.apply_batch(
        batch,
        sample_ids=ids,
        occurrence_ids=np.arange(4),
        step=4,
        view="unlabeled_strong",
    )
    split = torch.cat(
        [
            augmenter.apply_batch(
                batch[:2],
                sample_ids=ids[:2],
                occurrence_ids=np.arange(2),
                step=4,
                view="unlabeled_strong",
            ),
            augmenter.apply_batch(
                batch[2:],
                sample_ids=ids[2:],
                occurrence_ids=np.arange(2, 4),
                step=4,
                view="unlabeled_strong",
            ),
        ],
        dim=0,
    )
    assert whole.shape == batch.shape
    assert whole.dtype == batch.dtype
    torch.testing.assert_close(whole, repeated)
    torch.testing.assert_close(whole, split)


def test_apply_batch_numpy_float_and_empty() -> None:
    augmenter = CifarReferenceAugmentation("google_fixmatch_ra", seed=1)
    batch = np.linspace(-1.0, 1.0, 2 * 32 * 32 * 3, dtype=np.float32).reshape(2, 32, 32, 3)
    output = augmenter.apply_batch(
        batch,
        sample_ids=np.array([3, 4]),
        step=1,
        view="labeled_weak",
    )
    empty = augmenter.apply_batch(
        batch[:0],
        sample_ids=np.array([], dtype=np.int64),
        step=1,
        view="labeled_weak",
    )
    assert isinstance(output, np.ndarray)
    assert output.shape == batch.shape
    assert output.dtype == np.float32
    assert float(output.min()) >= -1.0 and float(output.max()) <= 1.0
    assert empty.shape == (0, 32, 32, 3)


def test_float_reference_paths_and_single_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODSSC_AUGMENT_THREADS", "1")
    google = CifarReferenceAugmentation("google_fixmatch_ra", seed=4)
    google_input = torch.linspace(-1.0, 1.0, 3 * 32 * 32).reshape(1, 3, 32, 32)
    google_output = google.apply_batch(
        google_input,
        sample_ids=np.array([5]),
        step=2,
        view="unlabeled_strong",
    )
    assert google_output.dtype == google_input.dtype
    assert float(google_output.min()) >= -1.0
    assert float(google_output.max()) <= 1.0

    torchssl = CifarReferenceAugmentation("torchssl_ra", seed=4)
    torchssl_input = torch.linspace(0.0, 1.0, 3 * 32 * 32).reshape(1, 3, 32, 32)
    torchssl_output = torchssl.apply_batch(
        torchssl_input,
        sample_ids=np.array([5]),
        step=2,
        view="unlabeled_weak",
    )
    assert torchssl_output.dtype == torchssl_input.dtype
    assert float(torchssl_output.min()) >= 0.0
    assert float(torchssl_output.max()) <= 1.0


def test_torchssl_pillow_path_without_cutout() -> None:
    augmenter = CifarReferenceAugmentation("torchssl_ra")
    draws = augmenter.sample_batch(
        np.array([7]),
        step=3,
        view="unlabeled_strong",
    )
    draws = replace(
        draws,
        cutout_size_fraction=np.zeros(1, dtype=np.float64),
    )
    raw = np.zeros((32, 32, 3), dtype=np.uint8)
    output = augmenter._pillow_task(("torchssl", raw, draws, 0))
    assert output.shape == raw.shape
    assert output.dtype == raw.dtype


@pytest.mark.parametrize(
    ("sample_ids", "step", "view", "message"),
    [
        (np.array([[1]]), 0, "labeled_weak", "1D integer"),
        (np.array([1.0]), 0, "labeled_weak", "1D integer"),
        (np.array([-1]), 0, "labeled_weak", "non-negative"),
        (np.array([1]), -1, "labeled_weak", "step"),
        (np.array([1]), True, "labeled_weak", "step"),
        (np.array([1]), 0, "bad", "view"),
    ],
)
def test_sample_batch_validation(sample_ids, step, view, message) -> None:
    augmenter = CifarReferenceAugmentation("google_fixmatch_ra")
    with pytest.raises(DataAugmentationValidationError, match=message):
        augmenter.sample_batch(sample_ids, step=step, view=view)


def test_occurrence_ids_validation() -> None:
    augmenter = CifarReferenceAugmentation("google_fixmatch_ra")
    with pytest.raises(DataAugmentationValidationError, match="equal size"):
        augmenter.sample_batch(
            np.array([1, 2]),
            occurrence_ids=np.array([0]),
            step=0,
            view="labeled_weak",
        )


@pytest.mark.parametrize(
    ("batch", "ids", "message"),
    [
        (torch.zeros(3, 32, 32), np.array([0]), "4D"),
        (torch.zeros(1, 1, 32, 32), np.array([0]), "3 channels"),
        (torch.zeros(2, 3, 32, 32), np.array([0]), "equal size"),
        (torch.zeros(1, 3, 4, 4), np.array([0]), "too small"),
        (torch.full((1, 3, 32, 32), 1.1), np.array([0]), r"\[-1, 1\]"),
        (torch.full((1, 3, 32, 32), float("nan")), np.array([0]), "finite"),
        (torch.zeros(1, 3, 32, 32, dtype=torch.int64), np.array([0]), "dtype"),
    ],
)
def test_apply_draws_validation(batch, ids, message) -> None:
    augmenter = CifarReferenceAugmentation("google_fixmatch_ra")
    draws = augmenter.sample_batch(ids, step=0, view="labeled_weak")
    with pytest.raises(DataAugmentationValidationError, match=message):
        augmenter.apply_draws(batch, draws)


def test_apply_draws_rejects_profile_mismatch() -> None:
    google = CifarReferenceAugmentation("google_fixmatch_ra")
    torchssl = CifarReferenceAugmentation("torchssl_ra")
    draws = torchssl.sample_batch(np.array([0]), step=0, view="labeled_weak")
    with pytest.raises(DataAugmentationValidationError, match="different profile"):
        google.apply_draws(torch.zeros(1, 3, 32, 32, dtype=torch.uint8), draws)


def test_pillow_operations_match_pinned_upstream_pixel_fixtures() -> None:
    """Hashes were generated directly from the two pinned upstream files."""

    from PIL import Image

    raw = np.random.default_rng(12345).integers(
        16,
        240,
        size=(17, 19, 3),
        dtype=np.uint8,
    )
    normalized = raw.astype(np.float32) * (2.0 / 255.0) - 1.0
    google_input = np.uint8((normalized * 0.5 + 0.5) * 255.0)
    expected_google = {
        "Identity": "69cc45a0b8f6f629b341fc0e1081bd29690fd8f58d104d0ce031df97d7557d48",
        "AutoContrast": "45c404c29864f599517cb7fcabd8b3456bb0d18b9e2945ad875f363d6692b9ce",
        "Equalize": "0262628baaa9c98224f41f134da5227dd7e5a9d08004840be8ff6e292cd6c1b5",
        "Rotate": "429003475cb7a5431f13add648a6d08fd0d70c594d49898b448ee4f70855fa85",
        "Solarize": "1d85b0062f7891ac8d6d2dd5512f9bf0940c3f93299230ff005a82b9af53a78c",
        "Contrast": "e2148bd9ab4b10f6532bef00a5288748a0cd768318587e9c66934bb6f2a4e185",
        "Brightness": "ea6b829d2856573e564b42a80fa387635343c737e8d4e008dea3a8c70e637cac",
        "Sharpness": "8e05fb82465d01332737e95e98e00f05fa803c99837c7d7b13ac252fc1a328f0",
        "ShearX": "3c08637b1e09e0bfc7f2ba68d6c2ba8fc9d1e60eed595837b08ec5115702c539",
        "TranslateX": "3afd906bc82968c975218d95b33b24d7658525c5f9087e5654d3d7b2047ae9b9",
        "TranslateY": "5e897377a05006cfea8a761760e0c0e40d8498402d435e68593406410d292ae4",
        "Posterize": "f70cedbd6d656b93f2873af3c19b034ae1f7975cb35230187070494d1a79020b",
        "ShearY": "306d76ab9430739b05fe386cb3db3494bf1579e72dd7fb37109848dfb619f9c9",
    }
    expected_google_by_platform = {
        # Pillow 12.3.0's RGBA ImageEnhance Color and Contrast results are
        # build/platform sensitive. Keep the complete upstream oracle bound to
        # a verified environment; an unknown platform must fail closed.
        ("Darwin", "arm64"): {
            **expected_google,
            "Color": "932286f7f4c1cfca70c635b25a5f4cc590a576ba71a4b0e640d31e76e9127550",
        },
        ("Linux", "x86_64"): {
            **expected_google,
            "Color": "afdaf683506227f89d6c00dd5455bf2347b15672ff42feaf4b21b7c6cd94c9dc",
            "Contrast": "da3b8cd87666687a99989fcb5138c29d74b9fd44cd7c4d23dceadb54b26979bb",
        },
    }
    platform_key = (platform.system(), platform.machine())
    assert platform_key in expected_google_by_platform, (
        f"unverified Pillow pixel oracle platform: {platform_key!r}"
    )
    expected_google = expected_google_by_platform[platform_key]
    google = CifarReferenceAugmentation("google_fixmatch_ra")
    for index, name in enumerate(google.operation_names):
        transformed = google._apply_pillow_operation(
            Image.fromarray(google_input).convert("RGBA"),
            operation_index=index,
            magnitude=7.0,
            sign=-1.0,
        )
        rgba = np.asarray(transformed, dtype=np.float64) / 255.0
        output = (rgba[:, :, :3] - 0.5) / 0.5
        output[rgba[:, :, 3] == 0] = 0.0
        digest = hashlib.sha256(output.astype(np.float32).tobytes()).hexdigest()
        assert digest == expected_google[name]

    expected_torchssl = {
        "AutoContrast": "550cc4c1d15b298b7a75b180bef15d48cd633a4a8cb9008dbd6e240ad48dd911",
        "Brightness": "dd732cb25a1c8db81f8caf7cf9f646931502f1ee26a3a9f9c217c1df728b448c",
        "Color": "2f0648c4a1bd3809aa41b0381cc1f4afb935ab86251d012fec4aeb19f61f6174",
        "Contrast": "b15a7465b20c61283a3812804fc6a842a34e879047459c6535baed72315f4487",
        "Equalize": "8fc27d2df0731345628b53190998fb8331d69540ffd43c36bbf90cedc43ed9c1",
        "Identity": "76aa9ee526af1393ba435cd2aa09a1db583fd70a0efc7dd52a1159806a74dd34",
        "Posterize": "e0abc375b54744413cb7d928330de4e3fbc551a1021d3966358fb66f39170be2",
        "Rotate": "ece3e77fff455d26015f77c41d9caca001ad64a3cc17af716a6ffb0aa78c779d",
        "Sharpness": "5ae3dfdb4f901ff8dda301df78c67b697d24924520064f451205140845bb021c",
        "ShearX": "ebf1fe71e22fd2f7982d27d2bcb645464cbb919784628abfd61ee21f69c77376",
        "ShearY": "03059f7d5da2c82738e35013a2ac6035bae5cc449250518bc237d4e7abff3abe",
        "Solarize": "19863d0a9783e250483469896e75636664e1c3ca70e27a1bec9a7700551ebe1d",
        "TranslateX": "69d3feba74403c89daa718a7c3ad209ace952daddd9b81a96025bed301e2621f",
        "TranslateY": "5f6a8ce58549edd01d1a000464fa949b13332428e72084db0a616467f422c267",
    }
    torchssl = CifarReferenceAugmentation("torchssl_ra")
    for index, name in enumerate(torchssl.operation_names):
        transformed = torchssl._apply_pillow_operation(
            Image.fromarray(raw).convert("RGB"),
            operation_index=index,
            magnitude=0.37,
            sign=1.0,
        )
        digest = hashlib.sha256(np.asarray(transformed, dtype=np.uint8).tobytes()).hexdigest()
        assert digest == expected_torchssl[name]


def test_apply_cutout_zero_and_profile_fills() -> None:
    batch = torch.zeros((1, 3, 8, 8), dtype=torch.uint8)
    google = CifarReferenceAugmentation("google_fixmatch_ra")
    weak = google.sample_batch(np.array([0]), step=0, view="labeled_weak")
    assert google._apply_cutout(batch, weak) is batch

    base = google.sample_batch(np.array([0]), step=0, view="unlabeled_strong")
    centered = replace(
        base,
        cutout_size_fraction=np.array([0.5]),
        cutout_center_y=np.array([0.5]),
        cutout_center_x=np.array([0.5]),
    )
    google_output = google._apply_cutout(
        torch.ones((1, 3, 8, 8), dtype=torch.float32),
        centered,
    )
    assert bool((google_output == 0.0).any())

    torchssl = CifarReferenceAugmentation("torchssl_ra")
    torchssl_draws = replace(centered, profile="torchssl_ra")
    torchssl_output = torchssl._apply_cutout(torch.zeros_like(batch), torchssl_draws)
    center_pixel = torchssl_output[0, :, 4, 4]
    torch.testing.assert_close(center_pixel, torch.tensor([125, 123, 114], dtype=torch.uint8))
