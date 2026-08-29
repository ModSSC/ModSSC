from __future__ import annotations

import pytest
import torch

from modssc.inductive.deep.wide_resnet import (
    WideResidualBlock,
    WideResNetCifar,
    resolve_wide_resnet_reference,
)
from modssc.inductive.errors import InductiveValidationError


def test_wide_resnet_28_2_matches_fixmatch_architecture() -> None:
    torch.manual_seed(7)
    model = WideResNetCifar(in_channels=3, num_classes=10)

    assert model.depth == 28
    assert model.widen_factor == 2
    assert model.blocks_per_stage == 4
    assert model.channels == (16, 32, 64, 128)
    assert len(model.blocks) == 12
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_469_642
    assert model.blocks[0].activate_before_residual is True
    assert model.blocks[1].activate_before_residual is False
    assert model.blocks[0].shortcut is not None
    assert model.blocks[1].shortcut is None
    assert model.blocks[4].conv1.stride == (2, 2)
    assert model.blocks[8].conv1.stride == (2, 2)
    assert model.blocks[0].bn1.momentum == pytest.approx(0.001)
    assert model.blocks[0].bn1.eps == pytest.approx(1e-3)
    assert torch.count_nonzero(model.stem.bias).item() == 0
    assert torch.count_nonzero(model.classifier.bias).item() == 0
    assert torch.allclose(model.bn.weight, torch.ones_like(model.bn.weight))

    features = model.forward_features(torch.randn(2, 3, 32, 32))
    head_logits = model.forward_head(features)
    logits = model(torch.randn(2, 3, 32, 32))
    assert features.shape == (2, 128)
    assert head_logits.shape == (2, 10)
    assert logits.shape == (2, 10)


def test_wide_resnet_internal_channel_normalization() -> None:
    model = WideResNetCifar(
        in_channels=3,
        num_classes=2,
        depth=10,
        widen_factor=1,
        input_mean=[0.1, 0.2, 0.3],
        input_std=[0.5, 0.25, 0.1],
    )
    seen: list[torch.Tensor] = []
    handle = model.stem.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    x = torch.full((2, 3, 8, 8), 0.5)
    model(x)
    handle.remove()

    expected = (x - model.input_mean) / model.input_std
    assert torch.allclose(seen[0], expected)
    assert model.input_mean.shape == (1, 3, 1, 1)
    assert model.input_std.shape == (1, 3, 1, 1)


def test_wide_residual_block_projection_variants() -> None:
    downsample = WideResidualBlock(
        in_channels=4,
        out_channels=4,
        stride=2,
        activate_before_residual=False,
        bn_momentum=0.001,
        bn_eps=1e-3,
    )
    assert downsample.shortcut is not None
    assert downsample(torch.randn(2, 4, 8, 8)).shape == (2, 4, 4, 4)

    identity = WideResidualBlock(
        in_channels=4,
        out_channels=4,
        stride=1,
        activate_before_residual=True,
        bn_momentum=0.001,
        bn_eps=1e-3,
    )
    assert identity.shortcut is None
    assert identity(torch.randn(2, 4, 8, 8)).shape == (2, 4, 8, 8)


def test_torchssl_wrn_matches_pinned_block_structure_and_transition_path() -> None:
    model = WideResNetCifar(
        in_channels=3,
        num_classes=10,
        reference_implementation="torchssl",
    )

    assert model.reference_implementation == "torchssl"
    assert model.blocks[0].conv1.bias is not None
    assert model.blocks[0].conv2.bias is not None
    assert model.blocks[0].shortcut.bias is not None
    assert model.blocks[0].bn1.eps == pytest.approx(1e-3)
    assert model.bn.eps == pytest.approx(1e-3)
    assert model.stem.bias is not None

    # The first block is the sole TorchSSL transition that activates before
    # both its residual projection and its first convolution.
    first = model.blocks[0]
    first_seen_conv: list[torch.Tensor] = []
    first_seen_shortcut: list[torch.Tensor] = []
    first_conv_handle = first.conv1.register_forward_pre_hook(
        lambda _module, args: first_seen_conv.append(args[0].detach().clone())
    )
    first_shortcut_handle = first.shortcut.register_forward_pre_hook(
        lambda _module, args: first_seen_shortcut.append(args[0].detach().clone())
    )
    first_input = torch.randn(2, 16, 8, 8)
    first_expected = first._activate(first.bn1(first_input))
    first(first_input)
    first_conv_handle.remove()
    first_shortcut_handle.remove()
    torch.testing.assert_close(first_seen_conv[0], first_expected)
    torch.testing.assert_close(first_seen_shortcut[0], first_expected)

    # In TorchSSL, a later channel-changing transition with
    # activate_before_residual=False sends the raw input to both conv1 and the
    # shortcut. This is the non-obvious branch in the official BasicBlock.
    transition = model.blocks[4]
    seen_conv: list[torch.Tensor] = []
    seen_shortcut: list[torch.Tensor] = []
    conv_handle = transition.conv1.register_forward_pre_hook(
        lambda _module, args: seen_conv.append(args[0].detach().clone())
    )
    shortcut_handle = transition.shortcut.register_forward_pre_hook(
        lambda _module, args: seen_shortcut.append(args[0].detach().clone())
    )
    x = torch.randn(2, 32, 8, 8)
    transition(x)
    conv_handle.remove()
    shortcut_handle.remove()
    torch.testing.assert_close(seen_conv[0], x)
    torch.testing.assert_close(seen_shortcut[0], x)


def test_torchssl_wrn_reset_preserves_conv_constructor_biases_only() -> None:
    torchssl = WideResNetCifar(
        in_channels=3,
        num_classes=10,
        depth=10,
        widen_factor=1,
        reference_implementation="torchssl",
    )
    google = WideResNetCifar(
        in_channels=3,
        num_classes=10,
        depth=10,
        widen_factor=1,
        reference_implementation="google_fixmatch",
    )
    standardized = WideResNetCifar(
        in_channels=3,
        num_classes=10,
        depth=10,
        widen_factor=1,
    )

    # The completed TorchSSL constructor must still expose PyTorch's Conv2d
    # bias draw, rather than a later all-zero rewrite.
    constructor_biases = [
        module.bias
        for module in torchssl.modules()
        if isinstance(module, torch.nn.Conv2d) and module.bias is not None
    ]
    assert constructor_biases
    assert all(torch.count_nonzero(bias).item() == bias.numel() for bias in constructor_biases)

    for model in (torchssl, google, standardized):
        for index, module in enumerate(model.modules(), start=1):
            if isinstance(module, torch.nn.Conv2d) and module.bias is not None:
                module.bias.detach().fill_(float(index) / 100.0)

    torchssl_before = {
        name: module.bias.detach().clone()
        for name, module in torchssl.named_modules()
        if isinstance(module, torch.nn.Conv2d) and module.bias is not None
    }
    torchssl._reset_parameters()
    google._reset_parameters()
    standardized._reset_parameters()

    assert torchssl_before
    for name, before in torchssl_before.items():
        torch.testing.assert_close(torchssl.get_submodule(name).bias, before)
    for model in (google, standardized):
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d) and module.bias is not None:
                assert torch.count_nonzero(module.bias).item() == 0


def test_wrn_reference_resolution_and_validation() -> None:
    assert resolve_wide_resnet_reference("google_fixmatch") == "google_fixmatch"
    assert resolve_wide_resnet_reference("torchssl") == "torchssl"
    assert resolve_wide_resnet_reference() == "standardized"
    with pytest.raises(InductiveValidationError, match="must be"):
        resolve_wide_resnet_reference("unknown")
    with pytest.raises(InductiveValidationError, match="Unknown WideResidualBlock"):
        WideResidualBlock(
            in_channels=4,
            out_channels=4,
            stride=1,
            activate_before_residual=False,
            bn_momentum=0.001,
            bn_eps=1e-3,
            reference_implementation="unknown",
        )
    with pytest.raises(InductiveValidationError, match="channel-changing"):
        WideResidualBlock(
            in_channels=4,
            out_channels=4,
            stride=2,
            activate_before_residual=False,
            bn_momentum=0.001,
            bn_eps=1e-3,
            reference_implementation="torchssl",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"depth": 4}, "depth must satisfy"),
        ({"depth": 12}, "depth must satisfy"),
        ({"widen_factor": 0}, "widen_factor"),
        ({"in_channels": 0}, "in_channels"),
        ({"num_classes": 0}, "num_classes"),
        ({"bn_momentum": 0.0}, "bn_momentum"),
        ({"bn_momentum": 1.1}, "bn_momentum"),
        ({"bn_eps": 0.0}, "bn_eps"),
        ({"input_std": [1.0, 1.0, 1.0]}, "provided together"),
        ({"input_mean": [0.0, 0.0, 0.0]}, "provided together"),
        (
            {"input_mean": [0.0], "input_std": [1.0, 1.0, 1.0]},
            "one value per input channel",
        ),
        (
            {"input_mean": [0.0, 0.0, 0.0], "input_std": [1.0]},
            "one value per input channel",
        ),
        (
            {"input_mean": [0.0, 0.0, 0.0], "input_std": [1.0, 0.0, 1.0]},
            "input_std values",
        ),
    ],
)
def test_wide_resnet_constructor_validation(kwargs: dict[str, object], message: str) -> None:
    base: dict[str, object] = {
        "in_channels": 3,
        "num_classes": 2,
        "depth": 10,
        "widen_factor": 1,
    }
    base.update(kwargs)
    with pytest.raises(InductiveValidationError, match=message):
        WideResNetCifar(**base)


def test_wide_resnet_forward_validation() -> None:
    model = WideResNetCifar(in_channels=3, num_classes=2, depth=10, widen_factor=1)
    with pytest.raises(InductiveValidationError, match="4D torch.Tensor"):
        model.forward_features([1.0, 2.0])
    with pytest.raises(InductiveValidationError, match="4D torch.Tensor"):
        model.forward_features(torch.randn(3, 8, 8))
    with pytest.raises(InductiveValidationError, match="expects 3 input channels"):
        model.forward_features(torch.randn(2, 1, 8, 8))
