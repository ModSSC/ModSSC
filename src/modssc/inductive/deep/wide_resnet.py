from __future__ import annotations

import math
from typing import Any, Literal

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.optional import import_torch

torch = import_torch()

WideResNetReference = Literal["standardized", "google_fixmatch", "torchssl"]


def resolve_wide_resnet_reference(
    reference_implementation: str | None = None,
) -> WideResNetReference:
    """Resolve a selectable WRN implementation without changing standardized runs."""

    if reference_implementation is None:
        return "standardized"
    if reference_implementation not in ("standardized", "google_fixmatch", "torchssl"):
        raise InductiveValidationError(
            "reference_implementation must be 'standardized', 'google_fixmatch', or 'torchssl'."
        )
    return reference_implementation  # type: ignore[return-value]


class WideResidualBlock(torch.nn.Module):
    """Pre-activation residual block used by the original FixMatch network."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int,
        activate_before_residual: bool,
        bn_momentum: float,
        bn_eps: float,
        reference_implementation: WideResNetReference = "standardized",
    ) -> None:
        super().__init__()
        if reference_implementation not in ("standardized", "google_fixmatch", "torchssl"):
            raise InductiveValidationError("Unknown WideResidualBlock reference implementation.")
        if (
            reference_implementation == "torchssl"
            and int(stride) != 1
            and int(in_channels) == int(out_channels)
        ):
            raise InductiveValidationError(
                "TorchSSL transition blocks require a channel-changing shortcut."
            )
        self.activate_before_residual = bool(activate_before_residual)
        self.reference_implementation = reference_implementation
        self.equal_in_out = int(in_channels) == int(out_channels)
        # Both pinned implementations use the framework default ``bias=True``
        # for every convolution (TensorFlow ``tf.layers.conv2d`` in FixMatch
        # and explicit ``bias=True`` in TorchSSL).
        conv_bias = True
        self.bn1 = torch.nn.BatchNorm2d(
            int(in_channels),
            eps=float(bn_eps),
            momentum=float(bn_momentum),
        )
        self.conv1 = torch.nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=3,
            stride=int(stride),
            padding=1,
            bias=conv_bias,
        )
        self.bn2 = torch.nn.BatchNorm2d(
            int(out_channels),
            eps=float(bn_eps),
            momentum=float(bn_momentum),
        )
        self.conv2 = torch.nn.Conv2d(
            int(out_channels),
            int(out_channels),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=conv_bias,
        )
        self.shortcut = (
            torch.nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=1,
                stride=int(stride),
                bias=conv_bias,
            )
            if int(in_channels) != int(out_channels)
            or (reference_implementation != "torchssl" and int(stride) != 1)
            else None
        )

    @staticmethod
    def _activate(x: Any) -> Any:
        return torch.nn.functional.leaky_relu(x, negative_slope=0.1)

    def forward(self, x: Any) -> Any:
        activated = self._activate(self.bn1(x))
        if self.reference_implementation == "torchssl":
            if not self.equal_in_out and self.activate_before_residual:
                x = activated
            conv_input = activated if self.equal_in_out else x
            out = self.conv1(conv_input)
            out = self.conv2(self._activate(self.bn2(out)))
            residual = x if self.shortcut is None else self.shortcut(x)
            return residual + out

        residual = activated if self.activate_before_residual else x
        out = self.conv1(activated)
        out = self.conv2(self._activate(self.bn2(out)))
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        return residual + out


class WideResNetCifar(torch.nn.Module):
    """CIFAR Wide ResNet matching the architecture in the FixMatch codebase."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        depth: int = 28,
        widen_factor: int = 2,
        bn_momentum: float = 0.001,
        bn_eps: float = 1e-3,
        input_mean: Any | None = None,
        input_std: Any | None = None,
        reference_implementation: str | None = None,
    ) -> None:
        super().__init__()
        reference = resolve_wide_resnet_reference(reference_implementation)
        depth = int(depth)
        widen_factor = int(widen_factor)
        if depth < 10 or (depth - 4) % 6 != 0:
            raise InductiveValidationError("depth must satisfy depth = 6n + 4 with n >= 1.")
        if widen_factor <= 0:
            raise InductiveValidationError("widen_factor must be > 0.")
        if int(in_channels) <= 0:
            raise InductiveValidationError("in_channels must be > 0.")
        if int(num_classes) <= 0:
            raise InductiveValidationError("num_classes must be > 0.")
        if not (0.0 < float(bn_momentum) <= 1.0):
            raise InductiveValidationError("bn_momentum must be in (0, 1].")
        if float(bn_eps) <= 0.0:
            raise InductiveValidationError("bn_eps must be > 0.")
        if input_mean is None and input_std is not None:
            raise InductiveValidationError("input_mean and input_std must be provided together.")
        if input_mean is not None and input_std is None:
            raise InductiveValidationError("input_mean and input_std must be provided together.")

        blocks_per_stage = (depth - 4) // 6
        channels = (16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor)
        self.in_channels = int(in_channels)
        self.depth = depth
        self.widen_factor = widen_factor
        self.blocks_per_stage = blocks_per_stage
        self.channels = channels
        self.reference_implementation = reference
        if input_mean is None:
            self.register_buffer("input_mean", None)
            self.register_buffer("input_std", None)
        else:
            mean = torch.as_tensor(input_mean, dtype=torch.float32).flatten()
            std = torch.as_tensor(input_std, dtype=torch.float32).flatten()
            if int(mean.numel()) != self.in_channels or int(std.numel()) != self.in_channels:
                raise InductiveValidationError(
                    "input_mean and input_std must contain one value per input channel."
                )
            if bool((std <= 0).any().item()):
                raise InductiveValidationError("input_std values must be > 0.")
            self.register_buffer("input_mean", mean.reshape(1, self.in_channels, 1, 1))
            self.register_buffer("input_std", std.reshape(1, self.in_channels, 1, 1))

        self.stem = torch.nn.Conv2d(
            self.in_channels,
            channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        blocks: list[Any] = []
        current_channels = channels[0]
        residual_bn_eps = float(bn_eps)
        for stage, out_channels in enumerate(channels[1:]):
            for block_index in range(blocks_per_stage):
                stride = 2 if stage > 0 and block_index == 0 else 1
                blocks.append(
                    WideResidualBlock(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        stride=stride,
                        activate_before_residual=stage == 0 and block_index == 0,
                        bn_momentum=float(bn_momentum),
                        bn_eps=residual_bn_eps,
                        reference_implementation=reference,
                    )
                )
                current_channels = out_channels
        self.blocks = torch.nn.Sequential(*blocks)
        self.bn = torch.nn.BatchNorm2d(
            current_channels,
            eps=1e-3 if reference == "torchssl" else float(bn_eps),
            momentum=float(bn_momentum),
        )
        self.classifier = torch.nn.Linear(current_channels, int(num_classes))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                if self.reference_implementation == "torchssl":
                    torch.nn.init.kaiming_normal_(
                        module.weight,
                        mode="fan_out",
                        nonlinearity="leaky_relu",
                    )
                else:
                    kernel_height, kernel_width = module.kernel_size
                    std = math.sqrt(
                        2.0 / (float(kernel_height) * float(kernel_width) * module.out_channels)
                    )
                    torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                # TorchSSL's ``weights_init`` only overwrites convolution
                # weights.  Its Conv2d biases deliberately retain the
                # framework constructor initialization.  Google FixMatch and
                # ModSSC's standardized profile keep their historical
                # zero-bias initialization.
                if module.bias is not None and self.reference_implementation != "torchssl":
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)
                torch.nn.init.zeros_(module.bias)

    @staticmethod
    def _activate(x: Any) -> Any:
        return torch.nn.functional.leaky_relu(x, negative_slope=0.1)

    def forward_features(self, x: Any) -> Any:
        if not isinstance(x, torch.Tensor) or int(x.ndim) != 4:
            raise InductiveValidationError(
                "WideResNetCifar expects a 4D torch.Tensor (N, C, H, W)."
            )
        if int(x.shape[1]) != self.in_channels:
            raise InductiveValidationError(
                f"WideResNetCifar expects {self.in_channels} input channels, got {int(x.shape[1])}."
            )
        if self.input_mean is not None:
            x = (x - self.input_mean.to(dtype=x.dtype)) / self.input_std.to(dtype=x.dtype)
        out = self.stem(x)
        out = self.blocks(out)
        out = self._activate(self.bn(out))
        return torch.nn.functional.adaptive_avg_pool2d(out, output_size=1).flatten(1)

    def forward_head(self, features: Any) -> Any:
        """Apply the classifier head to an already pooled feature tensor."""

        return self.classifier(features)

    def forward(self, x: Any) -> Any:
        return self.forward_head(self.forward_features(x))


__all__ = [
    "WideResNetCifar",
    "WideResNetReference",
    "WideResidualBlock",
    "resolve_wide_resnet_reference",
]
