from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.optional import optional_import

from .types import TorchModelBundle


def _torch():
    return optional_import("torch", extra="inductive-torch")


def _normalize_hidden_sizes(
    hidden_sizes: Any | None, hidden_dim: Any | None, *, default: Sequence[int]
) -> tuple[int, ...]:
    if hidden_sizes is None:
        hidden_sizes = default if hidden_dim is None else (int(hidden_dim),)
    if isinstance(hidden_sizes, int):
        return (int(hidden_sizes),)
    if isinstance(hidden_sizes, (list, tuple)):
        return tuple(int(h) for h in hidden_sizes)
    raise InductiveValidationError("hidden_sizes must be an int or a sequence of ints.")


def _make_activation(name: str, torch):
    if name == "relu":
        return torch.nn.ReLU()
    if name == "gelu":
        return torch.nn.GELU()
    if name == "tanh":
        return torch.nn.Tanh()
    raise InductiveValidationError(f"Unknown activation: {name!r}")


def _infer_input_dim(sample: Any) -> int:
    torch = _torch()
    if not isinstance(sample, torch.Tensor):
        raise InductiveValidationError("Torch model bundle requires torch.Tensor features.")
    if int(sample.ndim) == 0:
        raise InductiveValidationError("Sample must have at least one dimension.")
    if int(sample.ndim) == 1:
        return int(sample.numel())
    return int(sample[0].numel())


def _take_sample(x: Any) -> Any:
    torch = _torch()
    if not isinstance(x, torch.Tensor):
        return x
    if int(x.ndim) >= 1 and int(x.shape[0]) > 1:
        return x[:1]
    return x


def _maybe_ema(model: Any, *, enabled: bool) -> Any | None:
    if not enabled:
        return None
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


def _build_mlp_bundle(
    sample: Any,
    *,
    num_classes: int,
    params: Mapping[str, Any],
    seed: int,
    ema: bool,
    force_hidden_sizes: tuple[int, ...] | None = None,
) -> TorchModelBundle:
    torch = _torch()
    input_dim = _infer_input_dim(sample)
    if force_hidden_sizes is None:
        hidden_sizes = _normalize_hidden_sizes(
            params.get("hidden_sizes"),
            params.get("hidden_dim"),
            default=(256, 128),
        )
    else:
        hidden_sizes = tuple(force_hidden_sizes)
    activation = str(params.get("activation", "relu"))
    dropout = float(params.get("dropout", 0.1))
    lr = float(params.get("lr", 1e-3))
    weight_decay = float(params.get("weight_decay", 0.0))

    torch.manual_seed(int(seed))
    layers: list[Any] = []
    in_features = int(input_dim)
    for h in hidden_sizes:
        if int(h) <= 0:
            raise InductiveValidationError("hidden_sizes must be positive.")
        layers.append(torch.nn.Linear(in_features, int(h)))
        layers.append(_make_activation(activation, torch))
        if float(dropout) > 0.0:
            layers.append(torch.nn.Dropout(p=float(dropout)))
        in_features = int(h)
    layers.append(torch.nn.Linear(in_features, int(num_classes)))

    model = torch.nn.Sequential(*layers).to(sample.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    ema_model = _maybe_ema(model, enabled=ema)
    return TorchModelBundle(model=model, optimizer=optimizer, ema_model=ema_model)


def _build_mlp_feature_bundle(
    sample: Any,
    *,
    num_classes: int,
    params: Mapping[str, Any],
    seed: int,
    ema: bool,
    force_hidden_sizes: tuple[int, ...] | None = None,
) -> TorchModelBundle:
    torch = _torch()
    input_dim = _infer_input_dim(sample)
    if force_hidden_sizes is None:
        hidden_sizes = _normalize_hidden_sizes(
            params.get("hidden_sizes"),
            params.get("hidden_dim"),
            default=(256, 128),
        )
    else:
        hidden_sizes = tuple(force_hidden_sizes)
    activation = str(params.get("activation", "relu"))
    dropout = float(params.get("dropout", 0.1))
    lr = float(params.get("lr", 1e-3))
    weight_decay = float(params.get("weight_decay", 0.0))

    class _MLPWithFeatures(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            in_features = int(input_dim)
            for h in hidden_sizes:
                if int(h) <= 0:
                    raise InductiveValidationError("hidden_sizes must be positive.")
                layers.append(torch.nn.Linear(in_features, int(h)))
                layers.append(_make_activation(activation, torch))
                if float(dropout) > 0.0:
                    layers.append(torch.nn.Dropout(p=float(dropout)))
                in_features = int(h)
            self.backbone = torch.nn.Sequential(*layers) if layers else None
            self.head = torch.nn.Linear(in_features, int(num_classes))

        def forward(self, x: Any) -> Any:
            feats = self.backbone(x) if self.backbone is not None else x
            logits = self.head(feats)
            return {"logits": logits, "feat": feats}

    torch.manual_seed(int(seed))
    model = _MLPWithFeatures().to(sample.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    ema_model = _maybe_ema(model, enabled=ema)
    return TorchModelBundle(model=model, optimizer=optimizer, ema_model=ema_model)


def _infer_image_shape(sample: Any, *, input_shape: Any | None) -> tuple[int, int, int]:
    torch = _torch()
    if not isinstance(sample, torch.Tensor):
        raise InductiveValidationError("image_cnn requires torch.Tensor features.")
    if int(sample.ndim) == 4:
        return (int(sample.shape[1]), int(sample.shape[2]), int(sample.shape[3]))
    if int(sample.ndim) == 3:
        return (1, int(sample.shape[1]), int(sample.shape[2]))
    if int(sample.ndim) == 2:
        if input_shape is None:
            raise InductiveValidationError("image_cnn requires input_shape for 2D features.")
        shape = tuple(int(s) for s in input_shape)
        if len(shape) == 2:
            return (1, int(shape[0]), int(shape[1]))
        if len(shape) == 3:
            return (int(shape[0]), int(shape[1]), int(shape[2]))
        raise InductiveValidationError("input_shape must be (H, W) or (C, H, W).")
    raise InductiveValidationError("image_cnn requires 2D, 3D, or 4D inputs.")


def _build_image_cnn_bundle(
    sample: Any,
    *,
    num_classes: int,
    params: Mapping[str, Any],
    seed: int,
    ema: bool,
) -> TorchModelBundle:
    from modssc.supervised.backends.torch import image_cnn as image_cnn_backend

    torch = _torch()
    conv_channels = tuple(int(c) for c in params.get("conv_channels", (32, 64)))
    kernel_size = int(params.get("kernel_size", 3))
    fc_dim = int(params.get("fc_dim", 128))
    activation = str(params.get("activation", "relu"))
    dropout = float(params.get("dropout", 0.2))
    lr = float(params.get("lr", 1e-3))
    weight_decay = float(params.get("weight_decay", 0.0))
    input_shape = params.get("input_shape")

    torch.manual_seed(int(seed))
    in_channels, _h, _w = _infer_image_shape(sample, input_shape=input_shape)

    class _InductiveImageCNN(image_cnn_backend._ImageCNN):
        def forward(self, x: Any):
            x = self.conv(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return self.head(x), x

    model = _InductiveImageCNN(
        in_channels=in_channels,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        activation=activation,
        dropout=dropout,
        fc_dim=fc_dim,
        n_classes=int(num_classes),
    ).to(sample.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    ema_model = _maybe_ema(model, enabled=ema)
    return TorchModelBundle(model=model, optimizer=optimizer, ema_model=ema_model)


def _infer_audio_channels(sample: Any, *, input_shape: Any | None) -> int:
    torch = _torch()
    if not isinstance(sample, torch.Tensor):
        raise InductiveValidationError("audio_cnn requires torch.Tensor features.")
    if int(sample.ndim) == 3:
        return int(sample.shape[1])
    if int(sample.ndim) in (1, 2):
        if (
            input_shape is not None
            and isinstance(input_shape, (list, tuple))
            and len(input_shape) == 2
        ):
            return int(input_shape[0])
        return 1
    raise InductiveValidationError("audio_cnn requires 1D, 2D, or 3D inputs.")


def _build_audio_cnn_bundle(
    sample: Any,
    *,
    num_classes: int,
    params: Mapping[str, Any],
    seed: int,
    ema: bool,
) -> TorchModelBundle:
    from modssc.supervised.backends.torch import audio_cnn as audio_cnn_backend

    torch = _torch()
    conv_channels = tuple(int(c) for c in params.get("conv_channels", (16, 32)))
    kernel_size = int(params.get("kernel_size", 5))
    fc_dim = int(params.get("fc_dim", 64))
    activation = str(params.get("activation", "relu"))
    dropout = float(params.get("dropout", 0.2))
    lr = float(params.get("lr", 1e-3))
    weight_decay = float(params.get("weight_decay", 0.0))
    input_shape = params.get("input_shape")

    torch.manual_seed(int(seed))
    in_channels = _infer_audio_channels(sample, input_shape=input_shape)
    model = audio_cnn_backend._AudioCNN(
        in_channels=in_channels,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        activation=activation,
        dropout=dropout,
        fc_dim=fc_dim,
        n_classes=int(num_classes),
    ).to(sample.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    ema_model = _maybe_ema(model, enabled=ema)
    return TorchModelBundle(model=model, optimizer=optimizer, ema_model=ema_model)


def _infer_text_shape(sample: Any, *, input_layout: str) -> tuple[int, int]:
    torch = _torch()
    if not isinstance(sample, torch.Tensor):
        raise InductiveValidationError("text_cnn requires torch.Tensor features.")
    if int(sample.ndim) == 3:
        if input_layout == "channels_last":
            return (int(sample.shape[2]), int(sample.shape[1]))
        if input_layout == "channels_first":
            return (int(sample.shape[1]), int(sample.shape[2]))
        raise InductiveValidationError("input_layout must be 'channels_last' or 'channels_first'.")
    if int(sample.ndim) == 2:
        return (1, int(sample.shape[1]))
    raise InductiveValidationError("text_cnn requires 2D or 3D inputs.")


def _build_text_cnn_bundle(
    sample: Any,
    *,
    num_classes: int,
    params: Mapping[str, Any],
    seed: int,
    ema: bool,
) -> TorchModelBundle:
    from modssc.supervised.backends.torch import text_cnn as text_cnn_backend

    torch = _torch()
    kernel_sizes = tuple(int(k) for k in params.get("kernel_sizes", (3, 4, 5)))
    num_filters = int(params.get("num_filters", 100))
    activation = str(params.get("activation", "relu"))
    dropout = float(params.get("dropout", 0.5))
    lr = float(params.get("lr", 1e-3))
    weight_decay = float(params.get("weight_decay", 0.0))
    input_layout = str(params.get("input_layout", "channels_last"))

    torch.manual_seed(int(seed))
    in_channels, seq_len = _infer_text_shape(sample, input_layout=input_layout)
    usable = tuple(k for k in kernel_sizes if int(k) <= int(seq_len))
    if not usable:
        raise InductiveValidationError("text_cnn kernel_sizes are larger than sequence length.")
    model = text_cnn_backend._TextCNN(
        in_channels=in_channels,
        kernel_sizes=usable,
        num_filters=num_filters,
        activation=activation,
        dropout=dropout,
        n_classes=int(num_classes),
    ).to(sample.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    ema_model = _maybe_ema(model, enabled=ema)
    return TorchModelBundle(model=model, optimizer=optimizer, ema_model=ema_model)


def _parse_image_input_shape(input_shape: Any | None) -> tuple[int, int, int] | None:
    if input_shape is None:
        return None
    if not isinstance(input_shape, (list, tuple)):
        raise InductiveValidationError("input_shape must be a list or tuple.")
    shape = tuple(int(s) for s in input_shape)
    if len(shape) == 2:
        return (1, int(shape[0]), int(shape[1]))
    if len(shape) == 3:
        return (int(shape[0]), int(shape[1]), int(shape[2]))
    raise InductiveValidationError("input_shape must be (H, W) or (C, H, W).")


def _build_image_pretrained_bundle(
    sample: Any,
    *,
    num_classes: int,
    params: Mapping[str, Any],
    seed: int,
    ema: bool,
) -> TorchModelBundle:
    from modssc.supervised.backends.torch import image_pretrained as image_pretrained_backend

    torch = _torch()

    class _ImagePretrainedWrapper(torch.nn.Module):
        def __init__(
            self,
            model: Any,
            head: Any,
            *,
            input_layout: str,
            input_shape: tuple[int, int, int] | None,
            expected_in_channels: int | None,
            auto_channel_repeat: bool,
            freeze_backbone: bool,
        ) -> None:
            super().__init__()
            self.model = model
            self.head = head
            self.input_layout = input_layout
            self.input_shape = input_shape
            self.expected_in_channels = expected_in_channels
            self.auto_channel_repeat = bool(auto_channel_repeat)
            self.freeze_backbone = bool(freeze_backbone)

        def _prepare(self, X: Any) -> Any:
            if not isinstance(X, torch.Tensor):
                raise InductiveValidationError("image_pretrained requires torch.Tensor input.")
            if X.ndim == 4:
                if self.input_layout == "channels_last":
                    X4 = X.permute(0, 3, 1, 2)
                elif self.input_layout == "channels_first":
                    X4 = X
                else:
                    raise InductiveValidationError(
                        "input_layout must be 'channels_first' or 'channels_last'."
                    )
            elif X.ndim == 3:
                X4 = X.unsqueeze(1)
            elif X.ndim == 2:
                if self.input_shape is None:
                    raise InductiveValidationError(
                        "image_pretrained requires input_shape for 2D features."
                    )
                c, h, w = self.input_shape
                if int(X.shape[1]) != int(c * h * w):
                    raise InductiveValidationError(
                        "input_shape does not match X feature dimension."
                    )
                X4 = X.reshape(int(X.shape[0]), int(c), int(h), int(w))
            else:
                raise InductiveValidationError("image_pretrained requires 2D, 3D, or 4D inputs.")

            expected = self.expected_in_channels
            if expected is not None and int(X4.shape[1]) != int(expected):
                if self.auto_channel_repeat and int(X4.shape[1]) == 1 and int(expected) == 3:
                    X4 = X4.repeat(1, 3, 1, 1)
                else:
                    raise InductiveValidationError(
                        f"Model expects {expected} channels, got {int(X4.shape[1])}."
                    )
            return X4.to(dtype=torch.float32)

        def train(self, mode: bool = True):
            super().train(mode)
            if self.freeze_backbone:
                self.model.eval()
                self.head.train(mode)
            return self

        def forward(self, X: Any):
            X4 = self._prepare(X)
            if not return_features:
                return self.model(X4)
            captured: dict[str, Any] = {}

            def _capture(_module, inputs, _output):
                if inputs:
                    captured["feat"] = inputs[0]

            hook = self.head.register_forward_hook(_capture)
            try:
                logits = self.model(X4)
            finally:
                hook.remove()
            feat = captured.get("feat")
            if feat is None:
                if isinstance(logits, torch.Tensor):
                    feat = logits
                else:
                    raise InductiveValidationError(
                        "image_pretrained return_features failed to capture features."
                    )
            return {"logits": logits, "feat": feat}

    model_name = str(params.get("model_name", "resnet18"))
    weights = params.get("weights", "DEFAULT")
    freeze_backbone = bool(params.get("freeze_backbone", True))
    input_layout = str(params.get("input_layout", "channels_first"))
    auto_channel_repeat = bool(params.get("auto_channel_repeat", True))
    input_shape = _parse_image_input_shape(params.get("input_shape"))
    return_features = bool(params.get("return_features", False))
    lr = float(params.get("lr", 1e-4))
    weight_decay = float(params.get("weight_decay", 1e-4))

    torch.manual_seed(int(seed))
    model = image_pretrained_backend._load_model(model_name, weights)
    head = image_pretrained_backend._replace_classifier(model, int(num_classes), torch)
    expected = image_pretrained_backend._infer_in_channels(model, torch)

    model = model.to(sample.device)
    wrapper = _ImagePretrainedWrapper(
        model,
        head,
        input_layout=input_layout,
        input_shape=input_shape,
        expected_in_channels=expected,
        auto_channel_repeat=auto_channel_repeat,
        freeze_backbone=freeze_backbone,
    ).to(sample.device)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
        params_to_opt = head.parameters()
    else:
        params_to_opt = model.parameters()

    optimizer = torch.optim.AdamW(params_to_opt, lr=float(lr), weight_decay=float(weight_decay))
    ema_model = _maybe_ema(wrapper, enabled=ema)
    return TorchModelBundle(model=wrapper, optimizer=optimizer, ema_model=ema_model)


def _prepare_audio_input(x: Any, torch):
    if x.ndim == 1:
        return x.view(1, -1)
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        if int(x.shape[1]) != 1:
            raise InductiveValidationError("audio_pretrained expects mono waveforms (C=1).")
        return x[:, 0, :]
    if x.ndim == 4 and int(x.shape[1]) == 1:
        # Treat as (N, H, W) e.g. spectrogram
        return x[:, 0, :, :]
    raise InductiveValidationError(
        "audio_pretrained requires 1D, 2D, or 3D inputs (4D with C=1 allowed)."
    )


def _build_audio_pretrained_bundle(
    sample: Any,
    *,
    num_classes: int,
    params: Mapping[str, Any],
    seed: int,
    ema: bool,
) -> TorchModelBundle:
    from modssc.supervised.backends.torch import audio_pretrained as audio_pretrained_backend

    torch = _torch()

    class _AudioPretrainedWrapper(torch.nn.Module):
        def __init__(self, backbone: Any, head: Any, *, freeze_backbone: bool) -> None:
            super().__init__()
            self.backbone = backbone
            self.head = head
            self.freeze_backbone = bool(freeze_backbone)

        def train(self, mode: bool = True):
            super().train(mode)
            self.backbone.train(mode)
            if self.freeze_backbone:
                self.backbone.eval()
                self.head.train(mode)
            return self

        def forward(self, X: Any):
            X2 = _prepare_audio_input(X, torch).to(dtype=torch.float32)
            # We must NOT use torch.no_grad() even if backbone is frozen, purely to allow
            # input gradients (e.g. for VAT) to flow through the backbone.
            # Freezing is handled by requires_grad=False on parameters.
            feats = audio_pretrained_backend._extract_features(self.backbone, X2, torch)
            logits = self.head(feats)
            if return_features:
                return {"logits": logits, "feat": feats}
            return logits

    bundle_name = str(params.get("bundle", "WAV2VEC2_BASE"))
    freeze_backbone = bool(params.get("freeze_backbone", True))
    return_features = bool(params.get("return_features", False))
    lr = float(params.get("lr", 1e-4))
    weight_decay = float(params.get("weight_decay", 1e-4))

    torch.manual_seed(int(seed))
    bundle = audio_pretrained_backend._load_bundle(bundle_name)
    backbone = bundle.get_model().to(sample.device)

    sample_x = _prepare_audio_input(_take_sample(sample), torch).to(dtype=torch.float32)
    with torch.no_grad():
        feats = audio_pretrained_backend._extract_features(backbone, sample_x, torch)
    feature_dim = int(feats.shape[1])
    head = torch.nn.Linear(feature_dim, int(num_classes)).to(sample.device)
    wrapper = _AudioPretrainedWrapper(
        backbone,
        head,
        freeze_backbone=freeze_backbone,
    ).to(sample.device)

    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
        params_to_opt = head.parameters()
    else:
        params_to_opt = list(backbone.parameters()) + list(head.parameters())

    optimizer = torch.optim.AdamW(params_to_opt, lr=float(lr), weight_decay=float(weight_decay))
    ema_model = _maybe_ema(wrapper, enabled=ema)
    return TorchModelBundle(model=wrapper, optimizer=optimizer, ema_model=ema_model)


def _build_lstm_bundle(
    sample: Any,
    *,
    num_classes: int,
    params: Mapping[str, Any],
    seed: int,
    ema: bool,
) -> TorchModelBundle:
    torch = _torch()
    sample = _take_sample(sample)

    vocab_size = int(params.get("vocab_size", 0))
    if vocab_size <= 0:
        # Try to guess from sample if integer
        vocab_size = int(sample.max().item()) + 1 if not sample.is_floating_point() else 20000

    embed_dim = int(params.get("embed_dim", 128))
    hidden_dim = int(params.get("hidden_dim", 128))
    num_layers = int(params.get("num_layers", 1))
    dropout = float(params.get("dropout", 0.0))
    lr = float(params.get("lr", 1e-3))
    weight_decay = float(params.get("weight_decay", 0.0))
    bidirectional = bool(params.get("bidirectional", True))

    class _LSTMClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
            self.lstm = torch.nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
            d_out = hidden_dim * 2 if bidirectional else hidden_dim
            self.fc = torch.nn.Linear(d_out, num_classes)

        def forward(self, x):
            x = x.to(dtype=torch.long)  # Ensure indices
            emb = self.embedding(x)
            self.lstm.flatten_parameters()
            _, (h_n, _) = self.lstm(emb)

            if bidirectional:
                # Concat fwd and bwd from last layer
                # h_n shape: (layers*2, N, H)
                # Take last two
                idx_fwd = -2
                idx_bwd = -1
                h = torch.cat([h_n[idx_fwd], h_n[idx_bwd]], dim=1)
            else:
                h = h_n[-1]

            return self.fc(h)

    torch.manual_seed(int(seed))
    model = _LSTMClassifier().to(sample.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ema_model = _maybe_ema(model, enabled=ema)
    return TorchModelBundle(model=model, optimizer=optimizer, ema_model=ema_model)


def _build_graphsage_bundle(
    sample: Any,
    *,
    num_classes: int,
    params: Mapping[str, Any],
    seed: int,
    ema: bool,
) -> TorchModelBundle:
    from modssc.supervised.optional import optional_import

    torch = _torch()
    # Ensure PyG is available
    optional_import(
        "torch", extra="supervised-torch-geometric", feature="supervised:graphsage_inductive"
    )

    try:
        from torch_geometric.nn import SAGEConv
    except ImportError as e:
        raise ImportError("torch_geometric is required for graphsage_inductive") from e

    hidden_sizes = params.get("hidden_sizes")
    hidden_channels = int(params.get("hidden_channels", 128))
    num_layers = int(params.get("num_layers", 2))
    if hidden_sizes is not None:
        if isinstance(hidden_sizes, int):
            hidden_sizes = [int(hidden_sizes)]
        elif isinstance(hidden_sizes, (list, tuple)):
            hidden_sizes = [int(h) for h in hidden_sizes]
        else:
            raise InductiveValidationError("hidden_sizes must be an int or a sequence of ints.")
        if any(h <= 0 for h in hidden_sizes):
            raise InductiveValidationError("hidden_sizes must be positive.")
        if "num_layers" in params and num_layers != len(hidden_sizes) + 1:
            raise InductiveValidationError(
                "num_layers must equal len(hidden_sizes) + 1 when hidden_sizes is provided."
            )
        num_layers = len(hidden_sizes) + 1
    dropout = float(params.get("dropout", 0.5))
    lr = float(params.get("lr", 1e-2))
    weight_decay = float(params.get("weight_decay", 5e-4))

    # Sample might be a Tensor (x) or a Dict.
    device = None
    if isinstance(sample, dict) and "x" in sample:
        in_channels = sample["x"].shape[-1]
        device = sample["x"].device
    elif hasattr(sample, "shape"):
        in_channels = sample.shape[-1]
        device = sample.device
    else:
        raise InductiveValidationError(
            "GraphSAGE requires sample with shape (tensor) or dict with 'x'."
        )

    class _GraphSAGEWrapper(torch.nn.Module):
        def __init__(self, layer_sizes: list[int], dropout: float):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            for in_channels, out_channels in zip(layer_sizes[:-1], layer_sizes[1:], strict=False):
                self.convs.append(SAGEConv(in_channels, out_channels))
            self.dropout = dropout

        def forward(self, x: Any):
            # x is expected to be a dict with 'x' and 'edge_index'
            if isinstance(x, dict):
                h = x["x"]
                edge_index = x["edge_index"]
            else:
                raise ValueError(
                    f"GraphSAGEWrapper expects a dict input with 'x' and 'edge_index', got {type(x)}."
                )

            for _i, conv in enumerate(self.convs[:-1]):
                h = conv(h, edge_index)
                h = h.relu()
                h = torch.nn.functional.dropout(h, p=self.dropout, training=self.training)

            feat = h
            logits = self.convs[-1](h, edge_index)
            return {"logits": logits, "feat": feat}

    torch.manual_seed(int(seed))
    if hidden_sizes is not None:
        layer_sizes = [int(in_channels), *hidden_sizes, int(num_classes)]
    else:
        layer_sizes = [int(in_channels)] + [hidden_channels] * (num_layers - 1) + [int(num_classes)]
    model = _GraphSAGEWrapper(layer_sizes=layer_sizes, dropout=dropout).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    ema_model = _maybe_ema(model, enabled=ema)
    return TorchModelBundle(model=model, optimizer=optimizer, ema_model=ema_model)


def build_torch_bundle_from_classifier(
    *,
    classifier_id: str,
    classifier_backend: str | None,
    classifier_params: Mapping[str, Any] | None,
    sample: Any,
    num_classes: int | None = None,
    seed: int = 0,
    ema: bool = True,
) -> TorchModelBundle:
    backend = str(classifier_backend or "auto")
    if backend == "auto":
        backend = "torch"
    if backend != "torch":
        raise InductiveValidationError(
            f"Deep model bundle requires classifier_backend='torch' (got {backend!r})."
        )

    params = dict(classifier_params or {})
    return_features = bool(params.get("return_features", False))
    torch = _torch()
    sample = _take_sample(sample)
    if not isinstance(sample, torch.Tensor) and not (isinstance(sample, dict) and "x" in sample):
        # Allow dict only if it looks like graph data (has "x")
        raise InductiveValidationError(
            "Torch model bundle requires torch.Tensor features or Graph Dict."
        )
    if num_classes is None:
        raise InductiveValidationError("num_classes must be provided for torch bundles.")
    num_classes = int(num_classes)
    if num_classes <= 0:
        raise InductiveValidationError("num_classes must be > 0.")

    key = str(classifier_id)
    if key == "mlp":
        if return_features:
            return _build_mlp_feature_bundle(
                sample,
                num_classes=num_classes,
                params=params,
                seed=seed,
                ema=ema,
            )
        return _build_mlp_bundle(
            sample,
            num_classes=num_classes,
            params=params,
            seed=seed,
            ema=ema,
        )
    if key == "logreg":
        if return_features:
            return _build_mlp_feature_bundle(
                sample,
                num_classes=num_classes,
                params=params,
                seed=seed,
                ema=ema,
                force_hidden_sizes=(),
            )
        return _build_mlp_bundle(
            sample,
            num_classes=num_classes,
            params=params,
            seed=seed,
            ema=ema,
            force_hidden_sizes=(),
        )
    if key == "image_cnn":
        return _build_image_cnn_bundle(
            sample,
            num_classes=num_classes,
            params=params,
            seed=seed,
            ema=ema,
        )
    if key == "audio_cnn":
        return _build_audio_cnn_bundle(
            sample,
            num_classes=num_classes,
            params=params,
            seed=seed,
            ema=ema,
        )
    if key == "text_cnn":
        return _build_text_cnn_bundle(
            sample,
            num_classes=num_classes,
            params=params,
            seed=seed,
            ema=ema,
        )
    if key == "image_pretrained":
        return _build_image_pretrained_bundle(
            sample,
            num_classes=num_classes,
            params=params,
            seed=seed,
            ema=ema,
        )
    if key == "audio_pretrained":
        return _build_audio_pretrained_bundle(
            sample,
            num_classes=num_classes,
            params=params,
            seed=seed,
            ema=ema,
        )
    if key == "lstm_scratch":
        return _build_lstm_bundle(
            sample,
            num_classes=num_classes,
            params=params,
            seed=seed,
            ema=ema,
        )
    if key == "graphsage_inductive":
        return _build_graphsage_bundle(
            sample,
            num_classes=num_classes,
            params=params,
            seed=seed,
            ema=ema,
        )

    raise InductiveValidationError(
        f"Unsupported torch classifier_id for deep bundle: {classifier_id!r}."
    )
