from __future__ import annotations

import inspect
import logging
from time import perf_counter
from typing import Any

from modssc.supervised.base import BaseSupervisedClassifier, FitResult
from modssc.supervised.errors import SupervisedValidationError
from modssc.supervised.optional import optional_import

logger = logging.getLogger(__name__)

_PRETRAINED_SENTINEL = object()
_WEIGHTS_MAP = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
    "resnet152": "ResNet152_Weights",
    "mobilenet_v2": "MobileNet_V2_Weights",
    "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
    "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
    "efficientnet_b0": "EfficientNet_B0_Weights",
    "efficientnet_b1": "EfficientNet_B1_Weights",
    "densenet121": "DenseNet121_Weights",
    "vgg16": "VGG16_Weights",
    "vit_b_16": "ViT_B_16_Weights",
}


def _torch():
    return optional_import("torch", extra="vision", feature="supervised:image_pretrained")


def _torchvision():
    return optional_import("torchvision", extra="vision", feature="supervised:image_pretrained")


def _supports_arg(fn: Any, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except Exception:
        return False


def _select_weights(weights_enum: Any, name: str) -> Any:
    if name.upper() == "DEFAULT":
        return weights_enum.DEFAULT
    if hasattr(weights_enum, name):
        return getattr(weights_enum, name)
    if hasattr(weights_enum, name.upper()):
        return getattr(weights_enum, name.upper())
    raise SupervisedValidationError(f"Unknown weights enum: {name!r}")


def _resolve_weights(models: Any, model_name: str, weights: Any) -> Any:
    if weights is None:
        return None
    if not isinstance(weights, str):
        return weights
    name = weights.strip()
    if name.lower() in {"none", "null", "false"}:
        return None
    if hasattr(models, "get_model_weights"):
        weights_enum = models.get_model_weights(model_name)
        return _select_weights(weights_enum, name)
    enum_name = _WEIGHTS_MAP.get(model_name)
    if enum_name and hasattr(models, enum_name):
        weights_enum = getattr(models, enum_name)
        return _select_weights(weights_enum, name)
    return _PRETRAINED_SENTINEL


def _infer_in_channels(model: Any, torch) -> int | None:
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            return int(module.in_channels)
    return None


def _replace_classifier(model: Any, n_classes: int, torch) -> Any:
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
        head = torch.nn.Linear(int(model.fc.in_features), int(n_classes))
        model.fc = head
        return head
    if hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, torch.nn.Linear):
            head = torch.nn.Linear(int(clf.in_features), int(n_classes))
            model.classifier = head
            return head
        if isinstance(clf, torch.nn.Sequential):
            for idx in range(len(clf) - 1, -1, -1):
                if isinstance(clf[idx], torch.nn.Linear):
                    head = torch.nn.Linear(int(clf[idx].in_features), int(n_classes))
                    clf[idx] = head
                    return head
    if hasattr(model, "heads"):
        heads = model.heads
        if hasattr(heads, "head") and isinstance(heads.head, torch.nn.Linear):
            head = torch.nn.Linear(int(heads.head.in_features), int(n_classes))
            heads.head = head
            return head
        if isinstance(heads, torch.nn.Linear):
            head = torch.nn.Linear(int(heads.in_features), int(n_classes))
            model.heads = head
            return head
    if hasattr(model, "head") and isinstance(model.head, torch.nn.Linear):
        head = torch.nn.Linear(int(model.head.in_features), int(n_classes))
        model.head = head
        return head
    raise SupervisedValidationError("Unable to replace classifier head for this torchvision model.")


def _load_model(model_name: str, weights: Any):
    tv = _torchvision()
    models = tv.models
    if hasattr(models, "get_model"):
        weights_obj = _resolve_weights(models, model_name, weights)
        if weights_obj is _PRETRAINED_SENTINEL:
            raise SupervisedValidationError(
                "Unable to resolve pretrained weights for model. Set weights=None for random init."
            )
        return models.get_model(model_name, weights=weights_obj)
    model_fn = getattr(models, model_name, None)
    if model_fn is None:
        raise SupervisedValidationError(f"Unknown torchvision model: {model_name!r}")
    weights_obj = _resolve_weights(models, model_name, weights)
    kwargs: dict[str, Any] = {}
    if weights_obj is _PRETRAINED_SENTINEL:
        if _supports_arg(model_fn, "pretrained"):
            kwargs["pretrained"] = True
        else:
            raise SupervisedValidationError(
                "Unable to resolve pretrained weights for model. Set weights=None for random init."
            )
    elif weights_obj is not None:
        if _supports_arg(model_fn, "weights"):
            kwargs["weights"] = weights_obj
        elif _supports_arg(model_fn, "pretrained"):
            kwargs["pretrained"] = True
        else:
            raise SupervisedValidationError(
                "This torchvision version does not support pretrained weights."
            )
    return model_fn(**kwargs)


class TorchImagePretrainedClassifier(BaseSupervisedClassifier):
    """Torchvision pretrained image classifier (fine-tunable)."""

    classifier_id = "image_pretrained"
    backend = "torch"

    def __init__(
        self,
        *,
        model_name: str = "resnet18",
        weights: str | None = "DEFAULT",
        freeze_backbone: bool = True,
        input_shape: tuple[int, int] | tuple[int, int, int] | None = None,
        input_layout: str = "channels_first",
        auto_channel_repeat: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 10,
        seed: int | None = 0,
        n_jobs: int | None = None,
    ):
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.model_name = str(model_name)
        self.weights = weights
        self.freeze_backbone = bool(freeze_backbone)
        self.input_shape = input_shape
        self.input_layout = str(input_layout)
        self.auto_channel_repeat = bool(auto_channel_repeat)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self._model: Any | None = None
        self._head: Any | None = None
        self._classes_t: Any | None = None
        self._input_shape: tuple[int, int, int] | None = None
        self._expected_in_channels: int | None = None

    @property
    def supports_proba(self) -> bool:
        return True

    def _prepare_X(self, X: Any, torch, *, allow_infer: bool) -> Any:
        if not isinstance(X, torch.Tensor):
            raise SupervisedValidationError(
                "TorchImagePretrainedClassifier requires torch.Tensor X."
            )
        if X.ndim == 4:
            if self.input_layout == "channels_last":
                X4 = X.permute(0, 3, 1, 2)
            elif self.input_layout == "channels_first":
                X4 = X
            else:
                raise SupervisedValidationError(
                    "input_layout must be 'channels_first' or 'channels_last'."
                )
            shape = (int(X4.shape[1]), int(X4.shape[2]), int(X4.shape[3]))
        elif X.ndim == 3:
            X4 = X.unsqueeze(1)
            shape = (1, int(X4.shape[2]), int(X4.shape[3]))
        elif X.ndim == 2:
            if self.input_shape is None and self._input_shape is None:
                raise SupervisedValidationError(
                    "image_pretrained requires 3D/4D inputs or input_shape for 2D features."
                )
            if self._input_shape is None:
                if len(tuple(self.input_shape)) == 2:
                    c, h, w = 1, int(self.input_shape[0]), int(self.input_shape[1])
                elif len(tuple(self.input_shape)) == 3:
                    c, h, w = (
                        int(self.input_shape[0]),
                        int(self.input_shape[1]),
                        int(self.input_shape[2]),
                    )
                else:
                    raise SupervisedValidationError("input_shape must be (H, W) or (C, H, W).")
                self._input_shape = (c, h, w)
            shape = self._input_shape
            if int(X.shape[1]) != int(shape[0] * shape[1] * shape[2]):
                raise SupervisedValidationError("input_shape does not match X feature dimension.")
            X4 = X.reshape(int(X.shape[0]), shape[0], shape[1], shape[2])
        else:
            raise SupervisedValidationError("X must be 2D, 3D, or 4D for image_pretrained.")

        if allow_infer:
            self._input_shape = shape
        elif self._input_shape is not None and shape != self._input_shape:
            raise SupervisedValidationError("X shape does not match fitted input_shape.")

        expected = self._expected_in_channels
        if expected is not None and int(X4.shape[1]) != int(expected):
            if self.auto_channel_repeat and int(X4.shape[1]) == 1 and int(expected) == 3:
                X4 = X4.repeat(1, 3, 1, 1)
            elif self.auto_channel_repeat and int(X4.shape[1]) == 2 and int(expected) == 3:
                raise SupervisedValidationError(
                    "Ambiguous 2-channel input for 3-channel model. "
                    "Automatic zero-padding is disabled for scientific rigor. "
                    "Please check if data is Grayscale+Alpha or similar and preprocess explicitly."
                )
            else:
                raise SupervisedValidationError(
                    f"Model expects {expected} channels, got {int(X4.shape[1])}."
                )
        return X4

    def _set_train_mode(self) -> None:
        if self._model is None or self._head is None:
            return
        if self.freeze_backbone:
            self._model.eval()
            self._head.train()
        else:
            self._model.train()

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug(
            "params model_name=%s weights=%s freeze_backbone=%s input_shape=%s input_layout=%s "
            "auto_channel_repeat=%s lr=%s weight_decay=%s batch_size=%s max_epochs=%s seed=%s",
            self.model_name,
            self.weights,
            self.freeze_backbone,
            self.input_shape,
            self.input_layout,
            self.auto_channel_repeat,
            self.lr,
            self.weight_decay,
            self.batch_size,
            self.max_epochs,
            self.seed,
        )
        torch = _torch()

        if not isinstance(X, torch.Tensor):
            raise SupervisedValidationError(
                "TorchImagePretrainedClassifier requires torch.Tensor X."
            )
        if not isinstance(y, torch.Tensor):
            raise SupervisedValidationError(
                "TorchImagePretrainedClassifier requires torch.Tensor y."
            )
        if self.input_layout not in {"channels_first", "channels_last"}:
            raise SupervisedValidationError(
                "input_layout must be 'channels_first' or 'channels_last'."
            )

        if y.ndim != 1:
            y = y.view(-1)
        if int(self.batch_size) <= 0:
            raise SupervisedValidationError("batch_size must be >= 1.")
        if int(self.max_epochs) <= 0:
            raise SupervisedValidationError("max_epochs must be >= 1.")
        if float(self.lr) <= 0:
            raise SupervisedValidationError("lr must be > 0.")
        if X.numel() == 0:
            raise SupervisedValidationError("X must be non-empty.")
        if X.device != y.device:
            raise SupervisedValidationError("X and y must be on the same device.")

        classes, y_enc = torch.unique(y, sorted=True, return_inverse=True)
        self._classes_t = classes
        self.classes_ = classes.detach().cpu().numpy()

        torch.manual_seed(int(self.seed or 0))

        model = _load_model(self.model_name, self.weights)
        head = _replace_classifier(model, int(classes.numel()), torch)
        model = model.to(X.device)
        self._expected_in_channels = _infer_in_channels(model, torch)
        self._model = model
        self._head = head

        X4 = self._prepare_X(X, torch, allow_infer=True)

        if X4.shape[0] != y.shape[0]:
            raise SupervisedValidationError("X and y must have matching first dimension.")

        if self.freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
            for p in head.parameters():
                p.requires_grad = True
            params = head.parameters()
        else:
            params = model.parameters()

        optimizer = torch.optim.AdamW(
            params, lr=float(self.lr), weight_decay=float(self.weight_decay)
        )

        n = int(X4.shape[0])
        for _epoch in range(int(self.max_epochs)):
            self._set_train_mode()
            order = torch.randperm(n, device=X4.device)
            for i in range(0, n, int(self.batch_size)):
                idx = order[i : i + int(self.batch_size)]
                logits = model(X4[idx].to(dtype=torch.float32))
                loss = torch.nn.functional.cross_entropy(logits, y_enc[idx].to(torch.long))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._fit_result = FitResult(
            n_samples=int(X4.shape[0]),
            n_features=int(X4.shape[1] * X4.shape[2] * X4.shape[3]),
            n_classes=int(classes.numel()),
        )
        logger.info("Finished %s.fit in %.3fs", self.classifier_id, perf_counter() - start)
        return self._fit_result

    def _scores(self, X: Any):
        torch = _torch()
        if self._model is None or self._classes_t is None:
            raise RuntimeError("Model is not fitted")
        X4 = self._prepare_X(X, torch, allow_infer=False)
        if X4.device != self._classes_t.device:
            raise SupervisedValidationError("X must be on the same device as the model.")
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X4.to(dtype=torch.float32))
            return torch.softmax(logits, dim=1)

    def predict_scores(self, X: Any):
        return self._scores(X)

    def predict_proba(self, X: Any):
        return self._scores(X)

    def predict(self, X: Any):
        if self._classes_t is None:
            raise RuntimeError("Model is not fitted")
        scores = self._scores(X)
        idx = scores.argmax(dim=1)
        return self._classes_t[idx]
