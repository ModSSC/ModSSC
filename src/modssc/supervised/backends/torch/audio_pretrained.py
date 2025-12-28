from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from modssc.supervised.base import BaseSupervisedClassifier, FitResult
from modssc.supervised.errors import SupervisedValidationError
from modssc.supervised.optional import optional_import

logger = logging.getLogger(__name__)


def _torch():
    return optional_import("torch", extra="audio", feature="supervised:audio_pretrained")


def _torchaudio():
    return optional_import("torchaudio", extra="audio", feature="supervised:audio_pretrained")


def _load_bundle(bundle_name: str):
    torchaudio = _torchaudio()
    bundle = getattr(torchaudio.pipelines, bundle_name, None)
    if bundle is None:
        raise SupervisedValidationError(f"Unknown torchaudio bundle: {bundle_name!r}")
    return bundle


def _extract_features(model: Any, waveforms: Any, torch) -> Any:
    if hasattr(model, "extract_features"):
        out = model.extract_features(waveforms)
    else:
        out = model(waveforms)

    feats = out[0] if isinstance(out, tuple) else out

    if isinstance(feats, (list, tuple)):
        feats = feats[-1]

    if not isinstance(feats, torch.Tensor):
        raise SupervisedValidationError("Unexpected feature output from audio backbone.")

    if feats.ndim == 3:
        return feats.mean(dim=1)
    if feats.ndim == 2:
        return feats
    raise SupervisedValidationError("Audio backbone returned invalid feature shape.")


class TorchAudioPretrainedClassifier(BaseSupervisedClassifier):
    """Torchaudio pretrained model with a linear head."""

    classifier_id = "audio_pretrained"
    backend = "torch"

    def __init__(
        self,
        *,
        bundle: str = "WAV2VEC2_BASE",
        freeze_backbone: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 16,
        max_epochs: int = 5,
        seed: int | None = 0,
        n_jobs: int | None = None,
    ):
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.bundle = str(bundle)
        self.freeze_backbone = bool(freeze_backbone)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self._backbone: Any | None = None
        self._head: Any | None = None
        self._classes_t: Any | None = None
        self._feature_dim: int | None = None

    @property
    def supports_proba(self) -> bool:
        return True

    def _prepare_X(self, X: Any, torch) -> Any:
        if not isinstance(X, torch.Tensor):
            raise SupervisedValidationError(
                "TorchAudioPretrainedClassifier requires torch.Tensor X."
            )
        if X.ndim == 1:
            X2 = X.view(1, -1)
        elif X.ndim == 2:
            X2 = X
        elif X.ndim == 3:
            if int(X.shape[1]) != 1:
                raise SupervisedValidationError("audio_pretrained expects mono waveforms (C=1).")
            X2 = X[:, 0, :]
        else:
            raise SupervisedValidationError("X must be 1D, 2D, or 3D for audio_pretrained.")
        return X2

    def _set_train_mode(self) -> None:
        if self._backbone is None:
            return
        if self.freeze_backbone:
            self._backbone.eval()
        else:
            self._backbone.train()
        if self._head is not None:
            self._head.train()

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug(
            "params bundle=%s freeze_backbone=%s lr=%s weight_decay=%s batch_size=%s "
            "max_epochs=%s seed=%s",
            self.bundle,
            self.freeze_backbone,
            self.lr,
            self.weight_decay,
            self.batch_size,
            self.max_epochs,
            self.seed,
        )
        torch = _torch()

        if not isinstance(X, torch.Tensor):
            raise SupervisedValidationError(
                "TorchAudioPretrainedClassifier requires torch.Tensor X."
            )
        if not isinstance(y, torch.Tensor):
            raise SupervisedValidationError(
                "TorchAudioPretrainedClassifier requires torch.Tensor y."
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

        bundle = _load_bundle(self.bundle)
        backbone = bundle.get_model().to(X.device)
        self._backbone = backbone

        X2 = self._prepare_X(X, torch)
        if X2.shape[0] != y.shape[0]:
            raise SupervisedValidationError("X and y must have matching first dimension.")

        with torch.no_grad():
            feats = _extract_features(backbone, X2[:1].to(dtype=torch.float32), torch)
        self._feature_dim = int(feats.shape[1])
        head = torch.nn.Linear(int(self._feature_dim), int(classes.numel())).to(X.device)
        self._head = head

        if self.freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False
            params = head.parameters()
        else:
            params = list(backbone.parameters()) + list(head.parameters())

        optimizer = torch.optim.AdamW(
            params, lr=float(self.lr), weight_decay=float(self.weight_decay)
        )

        n = int(X2.shape[0])
        for _epoch in range(int(self.max_epochs)):
            self._set_train_mode()
            order = torch.randperm(n, device=X2.device)
            for i in range(0, n, int(self.batch_size)):
                idx = order[i : i + int(self.batch_size)]
                batch = X2[idx].to(dtype=torch.float32)
                if self.freeze_backbone:
                    with torch.no_grad():
                        feats = _extract_features(backbone, batch, torch)
                else:
                    feats = _extract_features(backbone, batch, torch)
                logits = head(feats)
                loss = torch.nn.functional.cross_entropy(logits, y_enc[idx].to(torch.long))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._fit_result = FitResult(
            n_samples=int(X2.shape[0]),
            n_features=int(self._feature_dim or 0),
            n_classes=int(classes.numel()),
        )
        logger.info("Finished %s.fit in %.3fs", self.classifier_id, perf_counter() - start)
        return self._fit_result

    def _scores(self, X: Any):
        torch = _torch()
        if self._backbone is None or self._head is None or self._classes_t is None:
            raise RuntimeError("Model is not fitted")
        X2 = self._prepare_X(X, torch)
        if X2.device != self._classes_t.device:
            raise SupervisedValidationError("X must be on the same device as the model.")
        self._backbone.eval()
        self._head.eval()
        with torch.no_grad():
            feats = _extract_features(self._backbone, X2.to(dtype=torch.float32), torch)
            logits = self._head(feats)
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
