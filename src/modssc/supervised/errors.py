from __future__ import annotations


class SupervisedError(Exception):
    """Base error for the supervised brick."""


class UnknownClassifierError(SupervisedError):
    def __init__(self, classifier_id: str):
        super().__init__(f"Unknown classifier: {classifier_id!r}")
        self.classifier_id = classifier_id


class UnknownBackendError(SupervisedError):
    def __init__(self, classifier_id: str, backend: str):
        super().__init__(f"Unknown backend {backend!r} for classifier {classifier_id!r}")
        self.classifier_id = classifier_id
        self.backend = backend


class OptionalDependencyError(SupervisedError):
    """Raised when an optional extra is required but not installed."""

    def __init__(self, *, extra: str, feature: str):
        super().__init__(
            f'Optional dependency missing for {feature!r}. Install with: pip install "modssc[{extra}]"'
        )
        self.extra = extra
        self.feature = feature


class NotSupportedError(SupervisedError):
    def __init__(self, message: str):
        super().__init__(message)


class SupervisedValidationError(SupervisedError):
    def __init__(self, message: str):
        super().__init__(message)
