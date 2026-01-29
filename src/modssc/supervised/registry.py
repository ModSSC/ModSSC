from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from modssc.supervised.errors import UnknownBackendError, UnknownClassifierError
from modssc.supervised.types import BackendSpec, ClassifierSpec

_REGISTRY: dict[str, ClassifierSpec] = {}
_BOOTSTRAPPED = False


def register_classifier(
    *,
    key: str,
    description: str,
    preferred_backends: tuple[str, ...] = ("sklearn", "numpy"),
) -> None:
    if key in _REGISTRY:
        # allow idempotent registration if identical
        existing = _REGISTRY[key]
        if existing.description != description or existing.preferred_backends != preferred_backends:
            raise ValueError(f"Classifier already registered with different metadata: {key!r}")
        return
    _REGISTRY[key] = ClassifierSpec(
        key=key,
        description=description,
        backends={},
        preferred_backends=preferred_backends,
    )


def register_backend(
    *,
    classifier_id: str,
    backend: str,
    factory: str,
    required_extra: str | None = None,
    supports_gpu: bool = False,
    notes: str = "",
) -> None:
    if classifier_id not in _REGISTRY:
        raise UnknownClassifierError(classifier_id)
    spec = _REGISTRY[classifier_id]
    if backend in spec.backends:
        raise ValueError(f"Backend already registered for {classifier_id!r}: {backend!r}")
    new_backends = dict(spec.backends)
    new_backends[backend] = BackendSpec(
        backend=backend,
        factory=factory,
        required_extra=required_extra,
        supports_gpu=bool(supports_gpu),
        notes=str(notes),
    )
    _REGISTRY[classifier_id] = replace(spec, backends=new_backends)


def ensure_bootstrap() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    # kNN
    register_classifier(
        key="knn",
        description="k-nearest neighbors classifier (classic baseline).",
        preferred_backends=("sklearn", "numpy"),
    )
    register_backend(
        classifier_id="knn",
        backend="numpy",
        factory="modssc.supervised.backends.numpy.knn:NumpyKNNClassifier",
        required_extra=None,
        supports_gpu=False,
        notes="Pure numpy implementation (slow for large n).",
    )
    register_backend(
        classifier_id="knn",
        backend="torch",
        factory="modssc.supervised.backends.torch.knn:TorchKNNClassifier",
        required_extra="supervised-torch",
        supports_gpu=True,
        notes="Torch implementation (CPU/GPU depending on tensors device).",
    )
    register_backend(
        classifier_id="knn",
        backend="sklearn",
        factory="modssc.supervised.backends.sklearn.knn:SklearnKNNClassifier",
        required_extra="sklearn",
        supports_gpu=False,
        notes="Uses scikit-learn KNeighborsClassifier.",
    )

    # SVM with RBF kernel
    register_classifier(
        key="svm_rbf",
        description="Support Vector Machine with RBF kernel (classic baseline).",
        preferred_backends=("sklearn",),
    )
    register_backend(
        classifier_id="svm_rbf",
        backend="sklearn",
        factory="modssc.supervised.backends.sklearn.svm_rbf:SklearnSVRBFClassifier",
        required_extra="sklearn",
        supports_gpu=False,
        notes="Uses scikit-learn SVC(kernel='rbf').",
    )

    # Logistic regression
    register_classifier(
        key="logreg",
        description="Multinomial logistic regression (classic baseline).",
        preferred_backends=("sklearn",),
    )
    register_backend(
        classifier_id="logreg",
        backend="sklearn",
        factory="modssc.supervised.backends.sklearn.logreg:SklearnLogRegClassifier",
        required_extra="sklearn",
        supports_gpu=False,
        notes="Uses scikit-learn LogisticRegression.",
    )
    register_backend(
        classifier_id="logreg",
        backend="torch",
        factory="modssc.supervised.backends.torch.logreg:TorchLogRegClassifier",
        required_extra="supervised-torch",
        supports_gpu=True,
        notes="Torch implementation (CPU/GPU depending on tensors device).",
    )

    # MLP (torch)
    register_classifier(
        key="mlp",
        description="Multilayer perceptron classifier (torch).",
        preferred_backends=("torch",),
    )
    register_backend(
        classifier_id="mlp",
        backend="torch",
        factory="modssc.supervised.backends.torch.mlp:TorchMLPClassifier",
        required_extra="supervised-torch",
        supports_gpu=True,
        notes="Torch MLP for vector features.",
    )

    # Image CNN (torch)
    register_classifier(
        key="image_cnn",
        description="Small CNN for image tensors (torch).",
        preferred_backends=("torch",),
    )
    register_backend(
        classifier_id="image_cnn",
        backend="torch",
        factory="modssc.supervised.backends.torch.image_cnn:TorchImageCNNClassifier",
        required_extra="supervised-torch",
        supports_gpu=True,
        notes="Torch CNN for image inputs (N, C, H, W).",
    )

    # Image pretrained (torchvision)
    register_classifier(
        key="image_pretrained",
        description="Torchvision pretrained image classifier (fine-tunable).",
        preferred_backends=("torch",),
    )
    register_backend(
        classifier_id="image_pretrained",
        backend="torch",
        factory="modssc.supervised.backends.torch.image_pretrained:TorchImagePretrainedClassifier",
        required_extra="vision",
        supports_gpu=True,
        notes="Torchvision pretrained backbone with a replaceable head.",
    )

    # Audio CNN (torch)
    register_classifier(
        key="audio_cnn",
        description="Small 1D CNN for audio tensors (torch).",
        preferred_backends=("torch",),
    )
    register_backend(
        classifier_id="audio_cnn",
        backend="torch",
        factory="modssc.supervised.backends.torch.audio_cnn:TorchAudioCNNClassifier",
        required_extra="supervised-torch",
        supports_gpu=True,
        notes="Torch CNN for audio inputs (N, C, L).",
    )

    # Audio pretrained (torchaudio)
    register_classifier(
        key="audio_pretrained",
        description="Torchaudio pretrained audio classifier (fine-tunable).",
        preferred_backends=("torch",),
    )
    register_backend(
        classifier_id="audio_pretrained",
        backend="torch",
        factory="modssc.supervised.backends.torch.audio_pretrained:TorchAudioPretrainedClassifier",
        required_extra="audio",
        supports_gpu=True,
        notes="Torchaudio pretrained backbone with a linear head.",
    )

    # Text CNN (torch)
    register_classifier(
        key="text_cnn",
        description="Text CNN for sequence embeddings (torch).",
        preferred_backends=("torch",),
    )
    register_backend(
        classifier_id="text_cnn",
        backend="torch",
        factory="modssc.supervised.backends.torch.text_cnn:TorchTextCNNClassifier",
        required_extra="supervised-torch",
        supports_gpu=True,
        notes="Torch CNN for text inputs (N, L, D) or (N, D, L).",
    )

    # Linear SVM
    register_classifier(
        key="linear_svm",
        description="Linear SVM classifier (hinge loss).",
        preferred_backends=("sklearn",),
    )
    register_backend(
        classifier_id="linear_svm",
        backend="sklearn",
        factory="modssc.supervised.backends.sklearn.linear_svm:SklearnLinearSVMClassifier",
        required_extra="sklearn",
        supports_gpu=False,
        notes="Uses scikit-learn LinearSVC (no predict_proba).",
    )

    # Ridge classifier
    register_classifier(
        key="ridge",
        description="Ridge classifier (linear model).",
        preferred_backends=("sklearn",),
    )
    register_backend(
        classifier_id="ridge",
        backend="sklearn",
        factory="modssc.supervised.backends.sklearn.ridge:SklearnRidgeClassifier",
        required_extra="sklearn",
        supports_gpu=False,
        notes="Uses scikit-learn RidgeClassifier (no predict_proba).",
    )

    # Random Forest
    register_classifier(
        key="random_forest",
        description="Random Forest classifier.",
        preferred_backends=("sklearn",),
    )
    register_backend(
        classifier_id="random_forest",
        backend="sklearn",
        factory="modssc.supervised.backends.sklearn.random_forest:SklearnRandomForestClassifier",
        required_extra="sklearn",
        supports_gpu=False,
        notes="Uses scikit-learn RandomForestClassifier.",
    )

    # Extra Trees
    register_classifier(
        key="extra_trees",
        description="Extra Trees classifier.",
        preferred_backends=("sklearn",),
    )
    register_backend(
        classifier_id="extra_trees",
        backend="sklearn",
        factory="modssc.supervised.backends.sklearn.extra_trees:SklearnExtraTreesClassifier",
        required_extra="sklearn",
        supports_gpu=False,
        notes="Uses scikit-learn ExtraTreesClassifier.",
    )

    # Gradient Boosting
    register_classifier(
        key="gradient_boosting",
        description="Gradient Boosting classifier.",
        preferred_backends=("sklearn",),
    )
    register_backend(
        classifier_id="gradient_boosting",
        backend="sklearn",
        factory=(
            "modssc.supervised.backends.sklearn.gradient_boosting:SklearnGradientBoostingClassifier"
        ),
        required_extra="sklearn",
        supports_gpu=False,
        notes="Uses scikit-learn GradientBoostingClassifier.",
    )

    # Naive Bayes
    register_classifier(
        key="gaussian_nb",
        description="Gaussian Naive Bayes classifier.",
        preferred_backends=("sklearn",),
    )
    register_backend(
        classifier_id="gaussian_nb",
        backend="sklearn",
        factory="modssc.supervised.backends.sklearn.naive_bayes:SklearnGaussianNBClassifier",
        required_extra="sklearn",
        supports_gpu=False,
        notes="Uses scikit-learn GaussianNB.",
    )
    register_classifier(
        key="multinomial_nb",
        description="Multinomial Naive Bayes classifier.",
        preferred_backends=("sklearn",),
    )
    register_backend(
        classifier_id="multinomial_nb",
        backend="sklearn",
        factory="modssc.supervised.backends.sklearn.naive_bayes:SklearnMultinomialNBClassifier",
        required_extra="sklearn",
        supports_gpu=False,
        notes="Uses scikit-learn MultinomialNB.",
    )
    register_classifier(
        key="bernoulli_nb",
        description="Bernoulli Naive Bayes classifier.",
        preferred_backends=("sklearn",),
    )
    register_backend(
        classifier_id="bernoulli_nb",
        backend="sklearn",
        factory="modssc.supervised.backends.sklearn.naive_bayes:SklearnBernoulliNBClassifier",
        required_extra="sklearn",
        supports_gpu=False,
        notes="Uses scikit-learn BernoulliNB.",
    )

    register_classifier(
        key="lstm_scratch",
        description="LSTM from scratch for text sequences (Tabula Rasa).",
        preferred_backends=("torch",),
    )
    register_backend(
        classifier_id="lstm_scratch",
        backend="torch",
        factory="modssc.supervised.backends.torch.lstm_scratch:TorchLSTMClassifier",
        required_extra="supervised-torch",
        supports_gpu=True,
        notes="Custom LSTM implementation.",
    )

    register_classifier(
        key="audio_cnn_scratch",
        description="2D CNN for Spectrograms from scratch (Tabula Rasa).",
        preferred_backends=("torch",),
    )
    register_backend(
        classifier_id="audio_cnn_scratch",
        backend="torch",
        factory="modssc.supervised.backends.torch.audio_cnn_scratch:TorchAudioCNNClassifier",
        required_extra="supervised-torch",
        supports_gpu=True,
        notes="Custom 2D CNN implementation.",
    )

    register_classifier(
        key="graphsage_inductive",
        description="GraphSAGE Inductive (Tabula Rasa).",
        preferred_backends=("torch",),
    )
    register_backend(
        classifier_id="graphsage_inductive",
        backend="torch",
        factory="modssc.supervised.backends.torch.graphsage_inductive:TorchGraphSAGEClassifier",
        required_extra="supervised-torch-geometric",
        supports_gpu=True,
        notes="Custom GraphSAGE implementation.",
    )

    _BOOTSTRAPPED = True


def list_classifiers() -> list[str]:
    ensure_bootstrap()
    return sorted(_REGISTRY.keys())


def iter_specs() -> Iterable[ClassifierSpec]:
    ensure_bootstrap()
    # deterministic order
    for key in sorted(_REGISTRY.keys()):
        yield _REGISTRY[key]


def get_spec(classifier_id: str) -> ClassifierSpec:
    ensure_bootstrap()
    if classifier_id not in _REGISTRY:
        raise UnknownClassifierError(classifier_id)
    return _REGISTRY[classifier_id]


def get_backend_spec(classifier_id: str, backend: str) -> BackendSpec:
    spec = get_spec(classifier_id)
    if backend not in spec.backends:
        raise UnknownBackendError(classifier_id, backend)
    return spec.backends[backend]
