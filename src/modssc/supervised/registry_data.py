from __future__ import annotations

BUILTIN_CLASSIFIERS = [
    {
        "key": "knn",
        "description": "k-nearest neighbors classifier (classic baseline).",
        "preferred_backends": ("sklearn", "numpy"),
        "backends": [
            {
                "backend": "numpy",
                "factory": "modssc.supervised.backends.numpy.knn:NumpyKNNClassifier",
                "required_extra": None,
                "supports_gpu": False,
                "notes": "Pure numpy implementation (slow for large n).",
            },
            {
                "backend": "torch",
                "factory": "modssc.supervised.backends.torch.knn:TorchKNNClassifier",
                "required_extra": "supervised-torch",
                "supports_gpu": True,
                "notes": "Torch implementation (CPU/GPU depending on tensors device).",
            },
            {
                "backend": "sklearn",
                "factory": "modssc.supervised.backends.sklearn.knn:SklearnKNNClassifier",
                "required_extra": "sklearn",
                "supports_gpu": False,
                "notes": "Uses scikit-learn KNeighborsClassifier.",
            },
        ],
    },
    {
        "key": "svm_rbf",
        "description": "Support Vector Machine with RBF kernel (classic baseline).",
        "preferred_backends": ("sklearn",),
        "backends": [
            {
                "backend": "sklearn",
                "factory": "modssc.supervised.backends.sklearn.svm_rbf:SklearnSVRBFClassifier",
                "required_extra": "sklearn",
                "supports_gpu": False,
                "notes": "Uses scikit-learn SVC(kernel='rbf').",
            },
        ],
    },
    {
        "key": "logreg",
        "description": "Multinomial logistic regression (classic baseline).",
        "preferred_backends": ("sklearn",),
        "backends": [
            {
                "backend": "sklearn",
                "factory": "modssc.supervised.backends.sklearn.logreg:SklearnLogRegClassifier",
                "required_extra": "sklearn",
                "supports_gpu": False,
                "notes": "Uses scikit-learn LogisticRegression.",
            },
            {
                "backend": "torch",
                "factory": "modssc.supervised.backends.torch.logreg:TorchLogRegClassifier",
                "required_extra": "supervised-torch",
                "supports_gpu": True,
                "notes": "Torch implementation (CPU/GPU depending on tensors device).",
            },
        ],
    },
    {
        "key": "mlp",
        "description": "Multilayer perceptron classifier (torch).",
        "preferred_backends": ("torch",),
        "backends": [
            {
                "backend": "torch",
                "factory": "modssc.supervised.backends.torch.mlp:TorchMLPClassifier",
                "required_extra": "supervised-torch",
                "supports_gpu": True,
                "notes": "Torch MLP for vector features.",
            },
        ],
    },
    {
        "key": "image_cnn",
        "description": "Small CNN for image tensors (torch).",
        "preferred_backends": ("torch",),
        "backends": [
            {
                "backend": "torch",
                "factory": "modssc.supervised.backends.torch.image_cnn:TorchImageCNNClassifier",
                "required_extra": "supervised-torch",
                "supports_gpu": True,
                "notes": "Torch CNN for image inputs (N, C, H, W).",
            },
        ],
    },
    {
        "key": "image_pretrained",
        "description": "Torchvision pretrained image classifier (fine-tunable).",
        "preferred_backends": ("torch",),
        "backends": [
            {
                "backend": "torch",
                "factory": "modssc.supervised.backends.torch.image_pretrained:TorchImagePretrainedClassifier",
                "required_extra": "vision",
                "supports_gpu": True,
                "notes": "Torchvision pretrained backbone with a replaceable head.",
            },
        ],
    },
    {
        "key": "audio_cnn",
        "description": "Small 1D CNN for audio tensors (torch).",
        "preferred_backends": ("torch",),
        "backends": [
            {
                "backend": "torch",
                "factory": "modssc.supervised.backends.torch.audio_cnn:TorchAudioCNNClassifier",
                "required_extra": "supervised-torch",
                "supports_gpu": True,
                "notes": "Torch CNN for audio inputs (N, C, L).",
            },
        ],
    },
    {
        "key": "audio_pretrained",
        "description": "Torchaudio pretrained audio classifier (fine-tunable).",
        "preferred_backends": ("torch",),
        "backends": [
            {
                "backend": "torch",
                "factory": "modssc.supervised.backends.torch.audio_pretrained:TorchAudioPretrainedClassifier",
                "required_extra": "audio",
                "supports_gpu": True,
                "notes": "Torchaudio pretrained backbone with a linear head.",
            },
        ],
    },
    {
        "key": "text_cnn",
        "description": "Text CNN for sequence embeddings (torch).",
        "preferred_backends": ("torch",),
        "backends": [
            {
                "backend": "torch",
                "factory": "modssc.supervised.backends.torch.text_cnn:TorchTextCNNClassifier",
                "required_extra": "supervised-torch",
                "supports_gpu": True,
                "notes": "Torch CNN for text inputs (N, L, D) or (N, D, L).",
            },
        ],
    },
    {
        "key": "linear_svm",
        "description": "Linear SVM classifier (hinge loss).",
        "preferred_backends": ("sklearn",),
        "backends": [
            {
                "backend": "sklearn",
                "factory": "modssc.supervised.backends.sklearn.linear_svm:SklearnLinearSVMClassifier",
                "required_extra": "sklearn",
                "supports_gpu": False,
                "notes": "Uses scikit-learn LinearSVC (no predict_proba).",
            },
        ],
    },
    {
        "key": "ridge",
        "description": "Ridge classifier (linear model).",
        "preferred_backends": ("sklearn",),
        "backends": [
            {
                "backend": "sklearn",
                "factory": "modssc.supervised.backends.sklearn.ridge:SklearnRidgeClassifier",
                "required_extra": "sklearn",
                "supports_gpu": False,
                "notes": "Uses scikit-learn RidgeClassifier (no predict_proba).",
            },
        ],
    },
    {
        "key": "random_forest",
        "description": "Random Forest classifier.",
        "preferred_backends": ("sklearn",),
        "backends": [
            {
                "backend": "sklearn",
                "factory": "modssc.supervised.backends.sklearn.random_forest:SklearnRandomForestClassifier",
                "required_extra": "sklearn",
                "supports_gpu": False,
                "notes": "Uses scikit-learn RandomForestClassifier.",
            },
        ],
    },
    {
        "key": "extra_trees",
        "description": "Extra Trees classifier.",
        "preferred_backends": ("sklearn",),
        "backends": [
            {
                "backend": "sklearn",
                "factory": "modssc.supervised.backends.sklearn.extra_trees:SklearnExtraTreesClassifier",
                "required_extra": "sklearn",
                "supports_gpu": False,
                "notes": "Uses scikit-learn ExtraTreesClassifier.",
            },
        ],
    },
    {
        "key": "gradient_boosting",
        "description": "Gradient Boosting classifier.",
        "preferred_backends": ("sklearn",),
        "backends": [
            {
                "backend": "sklearn",
                "factory": (
                    "modssc.supervised.backends.sklearn.gradient_boosting:"
                    "SklearnGradientBoostingClassifier"
                ),
                "required_extra": "sklearn",
                "supports_gpu": False,
                "notes": "Uses scikit-learn GradientBoostingClassifier.",
            },
        ],
    },
    {
        "key": "gaussian_nb",
        "description": "Gaussian Naive Bayes classifier.",
        "preferred_backends": ("sklearn",),
        "backends": [
            {
                "backend": "sklearn",
                "factory": "modssc.supervised.backends.sklearn.naive_bayes:SklearnGaussianNBClassifier",
                "required_extra": "sklearn",
                "supports_gpu": False,
                "notes": "Uses scikit-learn GaussianNB.",
            },
        ],
    },
    {
        "key": "multinomial_nb",
        "description": "Multinomial Naive Bayes classifier.",
        "preferred_backends": ("sklearn",),
        "backends": [
            {
                "backend": "sklearn",
                "factory": "modssc.supervised.backends.sklearn.naive_bayes:SklearnMultinomialNBClassifier",
                "required_extra": "sklearn",
                "supports_gpu": False,
                "notes": "Uses scikit-learn MultinomialNB.",
            },
        ],
    },
    {
        "key": "bernoulli_nb",
        "description": "Bernoulli Naive Bayes classifier.",
        "preferred_backends": ("sklearn",),
        "backends": [
            {
                "backend": "sklearn",
                "factory": "modssc.supervised.backends.sklearn.naive_bayes:SklearnBernoulliNBClassifier",
                "required_extra": "sklearn",
                "supports_gpu": False,
                "notes": "Uses scikit-learn BernoulliNB.",
            },
        ],
    },
    {
        "key": "lstm_scratch",
        "description": "LSTM from scratch for text sequences (Tabula Rasa).",
        "preferred_backends": ("torch",),
        "backends": [
            {
                "backend": "torch",
                "factory": "modssc.supervised.backends.torch.lstm_scratch:TorchLSTMClassifier",
                "required_extra": "supervised-torch",
                "supports_gpu": True,
                "notes": "Custom LSTM implementation.",
            },
        ],
    },
    {
        "key": "audio_cnn_scratch",
        "description": "2D CNN for Spectrograms from scratch (Tabula Rasa).",
        "preferred_backends": ("torch",),
        "backends": [
            {
                "backend": "torch",
                "factory": "modssc.supervised.backends.torch.audio_cnn_scratch:TorchAudioCNNClassifier",
                "required_extra": "supervised-torch",
                "supports_gpu": True,
                "notes": "Custom 2D CNN implementation.",
            },
        ],
    },
    {
        "key": "graphsage_inductive",
        "description": "GraphSAGE Inductive (Tabula Rasa).",
        "preferred_backends": ("torch",),
        "backends": [
            {
                "backend": "torch",
                "factory": (
                    "modssc.supervised.backends.torch.graphsage_inductive:TorchGraphSAGEClassifier"
                ),
                "required_extra": "supervised-torch-geometric",
                "supports_gpu": True,
                "notes": "Custom GraphSAGE implementation.",
            },
        ],
    },
]
