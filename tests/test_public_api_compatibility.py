from __future__ import annotations

from dataclasses import fields
from inspect import Parameter, signature

import pytest

from modssc.inductive.methods.co_training import CoTrainingSpec
from modssc.inductive.methods.democratic_co_learning import DemocraticCoLearningSpec
from modssc.inductive.methods.fixmatch import FixMatchSpec
from modssc.inductive.methods.flexmatch import FlexMatchSpec
from modssc.inductive.methods.free_match import FreeMatchSpec
from modssc.inductive.methods.pseudo_label import PseudoLabelSpec
from modssc.inductive.methods.self_training import SelfTrainingSpec
from modssc.inductive.methods.softmatch import SoftMatchSpec
from modssc.inductive.methods.tri_training import TriTrainingSpec
from modssc.preprocess.steps.core.vae import VaeStep
from modssc.preprocess.steps.vision.aet import AetStep
from modssc.transductive.methods.classic.laplace_learning import LaplaceLearningSpec
from modssc.transductive.methods.pde.poisson_learning import PoissonLearningSpec

_COMPATIBILITY_CASES = {
    PseudoLabelSpec: (
        (
            "classifier_id",
            "classifier_backend",
            "classifier_params",
            "max_iter",
            "confidence_threshold",
            "max_new_labels",
            "min_new_labels",
        ),
        (
            "paper_input_dim",
            "paper_hidden_units",
            "paper_num_classes",
            "paper_epochs",
            "paper_labeled_batch_size",
            "paper_unlabeled_batch_size",
            "paper_hidden_dropout",
            "paper_input_dropout",
            "paper_initial_learning_rate",
            "paper_learning_rate_decay",
            "paper_momentum_initial",
            "paper_momentum_final",
            "paper_momentum_ramp_epochs",
            "paper_alpha_final",
            "paper_alpha_start_epoch",
            "paper_alpha_end_epoch",
            "training_mode",
        ),
    ),
    FixMatchSpec: (
        (
            "model_bundle",
            "lambda_u",
            "p_cutoff",
            "temperature",
            "mu",
            "hard_label",
            "use_cat",
            "batch_size",
            "max_epochs",
            "detach_target",
        ),
        (
            "max_steps",
            "training_mode",
            "reference_implementation",
            "sampler_mode",
            "sampler_shuffle_buffer",
            "augmentation_profile",
            "interleave_bn",
            "evaluation_interval_steps",
            "evaluation_tail_interval_steps",
            "evaluation_tail_start_fraction",
            "checkpoint_interval_steps",
            "reporting_policy",
            "reporting_window_checkpoints",
            "allow_short_run",
        ),
    ),
    FlexMatchSpec: (
        (
            "model_bundle",
            "lambda_u",
            "p_cutoff",
            "temperature",
            "mu",
            "hard_label",
            "thresh_warmup",
            "use_cat",
            "batch_size",
            "max_epochs",
            "detach_target",
        ),
        (
            "max_steps",
            "training_mode",
            "reference_implementation",
            "sampler_mode",
            "sampler_shuffle_buffer",
            "augmentation_profile",
            "interleave_bn",
            "evaluation_interval_steps",
            "evaluation_tail_interval_steps",
            "evaluation_tail_start_fraction",
            "checkpoint_interval_steps",
            "reporting_policy",
            "reporting_window_checkpoints",
            "allow_short_run",
        ),
    ),
    FreeMatchSpec: (
        (
            "model_bundle",
            "lambda_u",
            "lambda_e",
            "temperature",
            "ema_p",
            "use_quantile",
            "clip_thresh",
            "hard_label",
            "use_cat",
            "mu",
            "batch_size",
            "max_epochs",
            "detach_target",
        ),
        (
            "max_steps",
            "training_mode",
            "reference_implementation",
            "sampler_mode",
            "sampler_shuffle_buffer",
            "augmentation_profile",
            "interleave_bn",
            "evaluation_interval_steps",
            "evaluation_tail_interval_steps",
            "evaluation_tail_start_fraction",
            "checkpoint_interval_steps",
            "reporting_policy",
            "reporting_window_checkpoints",
            "allow_short_run",
        ),
    ),
    SoftMatchSpec: (
        (
            "model_bundle",
            "lambda_u",
            "temperature",
            "ema_p",
            "n_sigma",
            "per_class",
            "dist_align",
            "dist_uniform",
            "hard_label",
            "use_cat",
            "mu",
            "batch_size",
            "max_epochs",
            "detach_target",
        ),
        (
            "max_steps",
            "training_mode",
            "reference_implementation",
            "sampler_mode",
            "sampler_shuffle_buffer",
            "augmentation_profile",
            "interleave_bn",
            "evaluation_interval_steps",
            "evaluation_tail_interval_steps",
            "evaluation_tail_start_fraction",
            "checkpoint_interval_steps",
            "reporting_policy",
            "reporting_window_checkpoints",
            "allow_short_run",
        ),
    ),
    LaplaceLearningSpec: (
        ("cg_tol", "cg_max_iter"),
        ("backend", "solver", "require_convergence"),
    ),
    PoissonLearningSpec: (
        ("backend", "laplacian_kind", "eps", "center_sources", "tol", "max_iter"),
        ("solver", "balance_scores", "class_priors", "min_iter", "require_convergence"),
    ),
    VaeStep: (
        (
            "latent_dim",
            "hidden_dims",
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "beta",
            "dropout",
            "input_scaling",
            "reconstruction_loss",
            "decoder_output",
            "standardize",
            "device",
            "max_fit_samples",
            "preset",
            "expected_input_dim",
            "model_cache",
            "model_cache_dir",
            "cache_key",
            "model_seed",
            "fit_scope",
            "mean_",
            "scale_",
            "impute_",
        ),
        ("require_cache_hit", "expected_model_fingerprint"),
    ),
    AetStep: (
        (
            "source",
            "preset",
            "checkpoint_path",
            "checkpoint_name",
            "model_cache_dir",
            "features_path",
            "labels_path",
            "extracted_npy_path",
            "train_offset",
            "test_offset",
            "expected_rows",
            "feature_layer",
            "input_scaling",
            "unit_normalize",
            "batch_size",
            "device",
        ),
        ("expected_features_sha256", "expected_labels_sha256"),
    ),
    CoTrainingSpec: (
        (
            "classifier_id",
            "classifier_backend",
            "classifier_params",
            "view_keys",
            "max_iter",
            "k_per_class",
            "confidence_threshold",
        ),
        (
            "protocol",
            "p",
            "n",
            "u",
            "k",
            "positive_label",
            "negative_label",
            "dynamic_feature_selection",
            "feature_selection_max_features",
            "selection_score",
        ),
    ),
    DemocraticCoLearningSpec: (
        (
            "classifier_id",
            "classifier_backend",
            "classifier_params",
            "max_iter",
            "confidence_level",
            "min_confidence",
            "n_learners",
            "classifier_specs",
        ),
        (
            "confidence_estimator",
            "confidence_interval",
            "confidence_folds",
            "confidence_seed",
            "diagnostic_trace",
            "control_mode",
            "training_mode",
            "require_convergence",
            "min_pseudo_labels_added",
        ),
    ),
    SelfTrainingSpec: (
        (
            "classifier_id",
            "classifier_backend",
            "classifier_params",
            "max_iter",
            "confidence_threshold",
            "max_new_labels",
            "min_new_labels",
            "use_group_propagation",
            "group_key",
            "group_min_count",
            "group_min_fraction",
            "group_confidence_threshold",
        ),
        (
            "selection_strategy",
            "paper_pool_size_unspecified",
            "paper_candidates_per_class_unspecified",
            "paper_distance_confidence_unspecified",
            "paper_feature_scaling_unspecified",
        ),
    ),
    TriTrainingSpec: (
        (
            "classifier_id",
            "classifier_backend",
            "classifier_params",
            "max_iter",
            "confidence_threshold",
            "max_new_labels",
            "bootstrap_ratio",
        ),
        (
            "retain_initial_ensemble",
            "prediction_rule",
            "training_mode",
            "require_convergence",
            "min_pseudo_labels_added",
        ),
    ),
}


@pytest.mark.parametrize("spec_class", tuple(_COMPATIBILITY_CASES))
def test_benchmark_profile_identity_stays_outside_modssc_specs(
    spec_class: type[object],
) -> None:
    assert not hasattr(spec_class(), "profile")


@pytest.mark.parametrize(
    ("spec_class", "historical_names", "extension_names"),
    [(spec_class, *names) for spec_class, names in _COMPATIBILITY_CASES.items()],
    ids=[spec_class.__name__ for spec_class in _COMPATIBILITY_CASES],
)
def test_f69_positional_constructor_order_and_keyword_only_extensions(
    spec_class: type[object],
    historical_names: tuple[str, ...],
    extension_names: tuple[str, ...],
) -> None:
    parameters = signature(spec_class).parameters
    positional_names = tuple(
        name
        for name, parameter in parameters.items()
        if parameter.kind is Parameter.POSITIONAL_OR_KEYWORD
    )
    keyword_only_names = tuple(
        name for name, parameter in parameters.items() if parameter.kind is Parameter.KEYWORD_ONLY
    )

    assert positional_names == historical_names
    assert keyword_only_names == extension_names

    positional_values = tuple(object() for _ in historical_names)
    instance = spec_class(*positional_values)
    assert all(
        getattr(instance, name) is value
        for name, value in zip(historical_names, positional_values, strict=True)
    )

    dataclass_fields = {item.name: item for item in fields(spec_class)}
    assert all(dataclass_fields[name].kw_only for name in extension_names)
