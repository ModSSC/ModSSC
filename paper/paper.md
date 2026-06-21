---
title: 'ModSSC: A Modular Scientific Python Framework for Reproducible Semi-Supervised Classification'
tags:
  - Python
  - semi-supervised learning
  - classification
  - reproducible research
  - transductive learning
  - inductive learning
  - graph-based learning
authors:
  - name: Melvin Barbaux
    orcid: 0009-0009-2269-967X
    affiliation: 1
  - name: Samia Boukir
    orcid: 0000-0002-0907-081X
    affiliation: 1
affiliations:
  - name: Univ. Bordeaux, CNRS, Bordeaux INP, IMB, UMR 5251, F-33400 Talence, France
    index: 1
date: 8 June 2026
bibliography: paper.bib
---

# Summary

ModSSC is an extensible and reliable open-source Python package for semi-supervised classification and reproducible experiments. It targets problems where only a part of a dataset is labelled and additional unlabelled samples are available. The package supports both inductive settings, where models are applied later to new samples, and transductive settings, where predictions are made on a fixed unlabelled set at training time.

The main goal of ModSSC is practical reuse: users can load data via Python objects, select a semi-supervised method, define a label budget, optionally preprocess features or build a graph, and evaluate predictions without reimplementing the full pipeline. The same components can be composed via YAML configuration files to repeat experiments across datasets, methods, random seeds, graph choices, preprocessing steps, and metrics.

ModSSC currently provides stable workflows for tabular, graph, image, and text classification, with optional provider support for audio datasets. The package features over 50 implemented semi-supervised methods, spanning inductive, transductive, graph-based, and neural algorithms, alongside supervised baselines. Its purpose is not to introduce a new algorithm, but to make established semi-supervised classification methods easier to apply, adapt, extend, and compare within a single Python framework.

# Statement of need

Semi-supervised classification (SSC) is used when labels are scarce but unlabelled data are readily available. This situation occurs in scientific and applied domains such as image analysis, text classification, tabular prediction, node classification, biomedical data analysis, and remote sensing. Although the methodological literature is broad [@vanEngelen2020survey], practical use often remains difficult because implementations are scattered across method-specific repositories, benchmark-specific codebases, and modality-specific pipelines.

A recurring difficulty in semi-supervised learning research is that inductive neural methods and transductive graph-based methods are often evaluated in separate software ecosystems. They frequently use different data splits, preprocessing steps, graph construction rules, training interfaces, and evaluation metrics. As a result, methodological comparisons can be affected by pipeline differences rather than by the semi-supervised method itself. This issue is especially important because evaluation protocols have a strong influence on conclusions in semi-supervised learning [@Oliver2018RealisticSSL].

ModSSC addresses this need at two levels, ensuring both flexibility and stability of comparisons. First, it is a Python package for applying SSC methods to custom data. Users can work directly with matrices and Python objects, for example `InductiveDataset`, `LoadedDataset`, or `NodeDataset`, and then call the relevant dataset, sampling, preprocessing, graph structure, SSC method, and evaluation components. Second, it is a configuration-driven experimental framework. YAML files can specify the dataset, label budget, preprocessing plan, graph construction, SSC method, evaluation metrics, seed, and output paths. This makes it possible to reuse the same experimental substrate while changing only the component under study.

This combination is important for both applied users and researchers. Applied researchers need a unified package to avoid the burden of reimplementing SSC algorithms directly from papers or integrating them from fragmented online sources spanning multiple programming languages, including Java, Python, and C++. Researchers need reproducible workflows for comparing inductive and transductive methods under shared protocols. ModSSC is designed to serve both use cases without forcing users into a single fixed benchmark.

# State of the field

Several software projects provide partial support for semi-supervised classification. Scikit-learn includes classical label propagation and label spreading estimators [@Pedregosa2011sklearn]. LAMDA-SSL provides a broad semi-supervised learning toolkit with many statistical and deep methods [@Jia2023LAMDA]. USB and TorchSSL focus on unified benchmarking of modern deep semi-supervised algorithms [@Wang2022USB; @TorchSSL]. GraphLearning provides graph-based learning tools, including semi-supervised learning methods [@graphlearning].

ModSSC complements and extends these projects in a unified way. Its distinctive focus is the integration of practical Python use and reproducible experiment composition across both inductive and transductive SSC settings. In ModSSC, sampling, preprocessing, graph construction, SSC method execution, and evaluation are separate components. This allows users to reuse, for example, the same data split and preprocessing while changing only the method, or to reuse a SSC method while changing the feature representation or graph. These gaps in tooling motivate ModSSC's integrated approach.

# Functionality

ModSSC provides the following functionalities:

- dataset loading and dataset catalogues for multiple data modalities;
- Python objects for user-provided data, including inductive and graph/node-classification settings;
- deterministic semi-supervised sampling plans, including holdout and k-fold splits;
- labelled-subset selection by fraction, total count, per-class count, or fixed indices;
- optional class-imbalance scenarios for labelled, training, or test subsets;
- preprocessing pipelines with transform, fittable, and featurizer steps;
- graph construction from feature matrices with k-nearest-neighbour, epsilon, and anchor schemes;
- graph backends based on NumPy, scikit-learn, PyTorch, and optional Faiss;
- graph post-processing through edge weighting, self-loops, symmetrization, and normalization;
- graph-derived views, including attribute, diffusion, and structural views;
- registries for supervised, semi-supervised inductive, and transductive classification methods;
- evaluation utilities and benchmark summaries;
- command-line tools and YAML-based benchmark execution.

The stable catalogue of 50 SSC methods includes representative families including both mainstream algorithms such as pseudo-labeling, self-training, co-training, tri-training, label propagation, label spreading, Laplace learning, Poisson learning, Poisson MBO, p-Laplace learning, and more advanced algorithms such as FixMatch/MixMatch-style workflows, Mean Teacher, and graph neural network baselines. Some deep-learning, graph, audio, and FAISS functionalities rely on optional dependencies. These optional extras allow the core package to remain lightweight while enabling richer workflows when the corresponding backend is installed.

# Software design and usage

ModSSC separates practical use from experiment orchestration. The Python API supports flexible workflows: load or construct datasets, sample labelled and unlabelled subsets, preprocess features, optionally build a graph, retrieve a SSC method from the registry, fit or predict, and evaluate. Common functions include `load_dataset`, `sample`, `preprocess`, `build_graph`, `get_method_class`, `fit`, `predict`, and `evaluate`.

For reproducible experiments, the YAML interface is recommended. A complete experiment can be launched from a configuration file specifying the dataset, label budget, preprocessing, graph construction, SSC method, evaluation metrics, and seed. For example, benchmark configurations are executed with:

```bash
python -m bench.main --config "$CONFIG"
```

The following structure illustrates an example with a minimal configuration:

```yaml
dataset:
  key: toy_blobs
sampling:
  split:
    kind: holdout
    test_fraction: 0.2
    val_fraction: 0.1
  labeling:
    mode: per_class
    value: 5
preprocess:
  steps:
    - id: core.ensure_2d
    - id: tabular.standard_scaler
graph:
  scheme: knn
  metric: cosine
  k: 10
method:
  name: label_propagation
evaluation:
  metrics:
    - accuracy
    - balanced_accuracy
seed: 0
```

Changing the SSC method block while keeping the same dataset, split, preprocessing plan, graph settings, and evaluation metrics reuses the same experimental protocol. This is the central software abstraction in ModSSC: users can straightforwardly adapt the code to various classification settings without reimplementing data handling, sampling, preprocessing, graph construction, and evaluation.

Reproducibility is supported through deterministic sampling plans, seed derivation for split generation and label selection, dataset and split fingerprints, cached split artefacts, preprocessing fingerprints, graph fingerprints, explicit cache directories, and structured output locations. Graph construction also supports caching and resumable computation for chunked NumPy backends. The repository includes configuration examples such as `bench/configs/experiments/toy_inductive.yaml`, `bench/configs/experiments/toy_transductive.yaml`, and benchmark configurations under `bench/configs/best/`. ModSSC is extensible and open for contributed development.

# Research impact

ModSSC is a research software tool as it supports the execution, adaptation, and comparison of SSC workflows, not only a single specific analysis. The associated preprint describes the scientific motivation and benchmark experiments based on the proposed framework [@Barbaux2025ModSSC]. This paper focuses on the reusable software infrastructure, while the preprint provides the broader experimental context.

The project provides several verifiable software-maturity signals. ModSSC is distributed on PyPI as `modssc` and the version targeted for this submission is `1.2.1`. The repository includes source code, documentation, tests, examples, benchmark configurations, a changelog, citation metadata, issue tracking, pull requests, and tagged releases. The continuous integration workflow runs linting and formatting checks, executes the test suite on Python 3.11 and 3.12, uploads coverage reports, and builds the source and wheel distributions.

The repository also includes executable examples for inductive pseudo-labeling, transductive label propagation, sampling, evaluation, hyperparameter-search primitives, and command-line smoke testing. These examples allow users to test the package locally before running larger or more complex benchmark configurations. The benchmark configurations associated with the current research workflows are stored under `bench/configs/best/`, with configurations grouped by experimental rounds.

ModSSC has been used by the authors to run SSC studies across tabular benchmarks, graph/node-classification workflows, image workflows, and text workflows based on embeddings. Its main contribution to research practice is to lower the engineering cost of using SSC methods and to make comparisons between inductive and transductive approaches easier to inspect and reproduce.

# Availability

The source code is available at <https://github.com/ModSSC/ModSSC> under the MIT License. The documentation is available at <https://modssc.github.io/ModSSC/>. The package can be installed from PyPI with:

```bash
pip install modssc
```

The package requires Python 3.11 or newer. Optional extras are provided for modality-specific and backend-specific functionality, including text, image, audio, graph, Torch-based methods, PyG-based methods, preprocessing backends, and full-feature installations.

# Conclusion and future work

ModSSC offers a unified Python framework for applying and benchmarking semi-supervised classification methods across inductive and transductive settings. Its modular design makes it possible to reuse the same datasets, splits, preprocessing steps, graphs, methods, and metrics in reproducible workflows. Future work will focus on expanding the method and dataset catalogues, improving documentation and examples, strengthening automated testing, and supporting additional modalities and backends while preserving a stable and extensible core.

# AI usage disclosure

No generative AI was used for architectural design or core algorithms. ChatGPT (OpenAI) was used for bug fixing, generating YAML configuration files, and making minor inline code changes. The authors reviewed, edited, tested, and validated all AI-assisted outputs.

# Acknowledgements

Computer time for this study was provided by the computing facilities of the MCIA (Mesocentre de Calcul Intensif Aquitain) on the Cali V3 cluster. This project was also provided with HPC/AI computing and storage resources by GENCI at IDRIS and TGCC through grant 2026-AD011017490 on the Jean Zay V100, A100 and H100 partitions, and on the Joliot Curie ROME partition.

The computing resource providers had no role in the design of the software, the interpretation of results, and the preparation of this manuscript.

# References
