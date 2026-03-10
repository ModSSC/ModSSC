# Multimodal Representation Standardization Policy

This note extends the benchmark fairness pass beyond vision to `text`, `tabular`, `audio`, and `graph`.

The benchmark rule remains the same:

- compare methods, not arbitrary legacy backbones
- keep transductive feature extractors uniform within a modality family
- keep resource-heavy extractors explicit and standardized when they are unavoidable
- keep augmentations modality-native whenever the training pipeline allows it
- document every exception where the method contract requires a different representation

## Text

### Inductive backbone policy

- Default trainable backbone: `lstm_scratch`
- Tokenization pipeline: `text.ensure_strings` -> `text.vocab_tokenizer`
- Shared tokenizer defaults:
  - `max_length: 256`
  - `vocab_size: 20000`

### Inductive augmentation policy

The text inductive benchmark tokenizes before online augmentation, so string-level augmenters are not compatible with the training path. The standardized policy therefore uses token-sequence perturbations:

- weak: `text.token_mask(p=0.05, mask_token_id=1, pad_token_id=0)`
- strong:
  - `text.token_mask(p=0.15, mask_token_id=1, pad_token_id=0)`
  - `text.token_swap(n_swaps=1, pad_token_id=0)`

### Transductive feature pipeline policy

- `text.ensure_strings`
- `text.sentence_transformer(model_id=st:all-MiniLM-L6-v2, batch_size=32)`
- `core.pca(n_components=128)`
- `core.to_numpy`

### Exceptions

- `co_training`: switched to frozen sentence embeddings + `logreg`
  - reason: view generation splits feature columns; doing this on token-position matrices is not a fair semantic multi-view benchmark
- `setred`: switched to frozen sentence embeddings + `logreg`
  - reason: Setred builds kNN graphs over 2D feature matrices; token-id matrices are a poor and unstable proxy for semantic text distance
- `tsvm`: switched to the same frozen sentence-embedding pipeline
  - reason: TSVM is a feature-space method; token-id matrices make comparisons depend on tokenizer indexing rather than text representation quality

## Tabular

### Inductive backbone policy

- Default trainable backbone: `mlp`
- Standard preprocessing:
  - numeric datasets: `core.ensure_2d` -> `tabular.impute` -> `tabular.standard_scaler`
  - categorical/mixed datasets: add `tabular.one_hot` first

### Inductive augmentation policy

- weak: `tabular.gaussian_noise`
- strong: `tabular.swap_noise`

These were retained because they already act in the correct feature space for tabular MLP training.

### Transductive feature pipeline policy

- numeric datasets:
  - `labels.encode`
  - `core.ensure_2d`
  - `tabular.impute`
  - `tabular.standard_scaler`
  - `core.pca(n_components=30)`
  - `core.to_numpy`
- mixed/categorical datasets:
  - add `tabular.one_hot` before the same pipeline

`PCA(30)` is retained as a shared resource-control step for tabular graph construction and transductive sweeps.

### Exceptions

- `tsvm` (inductive): switched to the same standardized tabular preprocessing but ending in `core.to_numpy`
  - reason: TSVM is a feature-space method and should not bypass imputation / scaling when other tabular methods do not

## Audio

### Inductive backbone policy

- Default trainable backbone: `audio_pretrained`
- Shared backbone bundle: `WAV2VEC2_BASE`
- Trainability: `freeze_backbone: false`

### Inductive augmentation policy

- weak: `audio.add_noise(std=0.002)`
- strong:
  - `audio.add_noise(std=0.005)`
  - `audio.time_shift(max_frac=0.1)`

This replaces generic tabular perturbations that previously operated directly on waveform tensors.

### Transductive feature pipeline policy

- `audio.wav2vec2(model_id=wav2vec2:base, batch_size=8)`
- `core.pca(n_components=128)`
- `core.to_numpy`

### Exceptions

- `co_training`: switched to frozen wav2vec2 features + `logreg`
  - reason: column splits over raw waveforms are not meaningful feature views
- `setred`: switched to frozen wav2vec2 features + `logreg`
  - reason: Setred kNN filtering over raw waveforms is not a fair audio representation benchmark
- `tsvm`: switched to the same frozen wav2vec2 feature pipeline
  - reason: TSVM should operate on standardized 2D audio embeddings, not raw waveforms

## Graph

### Inductive backbone policy

- Default trainable backbone: `graphsage_inductive`
- Standard graph preprocess: `graph.sparse_adjacency` -> `core.to_torch`

### Inductive augmentation policy

- weak: `graph.feature_mask(p=0.05)`
- strong:
  - `graph.feature_mask(p=0.1)`
  - `graph.edge_dropout(p=0.2)`

This replaces generic tabular perturbations and keeps augmentation aligned with graph SSL practice.

### Transductive feature pipeline policy

- `labels.encode`
- `core.ensure_2d`
- `core.to_numpy`

The native graph topology remains part of the dataset / method contract and is not replaced by an external encoder.

### Exceptions

- no additional graph-specific representation exception was applied in this pass
- `co_training` and `setred` were left on graph-native inputs because the feature views operate on node-feature columns while preserving the graph structure, which is materially different from splitting image pixels, token positions, or waveform samples
