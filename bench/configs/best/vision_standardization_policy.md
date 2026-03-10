# Vision Benchmark Standardization Policy

This benchmark family is standardized so that the main variable is the SSL method, not a legacy backbone or feature extractor choice.

## Inductive backbone policy

- Default inductive backbone: `image_pretrained` with `resnet18`.
- Default inductive backbone settings: `weights: null`, `freeze_backbone: false`, `input_layout: channels_first`.
- Default image preprocess for trainable inductive methods:
  - `vision.ensure_num_channels(num_channels=3)`
  - `vision.channels_order(order=NCHW)`
  - native family resize
  - `vision.normalize` with the shared ResNet/ImageNet input statistics
  - `core.copy_raw`

## Transductive feature policy

- Default transductive image representation: frozen `torchvision:resnet18` penultimate-layer embeddings.
- OpenCLIP is not the benchmark default for vision anymore.
- Default transductive image preprocess:
  - `vision.ensure_num_channels(num_channels=3)`
  - `vision.channels_order(order=NCHW)`
  - native family resize
  - `vision.normalize` with the shared ResNet/ImageNet input statistics
  - `embeddings.auto(model_id_vision=torchvision:resnet18)`
  - `core.to_numpy`

## Resolution policy

- `mnist`: `28x28`
- `cifar10`, `cifar100`, `svhn`: `32x32`
- `stl10`: `96x96`

## Augmentation policy

- `mnist`: crop-pad(2); strong adds cutout(`frac=0.25`)
- `cifar10`, `cifar100`: crop-pad(4) + horizontal flip; strong adds cutout(`frac=0.25`)
- `svhn`: crop-pad(4); strong adds cutout(`frac=0.25`)
- `stl10`: crop-pad(12) + horizontal flip; strong adds cutout(`frac=0.25`)

## Exceptions

- `mnist` and `svhn` do not use horizontal flips because mirroring changes digit semantics.
- Single-channel datasets are repeated to 3 channels because the standardized ResNet-18 backbone expects RGB-shaped inputs.
- `co_training` uses frozen `resnet18` embeddings with a linear classifier because the current view pipeline splits feature columns, not spatial image views; applying it directly to raw 4D tensors is not method-fair.
- `setred` uses frozen `resnet18` embeddings with a linear classifier because SetRED internally constructs a kNN graph over 2D feature matrices; raw 4D image tensors are invalid for that path.
- Inductive and transductive regimes intentionally differ in trainability:
  - inductive methods train the standardized `resnet18` backbone
  - transductive methods consume a standardized frozen `resnet18` feature pipeline
