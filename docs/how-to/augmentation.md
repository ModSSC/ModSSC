# How to use data augmentation

This page focuses on training-time augmentation plans and how to inspect available ops. If you are looking for feature engineering and caching that happens before training, see [Preprocess how-to](preprocess.md).


## Problem statement
You want to apply deterministic, training-time augmentations to inputs for SSL methods (for example, weak/strong views). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup> Keeping augmentations separate makes it easier to swap them without touching cached preprocessing. <sup class="cite"><a href="#source-7">[7]</a><a href="#source-8">[8]</a></sup>


## When to use
Use this when your method expects stochastic augmentations (FixMatch-style, strong/weak pipelines). <sup class="cite"><a href="#source-1">[1]</a></sup>


## Steps
1) Inspect available ops for your modality. <sup class="cite"><a href="#source-3">[3]</a><a href="#source-4">[4]</a></sup>

2) Define an `AugmentationPlan` (list of ops + params). <sup class="cite"><a href="#source-2">[2]</a></sup>

3) Build and apply a pipeline with a deterministic context. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-5">[5]</a></sup>


## Copy-paste example
Use the CLI when you want to inspect ops quickly (`modssc augmentation` in [`src/modssc/cli/augmentation.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/augmentation.py)), and use Python when you want to build pipelines in code (API in [`src/modssc/data_augmentation/api.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/api.py)). <sup class="cite"><a href="#source-4">[4]</a><a href="#source-1">[1]</a></sup>

CLI:

```bash
modssc augmentation list --modality vision
modssc augmentation info vision.random_horizontal_flip --as-json
```

Python:

```python
import numpy as np
from modssc.data_augmentation import AugmentationContext, AugmentationPlan, StepConfig, build_pipeline

plan = AugmentationPlan(
    steps=(
        StepConfig(op_id="vision.random_horizontal_flip", params={"p": 0.5}),
        StepConfig(op_id="vision.cutout", params={"frac": 0.25, "fill": 0.0}),
    ),
    modality="vision",
)
pipeline = build_pipeline(plan)

x = np.zeros((32, 32, 3), dtype=np.float32)
ctx = AugmentationContext(seed=0, sample_id=0, epoch=0)
aug_x = pipeline(x, ctx=ctx)
print(aug_x.shape)
```

Ops and plan mechanics are defined in [`src/modssc/data_augmentation/ops/`](https://github.com/ModSSC/ModSSC/tree/main/src/modssc/data_augmentation/ops) and [`src/modssc/data_augmentation/api.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/api.py). <sup class="cite"><a href="#source-6">[6]</a><a href="#source-1">[1]</a></sup>


## Pitfalls
!!! warning
    Augmentations are applied at training time; [preprocessing](preprocess.md) is a separate, cacheable step. Do not mix the two in the same pipeline. <sup class="cite"><a href="#source-7">[7]</a><a href="#source-8">[8]</a></sup>


!!! tip
    Use `AugmentationContext` to make randomness deterministic across epochs and samples. <sup class="cite"><a href="#source-5">[5]</a><a href="#source-9">[9]</a></sup>


## Related links
- [Configuration reference](../reference/configuration.md)
- [Inductive tutorial](../tutorials/inductive-toy.md)
- [Catalogs and registries](../reference/catalogs.md)

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/api.py"><code>src/modssc/data_augmentation/api.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/plan.py"><code>src/modssc/data_augmentation/plan.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/registry.py"><code>src/modssc/data_augmentation/registry.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/augmentation.py"><code>src/modssc/cli/augmentation.py</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/types.py"><code>src/modssc/data_augmentation/types.py</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/ops/vision.py"><code>src/modssc/data_augmentation/ops/vision.py</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/__init__.py"><code>src/modssc/data_augmentation/__init__.py</code></a></li>
  <li id="source-8"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/__init__.py"><code>src/modssc/preprocess/__init__.py</code></a></li>
  <li id="source-9"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/utils.py"><code>src/modssc/data_augmentation/utils.py</code></a></li>
</ol>
</details>
