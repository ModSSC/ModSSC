# Data augmentation API

This page documents the data augmentation API. For usage patterns, see [Data augmentation how-to](../../how-to/augmentation.md).


## What it is for
The data augmentation brick defines training-time, stochastic transforms for multiple modalities with deterministic contexts. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## Examples
List available ops:

```python
from modssc.data_augmentation import available_ops

print(available_ops(modality="vision"))
```

Build and apply a pipeline:

```python
import numpy as np
from modssc.data_augmentation import AugmentationContext, AugmentationPlan, StepConfig, build_pipeline

plan = AugmentationPlan(steps=(StepConfig(op_id="tabular.gaussian_noise", params={"std": 0.1}),))
pipeline = build_pipeline(plan)

x = np.zeros((4,), dtype=np.float32)
ctx = AugmentationContext(seed=0, sample_id=0, epoch=0)
print(pipeline(x, ctx=ctx))
```

The registry and plan schema are defined in [`src/modssc/data_augmentation/registry.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/registry.py) and [`src/modssc/data_augmentation/plan.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/plan.py). <sup class="cite"><a href="#source-3">[3]</a><a href="#source-4">[4]</a></sup>


## API reference

::: modssc.data_augmentation

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/api.py"><code>src/modssc/data_augmentation/api.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/types.py"><code>src/modssc/data_augmentation/types.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/registry.py"><code>src/modssc/data_augmentation/registry.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/plan.py"><code>src/modssc/data_augmentation/plan.py</code></a></li>
</ol>
</details>
