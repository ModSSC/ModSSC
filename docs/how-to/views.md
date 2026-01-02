# How to generate multi-view features

Multi-view methods expect multiple feature sets for the same samples. This recipe shows how to define and generate those views, and it includes a bench config excerpt for reference. For an end-to-end example, see the [inductive tutorial](../tutorials/inductive-toy.md).


## Problem statement
You need multiple feature views of the same dataset for classic multi-view SSL methods like co-training. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a><a href="#source-3">[3]</a></sup> If your method consumes a single feature matrix, you can skip views and rely on preprocessing alone.


## When to use
Use this when a method expects `data.views` instead of a single feature matrix (for example, co-training). <sup class="cite"><a href="#source-3">[3]</a><a href="#source-4">[4]</a></sup>


## Steps
1) Define a `ViewsPlan` (two or more views). <sup class="cite"><a href="#source-1">[1]</a></sup>

2) Optionally attach preprocessing to each view. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-5">[5]</a></sup>

3) Generate views and pass them to inductive methods. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-4">[4]</a></sup>


## Copy-paste example
Use the Python helper when you want to generate views directly in code. The YAML excerpt shows the equivalent `views` block inside a bench config. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-6">[6]</a><a href="#source-7">[7]</a></sup>

Python:

```python
from modssc.data_loader import load_dataset
from modssc.views import generate_views, two_view_random_feature_split

plan = two_view_random_feature_split(fraction=0.5)
ds = load_dataset("toy", download=True)
views = generate_views(ds, plan=plan, seed=0)
print(list(views.views.keys()))
```

Bench config excerpt (co-training views): <sup class="cite"><a href="#source-6">[6]</a></sup>


```yaml
views:
  seed: 2
  plan:
    views:
    - name: view_a
      columns:
        mode: random
        fraction: 0.5
    - name: view_b
      columns:
        mode: complement
        complement_of: view_a
```

The view plan schema is defined in [`src/modssc/views/plan.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/views/plan.py), and the bench schema accepts `views.plan` when present. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-7">[7]</a></sup>


## Pitfalls
!!! warning
    `ViewsPlan` must define at least two views, and complement views must refer to an earlier view in the plan. <sup class="cite"><a href="#source-1">[1]</a></sup>


!!! tip
    If you need preprocessing per view, attach a `PreprocessPlan` to each `ViewSpec`. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-5">[5]</a></sup>


## Related links
- [Inductive tutorial](../tutorials/inductive-toy.md)
- [Preprocess how-to](preprocess.md)
- [Configuration reference](../reference/configuration.md)

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/views/plan.py"><code>src/modssc/views/plan.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/views/api.py"><code>src/modssc/views/api.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/methods/co_training.py"><code>src/modssc/inductive/methods/co_training.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/types.py"><code>src/modssc/inductive/types.py</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/plan.py"><code>src/modssc/preprocess/plan.py</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/best/inductive/co_training/text/imdb.yaml"><code>bench/configs/experiments/best/inductive/co_training/text/imdb.yaml</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py"><code>bench/schema.py</code></a></li>
</ol>
</details>
