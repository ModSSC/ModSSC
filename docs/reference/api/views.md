# Views API

This page documents the views API. For workflows, see [Views how-to](../../how-to/views.md).


## What it is for
The views brick creates multiple feature views for multi-view SSL methods like co-training. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## Examples
Generate two random feature views:

```python
from modssc.data_loader import load_dataset
from modssc.views import generate_views, two_view_random_feature_split

ds = load_dataset("toy", download=True)
plan = two_view_random_feature_split(fraction=0.5)
views = generate_views(ds, plan=plan, seed=0)
print(list(views.views.keys()))
```

Create a custom plan with explicit indices:

```python
from modssc.views import ColumnSelectSpec, ViewSpec, ViewsPlan

plan = ViewsPlan(
    views=(
        ViewSpec(name="view_a", columns=ColumnSelectSpec(mode="indices", indices=(0, 1, 2))),
        ViewSpec(name="view_b", columns=ColumnSelectSpec(mode="complement", complement_of="view_a")),
    )
)
```

The view plan schema is defined in [`src/modssc/views/plan.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/views/plan.py). <sup class="cite"><a href="#source-2">[2]</a></sup>


## API reference

::: modssc.views

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/views/api.py"><code>src/modssc/views/api.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/views/plan.py"><code>src/modssc/views/plan.py</code></a></li>
</ol>
</details>
