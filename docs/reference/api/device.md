# Runtime device API

This page documents device utilities exposed from the `modssc.runtime` package. For configuration context, see the [Configuration reference](../configuration.md).


## What it is for
Device utilities resolve `auto` device selection based on available torch backends. <sup class="cite"><a href="#source-1">[1]</a></sup>


## Examples
Resolve a device name without importing torch explicitly:

```python
from modssc.runtime.device import resolve_device_name

print(resolve_device_name("auto"))
```

Resolve with an existing torch module:

```python
import torch
from modssc.runtime.device import resolve_device_name

print(resolve_device_name("auto", torch=torch))
```

Device resolution logic is implemented in [`src/modssc/runtime/device.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/runtime/device.py) and re-exported from [`src/modssc/runtime/__init__.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/runtime/__init__.py). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## API reference

::: modssc.runtime.device

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/runtime/device.py"><code>src/modssc/runtime/device.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/runtime/__init__.py"><code>src/modssc/runtime/__init__.py</code></a></li>
</ol>
</details>
