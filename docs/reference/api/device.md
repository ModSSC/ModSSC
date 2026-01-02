# Device API

This page documents device utilities used by ModSSC. For configuration context, see the [Configuration reference](../configuration.md).


## What it is for
Device utilities resolve `auto` device selection based on available torch backends. <sup class="cite"><a href="#source-1">[1]</a></sup>


## Examples
Resolve a device name without importing torch explicitly:

```python
from modssc.device import resolve_device_name

print(resolve_device_name("auto"))
```

Resolve with an existing torch module:

```python
import torch
from modssc.device import resolve_device_name

print(resolve_device_name("auto", torch=torch))
```

Device resolution logic is defined in [`src/modssc/device.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/device.py). <sup class="cite"><a href="#source-1">[1]</a></sup>


## API reference

::: modssc.device

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/device.py"><code>src/modssc/device.py</code></a></li>
</ol>
</details>
