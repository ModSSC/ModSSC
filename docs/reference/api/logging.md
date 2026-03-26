# Runtime logging API

This page documents logging utilities exposed from the `modssc.runtime` package. For CLI flags, see the [CLI reference](../cli.md).


## What it is for
The logging helpers configure consistent log levels across ModSSC and bench modules. <sup class="cite"><a href="#source-1">[1]</a></sup>


## Examples
Resolve a log level and configure logging:

```python
from modssc.runtime.logging import configure_logging, resolve_log_level

level = resolve_log_level("detailed")
configure_logging(level)
```

Use the CLI option format:

```python
from modssc.runtime.logging import normalize_log_level

print(normalize_log_level("full"))
```

Logging helpers are implemented in [`src/modssc/runtime/logging.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/runtime/logging.py) and re-exported from [`src/modssc/runtime/__init__.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/runtime/__init__.py). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## API reference

::: modssc.runtime.logging

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/runtime/logging.py"><code>src/modssc/runtime/logging.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/runtime/__init__.py"><code>src/modssc/runtime/__init__.py</code></a></li>
</ol>
</details>
