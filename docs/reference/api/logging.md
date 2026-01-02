# Logging API

This page documents logging utilities. For CLI flags, see the [CLI reference](../cli.md).


## What it is for
The logging helpers configure consistent log levels across ModSSC and bench modules. <sup class="cite"><a href="#source-1">[1]</a></sup>


## Examples
Resolve a log level and configure logging:

```python
from modssc.logging import configure_logging, resolve_log_level

level = resolve_log_level("detailed")
configure_logging(level)
```

Use the CLI option format:

```python
from modssc.logging import normalize_log_level

print(normalize_log_level("full"))
```

Log level aliases and configuration are defined in [`src/modssc/logging.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/logging.py). <sup class="cite"><a href="#source-1">[1]</a></sup>


## API reference

::: modssc.logging

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/logging.py"><code>src/modssc/logging.py</code></a></li>
</ol>
</details>
