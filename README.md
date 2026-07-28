# confingo

A dataclass-driven configuration toolkit. Define your program's settings once as typed dataclasses, then load them from a config file, with everything validated and coerced against the schema on the way in.

The dataclass declaration is the single source of truth: it serves at once as the schema, the type validator, and the default values. Defaults are validated against the same annotations supplied values are coerced toward, so every authored default that reaches the object has a plain serializable form. Config objects compare by value and are unhashable; `config_hash` is the stable value-identity operation.


## Installation

```bash
pip install confingo
```

Runs on Python 3.12 and newer.


## Quick example

Define the schema as dataclasses. Any of them subclasses `ConfigNode` to get load, save, and hash methods over its own subtree; the `optimizer` section carries a bare annotation and builds itself, so `optimizer.name` is the one value the file must supply.

<!-- canonical:config.py -->
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from confingo import ConfigNode


@dataclass
class OptimizerConfig(ConfigNode):
    name: Literal["adamw", "sgd"]
    lr: float = 3e-4


@dataclass
class TrainingConfig(ConfigNode):
    optimizer: OptimizerConfig
    seed: int = 0
    output_dir: Path = Path("runs")
```

Write a config file that supplies the required value and any leaves that differ from the defaults:

<!-- canonical:train.json -->
```json
{
  "optimizer": {"name": "adamw", "lr": 0.001}
}
```

Load it into a typed, validated object and derive a stable run identity:

```python
config = TrainingConfig.cfg.load_json("train.json")

config.optimizer.lr             # 0.001, coerced to float
config.seed                     # 0, from the default
run_id = config.cfg.hash()   # "344e28a35dd4"
saved = config.cfg.save_json(config.output_dir / run_id / "resolved.json")
saved.as_posix()                # "runs/344e28a35dd4/resolved.json"
```

The runnable version lives in [examples/quickstart/](https://github.com/belfner/confingo/tree/main/examples/quickstart) and is walked through in [Getting started](https://github.com/belfner/confingo/blob/main/docs/getting-started.md).

Validation walks the whole tree in one pass and reports every problem at once, each tagged with a dotted path:

```
confingo.ConfigError: config file train.json has 3 issues:
  - sed: unknown key (known keys: optimizer, output_dir, seed)
  - optimizer.name: expected one of 'adamw' | 'sgd', got 'adam'
  - optimizer.lr: expected float, got str
```


## Arrays and tensors

NumPy arrays and PyTorch tensors work as field types whenever your application already imports the backend; the array/tensor integration activates from that already-imported backend and detects it at runtime. Values serialize as plain JSON data (a scalar for a 0-d value, nested lists otherwise) and rebuild against the annotated dtype, with bare `torch.Tensor` pinned to value-stable dtypes and `Annotated[torch.Tensor, torch.float32]` pinning a specific one. The rules live in [arrays and tensors](https://github.com/belfner/confingo/blob/main/docs/arrays-and-tensors.md).


## Documentation

Full documentation lives in [docs/](https://github.com/belfner/confingo/blob/main/docs/README.md), which offers two routes through one set of pages.

**Essentials** covers everything needed to write, load, save, and debug a config:

- [Getting started](https://github.com/belfner/confingo/blob/main/docs/getting-started.md): the linear introduction, from a runnable example to a realistic training config with defaults, sections, unions, and a run hash.
- [Arrays and tensors](https://github.com/belfner/confingo/blob/main/docs/arrays-and-tensors.md): NumPy and PyTorch fields, dtype and shape choices.
- [Files, formats, and run identity](https://github.com/belfner/confingo/blob/main/docs/files-and-identity.md): JSON, YAML, extension dispatch, atomic saves.
- [Recipes](https://github.com/belfner/confingo/blob/main/docs/recipes.md): copyable answers to common tasks.

**Exact reference** holds the precise rules, for lookup:

- [Schema design](https://github.com/belfner/confingo/blob/main/docs/schema-design.md): implicit sections, leaf-level requirements, defaults, `ConfigNode`.
- [Types and coercion](https://github.com/belfner/confingo/blob/main/docs/types-and-coercion.md): accepted annotations and exact conversion rules.
- [Validation and errors](https://github.com/belfner/confingo/blob/main/docs/validation-and-errors.md): the collect-all error model and every issue source.
- [Equality and hashing](https://github.com/belfner/confingo/blob/main/docs/equality-and-hashing.md): canonical equality, unhashable config objects, stable run identity.
- [API reference](https://github.com/belfner/confingo/blob/main/docs/api-reference.md): signatures for every public name.


## In one line

confingo packages the "config file plus dataclass schema" pattern into a reusable toolkit: a typed marshal / unmarshal pair over plain stdlib dataclasses, with exhaustive error reporting and a reproducible fingerprint.
