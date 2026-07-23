# confingo

A dataclass-driven configuration toolkit. Define your program's settings once as typed dataclasses, then load them from a config file, with everything validated and coerced against the schema on the way in.

The dataclass declaration is the single source of truth: it serves at once as the schema, the type validator, and the default values.


## Installation

```bash
pip install confingo
```

Runs on Python 3.11 and newer.


## Quick example

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from confingo import ConfigRoot


@dataclass
class TrainerConfig:
    lr: float = 3e-4
    algorithm: Literal["sac", "td3"] = "td3"


@dataclass
class TrainingConfig(ConfigRoot):
    trainer: TrainerConfig  # section; builds from its defaults when omitted
    seed: int = 0
    output_dir: Path = Path("runs")


config = TrainingConfig.load_json("config.json")   # typed, validated tree
run_id = config.config_hash()                      # stable fingerprint
config.save_json(config.output_dir / run_id / "resolved.json")
```

Validation walks the whole tree in one pass and reports every problem at once, each tagged with a dotted path:

```
confingo.ConfigError: config file config.json has 2 issues:
  - sed: unknown key (known keys: output_dir, seed, trainer)
  - trainer.lr: expected float, got str
```


## Arrays and tensors

NumPy arrays and PyTorch tensors work as field types whenever your application already imports the backend; confingo stays stdlib-only and detects them at runtime. Values serialize as plain JSON data (a scalar for a 0-d value, nested lists otherwise) and rebuild against the annotated dtype, with bare `torch.Tensor` pinned to value-stable dtypes and `Annotated[torch.Tensor, torch.float32]` pinning a specific one. The rules live in [types and coercion](docs/types-and-coercion.md#arrays-and-tensors).


## Documentation

Full documentation lives in [docs/](docs/README.md):

- [Getting started](docs/getting-started.md): install to a loaded, hashed training config in five minutes.
- [Schema design](docs/schema-design.md): implicit sections, leaf-level requirements, defaults, `ConfigRoot`.
- [Types and coercion](docs/types-and-coercion.md): accepted annotations and exact conversion rules.
- [Validation and errors](docs/validation-and-errors.md): the collect-all error model and custom invariants.
- [Files, formats, and run identity](docs/files-and-identity.md): JSON, YAML, extension dispatch, atomic saves, `config_hash`.
- [API reference](docs/api-reference.md): signatures for every public name.


## In one line

confingo packages the "config file plus dataclass schema" pattern into a reusable toolkit: a typed marshal / unmarshal pair over plain stdlib dataclasses, with exhaustive error reporting and a reproducible fingerprint.
