[Documentation home](README.md)

# Getting started

This page starts with a complete, runnable example you can copy in one minute,
then grows that same schema into a realistic training configuration, saves the
resolved config, and derives a run identifier. Everything here runs on the core
install.


## Install

```bash
pip install confingo
```

The library runs on Python 3.11+. JSON and YAML file IO both work from the base
install; YAML is covered in [Files, formats, and run identity](files-and-identity.md#yaml).


## Minimal example

Three files make a complete confingo program: a schema, a config file, and a
script that loads it.

```
quickstart/
  config.py
  train.json
  run.py
```

The schema is a tree of `@dataclass` declarations. Each field's annotation is its
validator and each default is its fallback value. A class subclasses `ConfigNode`
to get load, save, and hash methods; here both classes do, so the optimizer
section carries them over its own subtree too. The `optimizer` field carries a
bare annotation and builds itself, so `optimizer.name` is the one value the file
must supply.

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

The config file supplies the required value and any leaves that differ from the
defaults. Declared defaults form the base layer, and the file overrides the leaves
it names: here `lr` is set explicitly while `seed` and `output_dir` stay at their
declared values. See [defaults and precedence](schema-design.md#leaf-defaults-and-precedence).

<!-- canonical:train.json -->
```json
{
  "optimizer": {"name": "adamw", "lr": 0.001}
}
```

The script loads the file into a typed object, reads coerced values, derives a
run identifier, and saves the resolved config.

<!-- canonical:run.py -->
```python
from __future__ import annotations

from config import TrainingConfig


def main() -> None:
    config = TrainingConfig.load_json("train.json")

    print(f"optimizer.name: {config.optimizer.name}")
    print(f"optimizer.lr: {config.optimizer.lr}")
    print(f"seed: {config.seed}")

    # OptimizerConfig is a ConfigNode too, so it fingerprints its own section.
    print(f"optimizer id: {config.optimizer.config_hash()}")

    run_id = config.config_hash()
    print(f"run id: {run_id}")

    resolved = config.save_json(config.output_dir / run_id / "resolved.json")
    print(f"saved: {resolved}")


if __name__ == "__main__":
    main()
```

Run it from the `quickstart/` directory:

```console
$ python run.py
optimizer.name: adamw
optimizer.lr: 0.001
seed: 0
optimizer id: be59896dec38
run id: 344e28a35dd4
saved: runs/344e28a35dd4/resolved.json
```

Every value arrived typed: the JSON number became a `float`, `output_dir` is a
`Path`, and `optimizer.name` was checked against its `Literal` options. The saved
`runs/344e28a35dd4/resolved.json` records the resolved config in full, defaults
included:

```json
{
  "optimizer": {
    "name": "adamw",
    "lr": 0.001
  },
  "seed": 0,
  "output_dir": "runs"
}
```

That is the whole loop: define a schema, write a file, load it typed, and save a
resolved snapshot keyed by a stable hash. The rest of this page grows the same
schema toward a realistic training run.


## Grow the schema

A real training config carries more sections. The `optimizer` section from the
minimal example stays; `model` and `data` join it, each a bare-annotated section
that builds from its own leaves. The required values are the bare-annotated
fields, which the file must supply:

- `model.architecture`
- `model.hidden_widths`
- `data.dataset_path`
- `optimizer.name`

Required-field issues carry these dotted paths, and required fields come before
defaulted ones. See [implicit sections](schema-design.md#implicit-sections-and-leaf-level-requirements)
for the full rules.

```python
from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from pathlib import Path
from typing import Literal

from confingo import ConfigNode


@dataclass
class ModelConfig:
    architecture: str
    hidden_widths: tuple[int, ...]
    dropout: float = 0.1


@dataclass
class DataConfig:
    dataset_path: Path
    batch_size: int = 64
    num_workers: int = 4
    augmentations: list[str] = field(default_factory=list)


@dataclass
class OptimizerConfig(ConfigNode):
    name: Literal["adamw", "sgd"]
    lr: float = 3e-4
    weight_decay: float = 0.01


@dataclass
class TrainingConfig(ConfigNode):
    model: ModelConfig
    data: DataConfig
    optimizer: OptimizerConfig
    seed: int = 0
    device: Literal["cpu", "cuda"] = "cpu"
    output_dir: Path = Path("runs")
```

Define schema classes at module level so their annotations resolve at load time.

A file for this schema supplies every required value and any leaves that differ
from the defaults. The declared defaults supply the remaining leaves (`dropout`,
`num_workers`, `weight_decay`, `seed`, `device`, `output_dir`).

```json
{
  "model": {"architecture": "mlp", "hidden_widths": [256, 128]},
  "data": {"dataset_path": "data/cifar10", "batch_size": 128},
  "optimizer": {"name": "adamw", "lr": 0.001}
}
```


## Load and use typed values

```python
config = TrainingConfig.load_json("train.json")

config.model.hidden_widths   # (256, 128), a tuple per the annotation
config.data.dataset_path     # Path("data/cifar10")
config.optimizer.lr          # 0.001
config.device                # "cpu", from the default
```

Every value has been coerced toward its annotation: the JSON array became a
`tuple[int, ...]`, the string became a `Path`, and `optimizer.name` was checked
against its `Literal` options. The full conversion rules live in
[Types and coercion](types-and-coercion.md).


## Save the resolved run and assign an identity

Saving writes the resolved in-memory object, defaults included, so the output
file is a complete record of the run:

```python
run_id = config.config_hash()          # "8e6ea26c7116"
run_dir = config.output_dir / run_id
config.save_json(run_dir / "resolved.json")
```

`config_hash` is a stable fingerprint of the resolved config: two processes
holding equal configs produce the same hash, which makes it a natural run
directory name. Details in [stable run identity](files-and-identity.md#stable-run-identity).


## When loading fails

confingo validates the whole tree in one pass and reports every problem at once,
each tagged with its dotted path:

```
confingo.ConfigError: config file train.json has 4 issues:
  - sed: unknown key (known keys: data, device, model, optimizer, output_dir, seed)
  - data.batch_sizes: unknown key (known keys: augmentations, batch_size, dataset_path, num_workers)
  - optimizer.name: expected one of 'adamw' | 'sgd', got 'adam'
  - optimizer.lr: expected float, got str
```

[Validation and errors](validation-and-errors.md) covers the error model and
custom validation hooks.


## Call methods on a section

`OptimizerConfig` subclasses `ConfigNode` too, so it carries the same methods
over its own subtree:

```python
config.optimizer.to_dict()      # {'name': 'adamw', 'lr': 0.001, 'weight_decay': 0.01}
config.optimizer.config_hash()  # "41263e6f3612"
config.optimizer.save_json("optimizer.json")
OptimizerConfig.load_json("optimizer.json")
```

Issue paths follow the same scope. Building the section on its own reports its
leaves relative to it:

```python
OptimizerConfig.from_dict({"lr": "fast"})
# config has 2 issues:
#   - name: missing required value
#   - lr: expected float, got str
```

A section that stays a plain dataclass keeps working exactly as before; the free
functions cover it.


## Free-function equivalent

Every `ConfigNode` method has a free-function twin, so a plain dataclass
works the same way:

```python
from confingo import config_hash, load_json, save_json

config = load_json(TrainingConfig, "train.json")
save_json(config, "resolved.json")
config_hash(config)
```

The [API reference](api-reference.md#confignode-method-map) maps each method to
its function.


## Next steps

- [Recipes](recipes.md): copyable answers to common tasks.
- [Schema design](schema-design.md): structuring larger config trees, factory defaults, partial files.
- [Types and coercion](types-and-coercion.md): the accepted types and exact conversion rules.
- [Validation and errors](validation-and-errors.md): the error model and custom invariants.
- [Files, formats, and run identity](files-and-identity.md): YAML, extension dispatch, atomic saves, hashing.


---

[Home](README.md) | [Next: Schema design](schema-design.md)
