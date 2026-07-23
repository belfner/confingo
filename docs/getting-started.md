[Documentation home](README.md)

# Getting started

This page takes you from installation to a loaded, typed training configuration, then saves the resolved config and derives a run identifier. The whole path is copyable and runs on the core install.


## Install

```bash
pip install confingo
```

The core library runs on Python 3.11+ and imports only the standard library. YAML support is an optional extra covered in [Files, formats, and run identity](files-and-identity.md#yaml-extra).


## Define the schema

A confingo schema is a tree of dataclasses declared with `@configclass`, a drop-in wrapper around `@dataclass` that adds [config-aware equality](schema-design.md#configclass-and-equality). Each field's annotation is its validator and each default is its fallback value. The root class subclasses `ConfigRoot` to gain load/save/hash methods; nested sections carry the decorator alone.

The sections (`model`, `data`, `optimizer`) carry bare annotations and build automatically; what the file must supply is decided by the fields inside them. Here the required values are the four fields declared with a bare annotation and no default:

- `model.architecture`
- `model.hidden_widths`
- `data.dataset_path`
- `optimizer.name`

Omitting one is reported at that dotted path, and required fields come before defaulted ones. See [implicit sections](schema-design.md#implicit-sections-and-leaf-level-requirements) for the full rules.

```python
from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import Literal

from confingo import ConfigRoot, configclass


@configclass
class ModelConfig:
    architecture: str
    hidden_widths: tuple[int, ...]
    dropout: float = 0.1


@configclass
class DataConfig:
    dataset_path: Path
    batch_size: int = 64
    num_workers: int = 4
    augmentations: list[str] = field(default_factory=list)


@configclass
class OptimizerConfig:
    name: Literal["adamw", "sgd"]
    lr: float = 3e-4
    weight_decay: float = 0.01


@configclass
class TrainingConfig(ConfigRoot):
    model: ModelConfig
    data: DataConfig
    optimizer: OptimizerConfig
    seed: int = 0
    device: Literal["cpu", "cuda"] = "cpu"
    output_dir: Path = Path("runs")
```

Define schema classes at module level so their annotations resolve at load time.


## Write `train.json`

The file supplies every required value and any leaves that differ from the defaults. Omitted leaves (`dropout`, `num_workers`, `weight_decay`, `seed`, `device`, `output_dir`) fall back to their declared defaults.

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

Every value has been coerced toward its annotation: the JSON array became a `tuple[int, ...]`, the string became a `Path`, and `optimizer.name` was checked against its `Literal` options. The full conversion rules live in [Types and coercion](types-and-coercion.md).


## Save the resolved run and assign an identity

Saving writes the resolved in-memory object, defaults included, so the output file is a complete record of the run:

```python
run_id = config.config_hash()          # "8e6ea26c7116"
run_dir = config.output_dir / run_id
config.save_json(run_dir / "resolved.json")
```

`config_hash` is a stable fingerprint of the resolved config: two processes holding equal configs produce the same hash, which makes it a natural run directory name. Details in [stable run identity](files-and-identity.md#stable-run-identity).


## When loading fails

confingo validates the whole tree in one pass and reports every problem at once, each tagged with its dotted path:

```
confingo.ConfigError: config file train.json has 4 issues:
  - sed: unknown key (known keys: data, device, model, optimizer, output_dir, seed)
  - data.batch_sizes: unknown key (known keys: augmentations, batch_size, dataset_path, num_workers)
  - optimizer.name: expected one of 'adamw' | 'sgd', got 'adam'
  - optimizer.lr: expected float, got str
```

[Validation and errors](validation-and-errors.md) covers the error model and custom validation hooks.


## Free-function equivalent

Every `ConfigRoot` method has a free-function twin, so a plain dataclass root works the same way:

```python
from confingo import config_hash, load_json, save_json

config = load_json(TrainingConfig, "train.json")
save_json(config, "resolved.json")
config_hash(config)
```

The [API reference](api-reference.md#configroot-method-map) maps each method to its function.


## Next steps

- [Schema design](schema-design.md): structuring larger config trees, factory defaults, partial files.
- [Types and coercion](types-and-coercion.md): the accepted types and exact conversion rules.
- [Validation and errors](validation-and-errors.md): the error model and custom invariants.
- [Files, formats, and run identity](files-and-identity.md): YAML, extension dispatch, atomic saves, hashing.


---

[Home](README.md) | [Next: Schema design](schema-design.md)
