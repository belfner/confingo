[Documentation home](README.md)

# Recipes

Copyable answers to common confingo tasks. Each recipe gives the code, the result,
and a link to the concept page that carries the exact rules. The recipes share the
quickstart `TrainingConfig` from [Getting started](getting-started.md) (root
`optimizer` section, `seed`, `output_dir`).

- [Load and save](#load-and-save)
- [Choose a surface](#choose-a-surface)
- [Resolve and identify a run](#resolve-and-identify-a-run)
- [Compose and diagnose](#compose-and-diagnose)


## Load and save

### Load a config file

Build a typed config from JSON or YAML, or route by suffix when the format is
data-driven:

```python
config = TrainingConfig.load_json("train.json")
config = TrainingConfig.load_yaml("train.yaml")
config = TrainingConfig.from_file(path)        # .json / .yaml / .yml by suffix
```

Each call returns a validated `TrainingConfig` with every value coerced to its
annotation. See [Files, formats, and run identity](files-and-identity.md) and
[extension dispatch](files-and-identity.md#extension-dispatch).

### Save a config file

Write the resolved object back out, choosing the format directly or by suffix:

```python
config.save_json("resolved.json")              # atomic write, returns the Path
config.save_yaml("resolved.yaml")              # sort_keys=False by default
config.to_file(run_dir / "resolved.json")      # by suffix
```

The save is atomic and returns the destination `Path`. See
[JSON](files-and-identity.md#json) and [YAML](files-and-identity.md#yaml).


## Choose a surface

### Use methods or free functions

A `ConfigNode` subclass carries methods; the free functions take the class as a
parameter. Both surfaces produce the same object:

```python
# method surface
config = TrainingConfig.load_json("train.json")
run_id = config.config_hash()

# free-function surface
from confingo import config_hash, load_json

config = load_json(TrainingConfig, "train.json")
run_id = config_hash(config)                   # same digest, "344e28a35dd4"
```

The method style reads well for a class that owns the operation scope; the free
functions suit library code that receives the class. See the
[method map](api-reference.md#confignode-method-map).


### Save or fingerprint one section

A section that subclasses `ConfigNode` carries the same methods over its own
subtree, so it can be persisted or identified on its own:

```python
config = TrainingConfig.load_json("train.json")

optimizer_id = config.optimizer.config_hash()   # "be59896dec38"
config.optimizer.save_json(run_dir / "optimizer.json")

# Reload that snapshot through the section's own class.
optimizer = OptimizerConfig.load_json(run_dir / "optimizer.json")
```

Issue paths from a section load are relative to that section, so `lr` locates the
value in the mapping that call received. Pass `context=` to label the source:

```python
OptimizerConfig.from_dict(overrides, context="optimizer override")
```


## Resolve and identify a run

### Name a run directory by its hash

Equal resolved configs share a hash across processes, so it makes a stable
directory name, and saving there records the full resolved config:

```python
from pathlib import Path

run_id = config.config_hash()                  # "344e28a35dd4"
run_dir = Path("runs") / run_id
run_dir.mkdir(parents=True, exist_ok=True)
config.save_json(run_dir / "resolved.json")    # runs/344e28a35dd4/resolved.json
```

The saved file carries defaults and overrides together, a complete record of the
run. See [resolved snapshots](files-and-identity.md#resolved-snapshots) and
[stable run identity](equality-and-hashing.md#stable-run-identity).

### Fan out a sweep

Each override set builds its own config and lands in its own hash-named directory.
Compose layers with `deep_merge` (below) so an override that touches part of a
section keeps the rest:

```python
for overrides in sweep:                        # e.g. {"optimizer": {"lr": 1e-3}}
    merged = deep_merge(base, overrides)
    config = TrainingConfig.from_dict(merged)
    run_dir = Path("runs") / config.config_hash()
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save_json(run_dir / "resolved.json")
```

Distinct hashing values give distinct directories. See
[stable run identity](equality-and-hashing.md#stable-run-identity).


## Compose and diagnose

### Overlay a base mapping with overrides

`from_dict` takes one mapping, so merge layers into plain data first. A recursive
merge keeps a section an override touches only in part:

```python
def deep_merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


base = {"optimizer": {"name": "adamw", "lr": 3e-4}}
config = TrainingConfig.from_dict(deep_merge(base, {"optimizer": {"lr": 1e-3}}))
config.optimizer.name    # "adamw", kept from the base layer
config.optimizer.lr      # 0.001, from the override layer
```

The recursive merge preserves `optimizer.name` while overriding `lr`; a shallow
`{**base, **overrides}` would replace the whole `optimizer` section. See
[defaults and precedence](schema-design.md#leaf-defaults-and-precedence).

### Inspect every reported issue

`ConfigError.issues` holds one `ConfigIssue` per problem, each with a dotted path:

```python
from confingo import ConfigError

try:
    config = TrainingConfig.from_dict({"optimizer": {"name": "adam", "lr": "fast"}})
except ConfigError as err:
    for issue in err.issues:
        print(f"{issue.path}: {issue.message}")
```

```
optimizer.name: expected one of 'adamw' | 'sgd', got 'adam'
optimizer.lr: expected float, got str
```

One pass reports both problems together. See
[`ConfigError` and `ConfigIssue`](validation-and-errors.md#configerror-and-configissue).

### Add a cross-field invariant

`__validate__` returns a message per problem, and each joins the collect-all report
at the dataclass's path:

```python
from dataclasses import dataclass


@dataclass
class ScheduleConfig:
    warmup_steps: int
    total_steps: int

    def __validate__(self) -> list[str]:
        messages = []
        if self.warmup_steps > self.total_steps:
            messages.append("warmup_steps must be <= total_steps")
        return messages
```

Building `ScheduleConfig` with `warmup_steps=100, total_steps=10` reports
`warmup_steps must be <= total_steps` at the section's path. See
[dataclass invariants](validation-and-errors.md#dataclass-invariants).


---

[Getting started](getting-started.md) | [Home](README.md) | [Validation and errors](validation-and-errors.md)
