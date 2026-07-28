[Documentation home](README.md)

# Getting started

This page starts with a complete, runnable example you can copy in one minute, then grows that same schema into a realistic training configuration, saves the resolved config, and derives a run identifier. Everything here runs on the core install, apart from the array field under [array fields](#fixed-size-groups-and-array-fields), which uses the NumPy your application already imports.


## Install

```bash
pip install confingo
```

The library runs on Python 3.12+. JSON and YAML file IO both work from the base install; YAML is covered in [Files, formats, and run identity](files-and-identity.md#yaml).


## Minimal example

Three files make a complete confingo program: a schema, a config file, and a script that loads it.

```
quickstart/
  config.py
  train.json
  run.py
```

The schema is a tree of `@dataclass` declarations. Each field's annotation is its validator and each default is its fallback value. A class subclasses `ConfigNode` to get load, save, and hash methods; here both classes do, so the optimizer section carries them over its own subtree too. The `optimizer` field carries a bare annotation and builds itself, so `optimizer.name` is the one value the file must supply.

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

The config file supplies the required value and any leaves that differ from the defaults. Declared defaults form the base layer, and the file overrides the leaves it names: here `lr` is set explicitly while `seed` and `output_dir` stay at their declared values. See [defaults and precedence](schema-design.md#leaf-defaults-and-precedence).

<!-- canonical:train.json -->
```json
{
  "optimizer": {"name": "adamw", "lr": 0.001}
}
```

The script loads the file into a typed object, reads coerced values, derives a run identifier, and saves the resolved config.

<!-- canonical:run.py -->
```python
from __future__ import annotations

from config import TrainingConfig


def main() -> None:
    config = TrainingConfig.cfg.load_json("train.json")

    print(f"optimizer.name: {config.optimizer.name}")
    print(f"optimizer.lr: {config.optimizer.lr}")
    print(f"seed: {config.seed}")

    # OptimizerConfig is a ConfigNode too, so it fingerprints its own section.
    print(f"optimizer id: {config.optimizer.cfg.hash()}")

    run_id = config.cfg.hash()
    print(f"run id: {run_id}")

    resolved = config.cfg.save_json(config.output_dir / run_id / "resolved.json")
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

Every value arrived typed: the JSON number became a `float`, `output_dir` is a `Path`, and `optimizer.name` was checked against its `Literal` options. The saved `runs/344e28a35dd4/resolved.json` records the resolved config in full, defaults included:

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

That is the whole loop: define a schema, write a file, load it typed, and save a resolved snapshot keyed by a stable hash. The rest of this page grows the same schema toward a realistic training run.


## Grow the schema

A real training config carries more sections. The `optimizer` section from the minimal example stays; `model` and `data` join it, each a bare-annotated section that builds from its own leaves. The required values are the bare-annotated fields, which the file must supply:

- `model.architecture`
- `model.hidden_widths`
- `data.dataset_path`
- `optimizer.name`

Required-field issues carry these dotted paths, and required fields come before defaulted ones. See [implicit sections](schema-design.md#implicit-sections-and-leaf-level-requirements) for the full rules.

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

Define schema classes at module level so their annotations resolve at load time. A schema names the concrete types a config file carries, so a class that takes type parameters (`class Config[T]`) is reported at preflight with the concrete type to write in its place.

A file for this schema supplies every required value and any leaves that differ from the defaults. The declared defaults supply the remaining leaves (`dropout`, `num_workers`, `weight_decay`, `seed`, `device`, `output_dir`).

```json
{
  "model": {"architecture": "mlp", "hidden_widths": [256, 128]},
  "data": {"dataset_path": "data/cifar10", "batch_size": 128},
  "optimizer": {"name": "adamw", "lr": 0.001}
}
```


## Choose an annotation

The annotation decides what a field accepts and what it becomes. These cover almost every config field:

| You want | Write | The file carries |
| --- | --- | --- |
| a number, a flag, a name | `int`, `float`, `bool`, `str` | the matching JSON value |
| a filesystem location | `Path` | a string |
| one of a fixed set of names | `Literal["adamw", "sgd"]` | one of those strings |
| a named set of options with behavior | an `Enum` subclass | the member's value |
| a variable-length list | `list[str]` | an array |
| a fixed-size group | `tuple[int, int]` | an array of exactly that length |
| any-length homogeneous group | `tuple[int, ...]` | an array |
| named sub-values | `dict[str, float]` | an object with string keys |
| a nested section | the section's dataclass | a nested object |
| a value that may be absent | `T \| None` | the value or `null` |
| a timestamp or date | `datetime`, `date`, `time` | an ISO 8601 string |
| open-ended plain data | `ConfigValue` | whatever plain JSON it holds |

[Types and coercion](types-and-coercion.md) has the full table and the exact conversion rules.


## Load and use typed values

```python
config = TrainingConfig.cfg.load_json("train.json")

config.model.hidden_widths   # (256, 128), a tuple per the annotation
config.data.dataset_path     # Path("data/cifar10")
config.optimizer.lr          # 0.001
config.device                # "cpu", from the default
```

Every value has been coerced toward its annotation: the JSON array became a `tuple[int, ...]`, the string became a `Path`, and `optimizer.name` was checked against its `Literal` options. The full conversion rules live in [Types and coercion](types-and-coercion.md).


## Write defaults in the annotated type

A supplied value is converted toward its annotation; a default is used exactly as you wrote it. So a default has to already be the thing the annotation names, and confingo checks that when it first reads the schema:

```python
@dataclass
class Paths:
    output_dir: Path = "runs"          # reported: a str where Path is annotated
    checkpoint_dir: Path = Path("ckpt")  # correct
```

```text
config has 1 issue:
  - output_dir: invalid authored default: expected a value already matching Path, got str; defaults are validated as written
```

The same rule covers `ratio: float = 1` (write `1.0`), a list where a tuple is annotated (write `("a", "b")`), and a mapping where a section is annotated (write `OptimizerConfig(name="sgd", lr=1e-3)`, a complete instance of the section's own class). The check runs on every authored default, so a wrong one surfaces in a project that always overrides it.

Lists, dicts, and sections are held through a factory, which Python requires for any mutable default:

```python
@dataclass
class DataConfig:
    dataset_path: Path
    augmentations: list[str] = field(default_factory=list)
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(name="sgd"))
```

The factory runs once, at the load that needs it, and its result goes through the same validation. A `list[str]` field the file omits is required unless it carries a factory, so `field(default_factory=list)` is how you say a container starts empty on purpose.


## Fixed-size groups and array fields

A `tuple` with a fixed length is checked for that length, so a truncated pair reports at load time:

```python
@dataclass
class ScheduleConfig:
    warmup_and_total: tuple[int, int] = (500, 10_000)
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
```

For real numeric payloads, annotate a NumPy array or a Torch tensor and confingo builds it from the JSON array, checking dtype, shape, and finiteness:

```python
import numpy as np
import numpy.typing as npt


@dataclass
class NormalizationConfig:
    channel_mean: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
```

The backends activate only when your application imports NumPy or PyTorch itself. [Arrays and tensors](arrays-and-tensors.md) covers dtype and shape choices.


## Let the file choose between two shapes

A union lets one field hold either of two sections. Give each a `Literal` discriminator so the file states which one it means:

```python
@dataclass
class AdamW:
    kind: Literal["adamw"]
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class SGD:
    kind: Literal["sgd"]
    lr: float = 1e-3
    momentum: float = 0.9


@dataclass
class RunConfig(ConfigNode):
    optimizer: AdamW | SGD
```

Members are tried in declaration order and the first that fits cleanly wins. When every member fails, the report names the branch that came closest and shows that branch's problems:

```text
config has 2 issues:
  - optimizer: expected AdamW | SGD; best match SGD failed with 1 issue
  - optimizer.lr: expected float, got str
```


## Derive values and check invariants

A computed field is declared `init=False` and set in `__post_init__`. It is runtime state, so the file, `to_dict`, and the run hash all range over the configured fields:

```python
@dataclass
class ScheduleConfig:
    total_steps: int = 10_000
    warmup_fraction: float = 0.1
    warmup_steps: int = field(init=False)

    def __post_init__(self) -> None:
        self.warmup_steps = int(self.total_steps * self.warmup_fraction)
```

For a rule spanning several fields, add `__validate__` returning one message per problem. Each becomes its own issue in the same report as everything else:

```python
@dataclass
class SplitConfig:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __validate__(self) -> list[str]:
        problems = []
        if abs(self.train + self.val + self.test - 1.0) > 1e-9:
            problems.append("train, val, and test must sum to 1.0")
        if self.val <= 0.0:
            problems.append("val must be positive to measure progress")
        return problems
```

`__validate__` returns an iterable of messages, and an empty list when the config is valid. Returning a bare string or `None` is reported as the contract slip it is rather than acted on, and an exception raised inside the hook becomes one issue beside the rest of the report. The same holds for `__post_init__` and for a `default_factory`: whatever they raise describes the config, so it arrives with every other problem in one `ConfigError`.


## Save the resolved run and assign an identity

Saving writes the resolved in-memory object, defaults included, so the output file is a complete record of the run:

```python
run_id = config.cfg.hash()          # "8e6ea26c7116"
run_dir = config.output_dir / run_id
config.cfg.save_json(run_dir / "resolved.json")
```

`config_hash` is a stable fingerprint of the resolved config: two processes holding equal configs produce the same hash, which makes it a natural run directory name. Details in [stable run identity](files-and-identity.md#stable-run-identity).


## Read an error in three steps

confingo validates the whole tree in one pass and reports every problem at once, so one run of a broken file tells you everything to fix:

```
confingo.ConfigError: config file train.json has 4 issues:
  - sed: unknown key (known keys: data, device, model, optimizer, output_dir, seed)
  - data.batch_sizes: unknown key (known keys: augmentations, batch_size, dataset_path, num_workers)
  - optimizer.name: expected one of 'adamw' | 'sgd', got 'adam'
  - optimizer.lr: expected float, got str
```

Read each line the same way:

1. **The summary line** names the source and the count. `config file train.json` is the file to open, and four issues means four independent edits, since each line reports a separate value.
2. **The path** locates the value. `optimizer.lr` is the `lr` key inside the `optimizer` object. A `<root>` path means the document itself.
3. **The message** states what was expected and what arrived, and every message that can name a fix does. `unknown key` lists the keys that exist, so a typo like `sed` for `seed` is visible in the same line.

[Validation and errors](validation-and-errors.md) covers the error model, every built-in issue source, and custom validation hooks.


## Call methods on a section

`OptimizerConfig` subclasses `ConfigNode` too, so it carries the same methods over its own subtree:

```python
config.optimizer.cfg.to_dict()      # {'name': 'adamw', 'lr': 0.001, 'weight_decay': 0.01}
config.optimizer.cfg.hash()  # "41263e6f3612"
config.optimizer.cfg.save_json("optimizer.json")
OptimizerConfig.cfg.load_json("optimizer.json")
```

Issue paths follow the same scope. Building the section on its own reports its leaves relative to it:

```python
OptimizerConfig.cfg.from_dict({"lr": "fast"})
# config has 2 issues:
#   - name: missing required value
#   - lr: expected float, got str
```

A section that stays a plain dataclass keeps working exactly as before; the free functions cover it.


## Free-function equivalent

Every `ConfigNode` method has a free-function twin, so a plain dataclass works the same way:

```python
from confingo.functional import config_hash, load_json, save_json

config = load_json(TrainingConfig, "train.json")
save_json(config, "resolved.json")
config_hash(config)
```

The [API reference](api-reference.md#confignode-method-map) maps each method to its function.


## Where to go next

You now have everything needed to write a real config. Pick by the job in front of you:

**I want to do a specific thing**

- Load or save YAML, dispatch by extension, or write a resolved snapshot: [Files, formats, and run identity](files-and-identity.md).
- Hold a NumPy array or a Torch tensor: [Arrays and tensors](arrays-and-tensors.md).
- Layer a base config with per-experiment overrides, or key runs by hash: [Recipes](recipes.md).

**I want to know exactly what confingo will do**

- What each annotation accepts and what it becomes: [Types and coercion](types-and-coercion.md).
- How sections, defaults, factories, and field options interact: [Schema design](schema-design.md).
- Every issue confingo can report, and when: [Validation and errors](validation-and-errors.md).
- What `==` compares and what `config_hash` covers: [Equality and hashing](equality-and-hashing.md).
- The signature of a specific function: [API reference](api-reference.md).


---

[Documentation home](README.md)
