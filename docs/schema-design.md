[Documentation home](README.md)

# Schema design

This page covers structuring a program's configuration as a dataclass tree: implicit sections, leaf-level requirements, defaults, and where `ConfigRoot` and validation hooks fit.


## A dataclass tree is the schema

One dataclass declaration serves three roles: the field names define the accepted keys, the annotations define the accepted types, and the defaults define the fallback values. Nested dataclasses define sections, and containers of dataclasses (`list[StageConfig]`, `dict[str, DatasetConfig]`) define repeated sections.


## Implicit sections and leaf-level requirements

Sections declare themselves. A dataclass-typed field with a bare annotation builds automatically when the file omits it, recursively through sub-sections of sub-sections.

That makes a section's required-ness a property of its leaves: every leaf field without a default must come from the file, at its nested position, wherever it sits in the tree. A required leaf inside an omitted section is reported at its dotted path, so the schema author thinks only about which values a run needs, and an error names exactly the value to add.

The implicit build applies to direct dataclass annotations only. Every other undefaulted field stays required when absent:

- scalars (`int`, `str`, `Path`, ...)
- unions, including `Section | None`
- `Any`
- containers (`list[StageConfig]`, `dict[str, DatasetConfig]`, `tuple[int, ...]`)

Containers stay required deliberately. An intentionally empty container is authored as `field(default_factory=list)`, which keeps "forgot to supply `stages`" distinguishable from "this run has zero stages". Elements the file does supply enforce their own required leaves (`stages.0.name`).

A self-referential section (`class Node: child: Node`) terminates with a missing-value issue at the point of recursion and needs an explicit default.

```python
from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from pathlib import Path

from confingo import ConfigRoot


@dataclass
class WarmupConfig:
    steps: int
    start_factor: float = 0.1


@dataclass
class ScheduleConfig:
    warmup: WarmupConfig
    decay: str = "cosine"


@dataclass
class RunConfig(ConfigRoot):
    schedule: ScheduleConfig
    stages: list[str]
    checkpoints: list[Path] = field(default_factory=list)
    seed: int = 0
```

Loading `{}` against this schema reports two issues. `schedule.warmup.steps: missing required value` is the requirement hoisted up through two implicit sections, and `stages: missing required value` is the required container.

A file supplying `{"schedule": {"warmup": {"steps": 500}}, "stages": ["warmup", "main"]}` builds the whole tree: `decay` and `start_factor` from their defaults, `checkpoints` empty by authored intent.


## Leaf defaults and precedence

Defaults form the base layer; the input mapping overrides whichever leaves it supplies.

The two layers are treated differently. Supplied values travel through [coercion](types-and-coercion.md), while defaults are trusted as authored: an omitted field receives exactly the object you wrote in the declaration, byte for byte. Author defaults that already have the annotated type (`Path("runs")` for a `Path` field, `0.1` for a `float` field) so both code paths produce the same shapes.


## Authored factories and partial files

An explicit `field(default_factory=...)` takes precedence over the implicit build and is used as authored, which makes it the tool for baseline sections whose fallback differs from the section's own defaults:

```python
from dataclasses import dataclass, field

from confingo import ConfigRoot


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01


@dataclass
class ExperimentConfig(ConfigRoot):
    seed: int = 0
    batch_size: int = 64
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=1e-3))
```

Partial files fall out of the leaf-level model everywhere: a minimal experiment file overrides two leaves and takes everything else from defaults, implicit or authored:

```json
{"seed": 7, "batch_size": 256}
```

An empty mapping (`{}`) builds the full default config whenever every leaf in the tree has a default, which also makes JSON `null` and empty YAML documents load as pure defaults ([document rules](files-and-identity.md#document-and-read-rules)).


## Root facade and nested sections

Apply `ConfigRoot` to the root class only; child sections stay plain dataclasses.

The base class is a thin facade: each method delegates to the matching free function (`TrainingConfig.load_json(path)` calls `load_json(TrainingConfig, path)`), so both styles are equivalent public surfaces. The full mapping is in the [API reference](api-reference.md#configroot-method-map).


## Runtime-resolvable annotations

Define schema classes at module scope. Annotations are resolved at load time from the defining module's namespace, so module-level classes make forward references (including `from __future__ import annotations` strings) resolvable. `Annotated[T, ...]` fields behave as their base type `T`.

A name that fails to resolve is reported as a schema error with the `config schema` context. See [validation phases](validation-and-errors.md#two-validation-phases).


## Constructor shape and schema preflight

Every field in the tree is constructor-settable (`init=True`, the dataclass default); frozen and ordinary dataclasses both work.

Before coercing any value, `from_dict` runs a recursive schema preflight over every annotation in the tree, including sections that the input omits and fields that will use defaults. An annotation outside the [accepted boundary](types-and-coercion.md#accepted-schema-boundary) produces a `ConfigError` at load time, so schema mistakes surface on the very first load.


## Schema-level invariants

Two hooks let a dataclass enforce invariants that span fields, and both feed the same collect-all error report:

- `__post_init__`: raise `ValueError` or `TypeError` for a hard invariant.
- `__validate__`: return an iterable of message strings; each becomes its own issue.

```python
@dataclass
class TrainerConfig:
    warmup_steps: int
    total_steps: int

    def __validate__(self) -> list[str]:
        messages = []
        if self.warmup_steps > self.total_steps:
            messages.append("warmup_steps must be <= total_steps")
        return messages
```

[Validation and errors](validation-and-errors.md#dataclass-invariants) covers how hook output is collected and where it appears in the report.


---

[Previous: Getting started](getting-started.md) | [Home](README.md) | [Next: Types and coercion](types-and-coercion.md)
