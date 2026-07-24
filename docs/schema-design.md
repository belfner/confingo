[Documentation home](README.md)

# Schema design

This page covers structuring a program's configuration as a dataclass tree: implicit sections, leaf-level requirements, defaults, authored factories, `ConfigRoot`, field options, validation hooks, and a summary of canonical equality.


## A dataclass tree is the schema

One dataclass declaration serves three roles: the field names define the accepted keys, the annotations define the accepted types, and the defaults define the fallback values. Nested dataclasses define sections, and containers of dataclasses (`list[StageConfig]`, `dict[str, DatasetConfig]`) define repeated sections.

Schema classes are ordinary `@dataclass` declarations. The root subclasses `ConfigRoot` for load/save/hash methods and config-aware equality, summarized in [canonical equality](#canonical-equality) below; sections are plain dataclasses.


## Implicit sections and leaf-level requirements

Sections declare themselves. A dataclass-typed field with a bare annotation defaults to an automatic build, recursively through sub-sections of sub-sections.

That makes a section's required-ness a property of its leaves: every required leaf must come from the file, at its nested position, wherever it sits in the tree. A required leaf reached through an implicit build is reported at its dotted path, so the schema author thinks only about which values a run needs, and an error names exactly the value to add.

The implicit build applies to direct dataclass annotations only. Every other field type is required, and a default makes it optional:

- scalars (`int`, `str`, `Path`, ...)
- unions, including `Section | None`
- `Any`
- containers (`list[StageConfig]`, `dict[str, DatasetConfig]`, `tuple[int, ...]`)

Containers stay required deliberately. An intentionally empty container is authored as `field(default_factory=list)`, which keeps an authored-empty container distinct from a required one. Elements the file does supply enforce their own required leaves (`stages.0.name`).

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

The two layers are treated differently. Supplied values travel through [coercion](types-and-coercion.md), while defaults are trusted as authored: a defaulted field receives exactly the object you wrote in the declaration, byte for byte. Author defaults that already have the annotated type (`Path("runs")` for a `Path` field, `0.1` for a `float` field) so both code paths produce the same shapes.


## Authored factories and partial files

An explicit `field(default_factory=...)` takes precedence over the implicit build and is used as authored, which makes it the tool for baseline sections whose fallback differs from the section's own defaults:

```python
from dataclasses import (
    dataclass,
    field,
)

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

Apply `ConfigRoot` to the root class only; child sections are plain dataclasses.

The base class is a thin facade: each method delegates to the matching free function (`TrainingConfig.load_json(path)` calls `load_json(TrainingConfig, path)`), so both styles are equivalent public surfaces. The full mapping is in the [API reference](api-reference.md#configroot-method-map).


## Runtime-resolvable annotations

Define schema classes at module scope. Annotations are resolved at load time from the defining module's namespace, so module-level classes make forward references (including `from __future__ import annotations` strings) resolvable. `Annotated[T, ...]` fields behave as their base type `T`.

A name that fails to resolve is reported as a schema error with the `config schema` context. See [validation phases](validation-and-errors.md#two-validation-phases).


## Field options

`init` is the master switch. An `init=True` field (the dataclass default) is loaded from the config, exported by `to_dict`, and — subject to `compare` and `hash` below — weighed by equality and `config_hash`. An `init=False` field is runtime state that its default or `__post_init__` populates; loading, export, equality, and the fingerprint all draw from the `init=True` fields. On an `init=True` field, `compare` and `hash` scope equality and the fingerprint:

| Field | Loaded | In `to_dict` | In equality | In `config_hash` |
| --- | :---: | :---: | :---: | :---: |
| `field()` (init=True default) | yes | yes | yes | yes |
| `field(init=False)` | no | no | no | no |
| `field(compare=False)` | yes | yes | no | no |
| `field(hash=False)` | yes | yes | yes | no |
| `field(hash=True, compare=False)` | reported as a contradiction |

The three projections nest: export ranges over the `init=True` fields, equality over the `compare=True` fields within them, and the fingerprint over the hashing fields within that — the compared fields with `hash` left at its default or set `True`. So a `compare=False` field round-trips through export and falls outside equality and the fingerprint, and a `hash=False` field takes part in export and equality while the fingerprint ranges over the hashing fields. `init=False` scopes a field to runtime state across all three, so its `compare` and `hash` flags are inert, and its annotation is exempt from the [accepted boundary](types-and-coercion.md#accepted-schema-boundary) — it may hold any resolvable runtime object.

```python
@dataclass
class Model:
    layers: int
    logger: logging.Logger = field(init=False)   # runtime state, any type

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(f"model.{self.layers}")
```

`from_dict(Model, {"layers": 4})` builds the config, runs `__post_init__`, and then checks that every `init=False` field was populated by `__post_init__`, reporting one still awaiting a value as `init=False field was not set during __post_init__`. Supplying an `init=False` field's key in the input is reported as `field is not configurable (init=False)`. On a frozen dataclass, `__post_init__` assigns runtime fields through `object.__setattr__`.

Before coercing any value, `from_dict` runs a recursive schema preflight over every annotation in the tree, including sections built implicitly and fields that will use defaults. An annotation outside the [accepted boundary](types-and-coercion.md#accepted-schema-boundary) on an `init=True` field produces a `ConfigError` at load time, so schema mistakes surface on the very first load.


## Canonical equality

Two configs are `==` exactly when their compared fields serialize to the same canonical plain form, with `NotImplemented` for a different class. Equality compares the fields that are `init=True` and `compare=True` (the defaults); a [`field(compare=False)`](#field-options) still serializes through `to_dict`, and an `init=False` field holds runtime state. The relation works uniformly for every supported field type, [array-valued fields](arrays-and-tensors.md) included, so the round-trip invariant `from_dict(cls, to_dict(config)) == config` reads literally at every level of the tree.

A `ConfigRoot` subclass carries canonical equality from class-creation time; every other schema dataclass receives the same canonical `__eq__` at its first schema processing. confingo owns equality and hashing on config dataclasses, so a hand-written `__eq__` or `__hash__`, or a `@dataclass` flag that conflicts with that ownership (`init=False`, `unsafe_hash=True`, `eq=False`, `order=True`), is rejected; `frozen`, `slots`, and `weakref_slot` are supported. `__hash__` stays object identity, so [`config_hash`](equality-and-hashing.md#stable-run-identity) is the value-identity tool, and the `config_equal` free function exposes the same relation ahead of any engine call.

The full account -- structural array and tensor comparison, the two installation doors, the ownership guard and its rejected-flag behavior, and `config_equal` -- lives in [Equality and hashing](equality-and-hashing.md#canonical-equality).


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
