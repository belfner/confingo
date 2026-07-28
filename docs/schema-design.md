[Documentation home](README.md)

# Schema design

This page covers structuring a program's configuration as a dataclass tree: implicit sections, leaf-level requirements, defaults, authored factories, `ConfigNode`, field options, validation hooks, and a summary of canonical equality.


## A dataclass tree is the schema

One dataclass declaration serves three roles: the field names define the accepted keys, the annotations define the accepted types, and the defaults define the fallback values. Nested dataclasses define sections, and containers of dataclasses (`list[StageConfig]`, `dict[str, DatasetConfig]`) define repeated sections.

Schema classes are ordinary `@dataclass` declarations. Any of them may subclass `ConfigNode` for load/save/hash methods and config-aware equality, summarized in [canonical equality](#canonical-equality) below; the rest are plain dataclasses the `confingo.functional` free functions cover.


## What a schema class may be

A schema class is a record and nothing else. Two rules say what that means, and both are reported at preflight with the shape named.

**A section is not also one of the kinds a walk dispatches on.** Every walk over a value asks what the value is, and it recognizes a section by its being a dataclass and a container, enum, path, or temporal value by its being an instance of that kind. A class that is both answers to two readings of one value, and which reading applies would follow from the order a particular walk happens to test in. So a schema class may not subclass `Mapping`, `Sequence`, `set`, `frozenset`, `Enum`, `Path`, `date`, or `time`. Carry that behavior on an object the section holds in an [`init=False`](#field-options) field.

**The constructor takes the class's own fields.** confingo builds a config object by calling the class with its `init=True` field names, so `__init__` has to accept exactly that call: every one of those names, and nothing further that it requires. The generated `__init__` satisfies this by construction, which is what makes the rule invisible for an ordinary declaration. A class that binds an `__init__` of its own is reported by the field its constructor cannot receive, and a required `InitVar` is reported as an argument no config file can supply, since `dataclasses.fields` leaves an `InitVar` outside the loadable surface while the constructor still asks for it. An `InitVar` carrying a default is answered by that default and builds.

Reading the signature is what settles the second rule, so it rests on what the constructor accepts rather than on telling a generated body from an authored one.


## Implicit sections and leaf-level requirements

Sections declare themselves. A dataclass-typed field with a bare annotation defaults to an automatic build, recursively through sub-sections of sub-sections.

That makes a section's required-ness a property of its leaves: every required leaf must come from the file, at its nested position, wherever it sits in the tree. A required leaf reached through an implicit build is reported at its dotted path, so the schema author thinks only about which values a run needs, and an error names exactly the value to add.

The implicit build applies to direct dataclass annotations only. Every other field type is required, and a default makes it optional:

- scalars (`int`, `str`, `Path`, ...)
- unions, including `Section | None`
- `ConfigValue` and `ConfigScalar`
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

from confingo import ConfigNode


@dataclass
class WarmupConfig:
    steps: int
    start_factor: float = 0.1


@dataclass
class ScheduleConfig:
    warmup: WarmupConfig
    decay: str = "cosine"


@dataclass
class RunConfig(ConfigNode):
    schedule: ScheduleConfig
    stages: list[str]
    checkpoints: list[Path] = field(default_factory=list)
    seed: int = 0
```

Loading `{}` against this schema reports two issues. `schedule.warmup.steps: missing required value` is the requirement hoisted up through two implicit sections, and `stages: missing required value` is the required container.

A file supplying `{"schedule": {"warmup": {"steps": 500}}, "stages": ["warmup", "main"]}` builds the whole tree: `decay` and `start_factor` from their defaults, `checkpoints` empty by authored intent.


## Leaf defaults and precedence

Defaults form the base layer; the input mapping overrides whichever leaves it supplies.

The two layers are treated differently. Supplied values travel through [coercion](types-and-coercion.md), while defaults are validated and then used exactly as authored: a defaulted field receives the object you wrote in the declaration, byte for byte. Write defaults that already have the annotated type (`Path("runs")` for a `Path` field, `0.1` for a `float` field), so both layers produce the same shapes.

Validation is what holds that rule up. A default has to arrive in the form coercion would otherwise have produced, and it has to have a plain serializable form, so a config whose values all come from defaults writes back out and reloads unchanged. A `__post_init__` that replaces an `init=True` value afterwards puts its own result into the snapshot, on that value's own terms. `output_dir: Path = "runs"` is reported as an authoring error rather than promoted to `Path("runs")`, and so are a list default for a tuple field, an integral float for an `int` field, and a mapping for a section. A direct `field(default=...)` is checked during schema preflight, whether or not the input supplies the field, so a wrong default surfaces even in a project that always overrides it:

```text
output_dir: invalid authored default: expected a value already matching Path, got str; defaults are validated as written
```


## Authored factories and partial files

An explicit `field(default_factory=...)` takes precedence over the implicit build and is used as authored, which makes it the tool for baseline sections whose fallback differs from the section's own defaults. The factory runs once, at the one build that selects it, and its product goes through the same validation a direct default does before it reaches the object:

```python
from dataclasses import (
    dataclass,
    field,
)

from confingo import ConfigNode


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01


@dataclass
class ExperimentConfig(ConfigNode):
    seed: int = 0
    batch_size: int = 64
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=1e-3))
```

A section an authored default supplies reaches the same lifecycle as one built from a file: the build that selects the default checks its `init=False` fields for completeness and reports what its `__validate__` returns, at that section's own path. Its `__post_init__` has already run inside the factory. So a baseline section is held to its own invariants, and the report is the same whether the file named the section or left it to the default.

A section-valued default is always written as a factory, frozen sections included. `@dataclass` reads a default whose class is unhashable as a mutable default and directs you to `default_factory`, and confingo [makes a config class unhashable](equality-and-hashing.md#config-objects-are-unhashable) at its first schema processing, so a direct `section: FrozenSection = FrozenSection()` in a class declared after that first load fails at decoration time. The factory form is correct regardless of declaration order.

Partial files fall out of the leaf-level model everywhere: a minimal experiment file overrides two leaves and takes everything else from defaults, implicit or authored:

```json
{"seed": 7, "batch_size": 256}
```

An empty mapping (`{}`) builds the full default config whenever every leaf in the tree has a default, which also makes JSON `null` and empty YAML documents load as pure defaults ([document rules](files-and-identity.md#document-and-read-rules)).


## Config nodes

Any dataclass in the tree may subclass `ConfigNode`, at any depth. A section that subclasses it gains the same methods over its own subtree; a section that stays a plain dataclass is walked by introspection, and the `confingo.functional` free functions operate on it.

The base class is a thin facade: each method delegates to the matching free function (`TrainingConfig.cfg.load_json(path)` calls `load_json(TrainingConfig, path)`), so both styles are equivalent public surfaces. The full mapping is in the [API reference](api-reference.md#confignode-method-map).

Each method is scoped to the node it is called on. `optimizer.cfg.to_dict()` renders that section, `optimizer.cfg.hash()` fingerprints that section, `optimizer.cfg.save_json(path)` writes a document that loads back through `OptimizerConfig`, and `OptimizerConfig.cfg.from_dict(...)` reports issue paths relative to that section, so a leaf reported as `optimizer.lr` through the enclosing config is reported as `lr` here.

Attaching `ConfigNode` to a section changes its method surface and nothing else. The engine reaches a nested section through the same generic recursion either way, so the built values, exported data, fingerprints, and issue paths of an enclosing load are identical whether or not the section subclasses it.

Subclassing reserves one name, `cfg`, which carries every operation. A node declares nothing under it: an annotation or class-body binding named `cfg` is rejected at class creation, as is one supplied by a base ahead of `ConfigNode` in the MRO, inherited as a field, or supplied as a metaclass data descriptor. What a class inherits is judged by the fields that actually land on it, so an inherited `ClassVar` named `cfg` stores nothing and is accepted. `cfg` is free on a plain dataclass, which shadows nothing, and every other name stays free on a node.

Each subclass that declares its own fields carries the `@dataclass` decorator. A subclass that declares annotations without it inherits the base's fields alone, so confingo reports it as a schema error naming the class.


## Runtime-resolvable annotations

Define schema classes at module scope. Annotations are resolved at load time from the defining module's namespace, so module-level classes make forward references (including `from __future__ import annotations` strings) resolvable. `Annotated[T, ...]` fields behave as their base type `T`.

A name that fails to resolve is reported as a schema error with the `config schema` context. See [validation phases](validation-and-errors.md#two-validation-phases).


## Field options

`init` is the master switch. An `init=True` field (the dataclass default) is loaded from the config, exported by `to_dict`, and, subject to `compare` and `hash` below, weighed by equality and `config_hash`. An `init=False` field is runtime state that its default or `__post_init__` populates; loading, export, equality, and the fingerprint all draw from the `init=True` fields. On an `init=True` field, `compare` and `hash` scope equality and the fingerprint:

| Field | Loaded | In `to_dict` | In equality | In `config_hash` |
| --- | :---: | :---: | :---: | :---: |
| `field()` (init=True default) | yes | yes | yes | yes |
| `field(init=False)` | no | no | no | no |
| `field(compare=False)` | yes | yes | no | no |
| `field(hash=False)` | yes | yes | yes | no |
| `field(hash=True, compare=False)` | reported as a contradiction |

The three projections nest: export ranges over the `init=True` fields, equality over the `compare=True` fields within them, and the fingerprint over the hashing fields within that, meaning the compared fields with `hash` left at its default or set `True`. So a `compare=False` field round-trips through export and falls outside equality and the fingerprint, and a `hash=False` field takes part in export and equality while the fingerprint ranges over the hashing fields. `init=False` scopes a field to runtime state across all three, so its `compare` and `hash` flags are inert, and its annotation is exempt from the [accepted boundary](types-and-coercion.md#accepted-schema-boundary), so it may hold any resolvable runtime object.

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

Two configs are `==` exactly when their compared fields serialize to the same canonical plain form, with `NotImplemented` for a different class. Equality compares the fields that are `init=True` and `compare=True` (the defaults); a [`field(compare=False)`](#field-options) still serializes through `to_dict`, and an `init=False` field holds runtime state. The relation works uniformly for every supported field type, [array-valued fields](arrays-and-tensors.md) included, so the round-trip invariant `from_dict(cls, to_dict(config)) == config` reads literally at every level of the tree. The invariant reads over a value that already carries the type its annotation names, which is what a load builds and what a validated authored default carries. A value assigned by hand renders as the object it is: Python's numeric tower lets a type checker accept `x: float = 1`, and the `int` that field then holds writes `1` and reloads as `1.0`, which the fingerprint tokenizes differently. A subclass instance behaves the same way, writing the plain form of the base class the annotation names and reloading as that base. Write the value the annotation names, and derive a subclass in an `init=False` field.

A `ConfigNode` subclass carries canonical equality from class-creation time; every other schema dataclass receives the same canonical `__eq__` at its first schema processing. confingo owns equality and hashing on config dataclasses, so a hand-written `__eq__` or `__hash__`, or a `@dataclass` flag that conflicts with that ownership (`init=False`, `unsafe_hash=True`, `eq=False`, `order=True`), is rejected; `frozen`, `slots`, and `weakref_slot` are supported. Config objects are [unhashable](equality-and-hashing.md#config-objects-are-unhashable), so [`config_hash`](equality-and-hashing.md#stable-run-identity) is the value-identity tool, and the `config_equal` free function exposes the same relation ahead of any engine call.

The full account lives in [Equality and hashing](equality-and-hashing.md#canonical-equality): structural array and tensor comparison, the two installation doors, the ownership guard and its rejected-flag behavior, and `config_equal`.


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

Exact reference: [Types and coercion](types-and-coercion.md) | [Validation and errors](validation-and-errors.md) | [Equality and hashing](equality-and-hashing.md) | [API reference](api-reference.md) | [Documentation home](README.md)
