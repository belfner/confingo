# confingo

A dataclass-driven configuration toolkit. Define your program's settings once as
typed dataclasses, then load them from a config file, with everything validated
and coerced against the schema on the way in.

The dataclass declaration is the single source of truth: it serves at once as the
schema, the type validator, and the default values.

## Installation

```bash
pip install confingo
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add confingo
```

confingo runs on the Python standard library alone and needs only Python 3.10 or
newer.

## Core operations

confingo is built around a marshal / unmarshal pair that treats the dataclass
tree as the spec:

- **unmarshal** (`from_dict`) turns a nested mapping (parsed from a config file)
  into a fully built dataclass tree, coercing each value toward its annotated
  type and reporting every problem it finds.
- **marshal** (`to_dict`) turns a config object back into plain, serializable
  Python data, in field-declaration order.

The two round-trip: `from_dict(cls, to_dict(config)) == config` holds for every
field whose annotation names a type.

File loading layers a config file on top of the dataclass defaults:

1. **Dataclass field defaults** - what is baked into the code.
2. **A config file** - values that override individual leaves.

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from confingo import load_json, save_json


@dataclass(frozen=True)
class TrainerConfig:
    lr: float = 3e-4
    algorithm: Literal["sac", "td3"] = "td3"


@dataclass(frozen=True)
class TrainingConfig:
    seed: int | None = None
    output_dir: Path = Path("runs")
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


config = load_json(TrainingConfig, "config.json")
save_json(config, config.output_dir / "config.json")
```

The same operations are available as methods when the root dataclass subclasses
``ConfigRoot``, so the class carries its own schema. The recommended shape makes
sub-config sections required, so a config is fully specified by its file, while
leaves keep sensible defaults:

```python
from dataclasses import dataclass
from pathlib import Path

from confingo import ConfigRoot


@dataclass
class TrainingConfig(ConfigRoot):
    trainer: TrainerConfig  # required section, supplied by the file
    seed: int | None = None
    output_dir: Path = Path("runs")


config = TrainingConfig.load_json("config.json")
config.save_json(config.output_dir / "config.json")
```

Only the root subclasses ``ConfigRoot``; nested sections stay plain dataclasses.
Giving a section a `field(default_factory=...)` default instead (as in the first
example) is fully supported and lets a partial file omit that whole section.

## What makes it more than "just load a file"

The value is in four cross-cutting guarantees:

- **Collect-all validation.** Building a config walks the whole dataclass tree
  before it raises, so one run reports every problem at once: unknown keys,
  missing required values, type mismatches, and failures raised from
  `__post_init__` or a custom `__validate__` method. Each issue is tagged with a
  dotted path such as `training.trainer.lr`.

- **Type coercion toward the annotation.** Config data is loosely typed, so
  values are nudged into shape: an integral float lands on an `int` field,
  strings resolve to `Enum` members and `Path` objects, and sequences become the
  exact container the annotation asks for (`tuple`, `set`, and friends). It stays
  strict where it matters, so a `bool` stays off an `int` field.

- **Save reflects the in-memory object.** Saving serializes the resolved config
  object as the caller holds it, so any programmatic change made after loading is
  captured in the written file.

- **Stable identity.** A config hash fingerprints the resolved config over its
  canonical JSON form, so equal configs hash equal across processes and key
  orderings. The digest is usable for run naming, deduplication, and confirming
  that a rerun used the same settings.

## Scope

- The core toolkit depends only on the Python standard library.
- **JSON** is the config file format supported first.
- **YAML** support arrives later as an optional extra, pulling in PyYAML only
  when installed.
- Targets Python 3.10 and newer.

## In one line

confingo packages the "config file plus dataclass schema" pattern into a
reusable toolkit: a typed marshal / unmarshal pair over plain stdlib dataclasses,
with exhaustive error reporting and a reproducible fingerprint.
