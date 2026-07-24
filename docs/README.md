# confingo documentation

confingo turns one dataclass declaration into a schema, a type validator, and a set of defaults: a nested mapping goes in through `from_dict` and comes out as a typed dataclass tree, and `to_dict` takes the tree back to plain serializable data. Validation walks the whole config in one pass and reports every problem with its dotted path.

```python
config = TrainingConfig.from_file("train.json")   # typed dataclass tree
run_id = config.config_hash()                     # stable fingerprint
config.to_file(Path("runs") / run_id / "resolved.json")
```


## Start here

```bash
pip install confingo
```

New to the library? [Getting started](getting-started.md) goes from install to a loaded, hashed training config in about five minutes, beginning with a complete runnable example. For copyable answers to specific tasks, jump to [Recipes](recipes.md).


## Reader paths

**First run**

- [Getting started](getting-started.md): a runnable minimal example, then a realistic training config, a resolved snapshot, and a run hash.

**Common tasks**

- [Recipes](recipes.md): load and save, choose a surface, name a run by its hash, compose overrides, and diagnose errors.

**Concepts and guarantees**

- [Schema design](schema-design.md): dataclass schemas, implicit sections, leaf-level requirements, defaults, factories, `ConfigRoot`, field options, invariants.
- [Types and coercion](types-and-coercion.md): a "choose an annotation" table, accepted annotations, and exact conversion rules.
- [Validation and errors](validation-and-errors.md): reading an error, the collect-all model, and custom invariants.
- [Files, formats, and run identity](files-and-identity.md): JSON, YAML, extension dispatch, atomic saves, resolved snapshots.
- [Equality and hashing](equality-and-hashing.md): canonical equality and stable run identity.
- [Arrays and tensors](arrays-and-tensors.md): NumPy and PyTorch fields, dtype and shape rules.

**API reference**

- [API reference](api-reference.md): signatures for every public name.


## Best fit

confingo fits programs that:

- treat a typed dataclass tree as the single source of truth for settings,
- want exhaustive, path-tagged error reporting on every load,
- save a resolved snapshot and key each run by a stable fingerprint,
- read JSON and YAML through one schema.

Typical homes are ML training runs, experiment sweeps, and services that load configuration at startup.


## Find an edge case

Schema:

- [Sections build implicitly, hoisting required values to their dotted paths](schema-design.md#implicit-sections-and-leaf-level-requirements)
- [Defaults are trusted as authored and used exactly as written](schema-design.md#leaf-defaults-and-precedence)

Types:

- [`Any` fields keep list shape after a round trip](types-and-coercion.md#any-and-plain-data)
- [Union members match in declaration order](types-and-coercion.md#unions-and-optionals)
- [`bool` and `int` stay separate; whole floats coerce to int](types-and-coercion.md#scalars)
- [Temporal fields keep `datetime` and `date` distinct](types-and-coercion.md#finite-numbers-and-temporal-exactness)
- [`Literal` matches value and exact type together](types-and-coercion.md#enums-and-literals)
- [Schema preflight checks every annotation, defaulted fields included](types-and-coercion.md#accepted-schema-boundary)

Validation:

- [Every problem is reported in one pass, each tagged with a dotted path](validation-and-errors.md#one-exception-every-discovered-issue)
- [Read an error in three steps: context, path, message](validation-and-errors.md#read-an-error-in-three-steps)

Files:

- [Tuples and sets serialize as lists; annotations restore them](files-and-identity.md#cross-format-round-trip)
- [Null documents load as defaults; empty JSON text is malformed](files-and-identity.md#document-and-read-rules)
- [Saves are atomic and preserve file modes](files-and-identity.md#atomic-writes)

Identity:

- [`config_hash` is stable across processes and hash seeds](equality-and-hashing.md#stable-run-identity)

Arrays:

- [Bare tensors rebuild with pinned dtypes; broad numpy families select their width by value](arrays-and-tensors.md)
- [Arrays under `Any` serialize as plain scalars and lists](types-and-coercion.md#any-and-plain-data)


## Core guarantees

- One dataclass declaration is the schema, the validator, and the defaults.
- Validation collects every issue in one pass, each with a dotted path.
- Coercion moves values toward annotations under explicit, documented rules.
- `from_dict(cls, to_dict(config))` restores exact types for concretely annotated fields, across JSON and YAML alike; `Any` fields round trip through their plain serialized shape.
- `config_hash` gives equal configs equal fingerprints across processes and hash seeds.


## Compatibility

Python 3.11+. The one runtime dependency is PyYAML (`>=6.0`), which carries the YAML loaders; JSON and hashing use only the standard library. NumPy and PyTorch fields activate through presence detection when the application imports those packages itself.
