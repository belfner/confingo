# confingo documentation

confingo turns one dataclass declaration into a schema, a type validator, and a set of defaults: a nested mapping goes in through `from_dict` and comes out as a typed dataclass tree, and `to_dict` takes the tree back to plain serializable data. Validation walks the whole config in one pass and reports every problem with its dotted path.

```python
config = TrainingConfig.cfg.from_file("train.json")   # typed dataclass tree
run_id = config.cfg.hash()                     # stable fingerprint
config.cfg.to_file(Path("runs") / run_id / "resolved.json")
```


## Start here

```bash
pip install confingo
```

[Getting started](getting-started.md) goes from install to a loaded, hashed training config, beginning with a complete runnable example. It is the one page written to be read start to finish, and finishing it is enough to build a real config.


## Two routes

The pages below are one set of content read two ways. **Essentials** is the task route: everything needed to write, load, save, and debug a config, in the order you meet those jobs. **Exact reference** is the contract route: the precise rules, edge cases, and guard behavior, organized for lookup.

Each route stands on its own: Essentials carries every rule its own examples depend on, and each reference page states its contracts in full for a reader who arrived by search.

**Essentials**

- [Getting started](getting-started.md): the linear introduction. A runnable example, then a realistic training config, defaults and sections, reading an error, a resolved snapshot, and a run hash.
- [Arrays and tensors](arrays-and-tensors.md): NumPy and PyTorch fields, choosing a dtype and shape, loading and saving them.
- [Files, formats, and run identity](files-and-identity.md): JSON, YAML, extension dispatch, atomic saves, resolved snapshots, naming a run.
- [Recipes](recipes.md): copyable answers to specific tasks (load and save, choose a surface, compose overrides, diagnose errors).

**Exact reference**

- [Schema design](schema-design.md): dataclass schemas, implicit sections, leaf-level requirements, defaults, factories, `ConfigNode`, variant groups, field options, invariants.
- [Types and coercion](types-and-coercion.md): a "choose an annotation" table, the accepted annotations, and exact conversion rules.
- [Validation and errors](validation-and-errors.md): reading an error, the collect-all model, custom invariants, and every built-in issue source.
- [Equality and hashing](equality-and-hashing.md): canonical equality, unhashable config objects, stable run identity.
- [API reference](api-reference.md): signatures for every public name.


## Best fit

confingo fits programs that:

- treat a typed dataclass tree as the single source of truth for settings,
- want exhaustive, path-tagged error reporting on every load,
- save a resolved snapshot and key each run by a stable fingerprint,
- read JSON and YAML through one schema.

Typical homes are ML training runs, experiment sweeps, and services that load configuration at startup.


## Core guarantees

- One dataclass declaration is the schema, the validator, and the defaults.
- Validation collects every issue in one pass, each with a dotted path.
- Coercion moves values toward annotations under explicit, documented rules; authored defaults are validated against those same annotations and used exactly as written.
- `from_dict(cls, to_dict(config))` restores exact types for concretely annotated fields, across JSON and YAML alike; `ConfigValue` fields round trip through their plain serialized shape. The invariant reads over a value carrying the type its annotation names; see [numeric fields](types-and-coercion.md#unions-and-optionals) for what a hand-assigned `int` in a `float` field does.
- `config_hash` gives equal configs equal fingerprints across processes and hash seeds.


## Exact-reference index

Direct links into the contract route, for a specific question about behavior at an edge.

Schema:

- [Sections build implicitly, hoisting required values to their dotted paths](schema-design.md#implicit-sections-and-leaf-level-requirements)
- [Defaults are validated against their annotation, then used exactly as written](schema-design.md#leaf-defaults-and-precedence)

Types:

- [`ConfigValue` fields keep list shape after a round trip](types-and-coercion.md#open-data)
- [Union members match in declaration order, with a numeric pair settled by the value's own class](types-and-coercion.md#unions-and-optionals)
- [A variant group lets the file name which section to build; a union names at most one section](types-and-coercion.md#variant-groups)
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
- [Config objects are unhashable; `config_hash` carries value identity](equality-and-hashing.md#config-objects-are-unhashable)

Arrays:

- [Bare tensors rebuild with pinned dtypes; broad numpy families select their width by value](arrays-and-tensors.md)
- [An array annotation carries an array; `ConfigValue` carries plain data](types-and-coercion.md#open-data)


## Compatibility

Python 3.12+. The one runtime dependency is PyYAML (`>=6.0`), which carries the YAML loaders; JSON and hashing use only the standard library. NumPy and PyTorch fields activate through presence detection when the application imports those packages itself.
