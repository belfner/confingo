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

New to the library? [Getting started](getting-started.md) goes from install to a loaded, hashed training config in about five minutes. YAML setup lives in [Files, formats, and run identity](files-and-identity.md#yaml-extra).


## Choose a topic

- [Getting started](getting-started.md): first schema, first load, first run hash.
- [Schema design](schema-design.md): implicit sections, leaf-level requirements, defaults, `ConfigRoot`.
- [Types and coercion](types-and-coercion.md): accepted annotations and exact conversion rules.
- [Validation and errors](validation-and-errors.md): the collect-all error model and custom invariants.
- [Files, formats, and run identity](files-and-identity.md): JSON, YAML, extension dispatch, atomic saves, `config_hash`.
- [API reference](api-reference.md): signatures for every public name.


## Find an edge case

- [Absent sections build implicitly with required values hoisted](schema-design.md#implicit-sections-and-leaf-level-requirements)
- [Tuples and sets serialize as lists; annotations restore them](files-and-identity.md#cross-format-round-trip)
- [`Any` fields keep list shape after a round trip](types-and-coercion.md#any-and-plain-data)
- [Union members match in declaration order](types-and-coercion.md#unions-and-optionals)
- [`bool` and `int` stay separate; whole floats coerce to int](types-and-coercion.md#scalars)
- [Defaults are trusted as authored and skip coercion](schema-design.md#leaf-defaults-and-precedence)
- [`datetime` values are rejected on `date` fields](types-and-coercion.md#finite-numbers-and-temporal-exactness)
- [`Literal` matches value and exact type together](types-and-coercion.md#enums-and-literals)
- [Schema preflight rejects out-of-boundary annotations on unused fields](types-and-coercion.md#accepted-schema-boundary)
- [Null documents load as defaults; empty JSON text is malformed](files-and-identity.md#document-and-read-rules)
- [Saves are atomic and preserve file modes](files-and-identity.md#atomic-writes)
- [`config_hash` is stable across processes and hash seeds](files-and-identity.md#stable-run-identity)


## Core guarantees

- One dataclass declaration is the schema, the validator, and the defaults.
- Validation collects every issue in one pass, each with a dotted path.
- Coercion moves values toward annotations under explicit, documented rules.
- `from_dict(cls, to_dict(config))` restores exact types for concretely annotated fields, across JSON and YAML alike; `Any` fields round trip through their plain serialized shape.
- `config_hash` gives equal configs equal fingerprints across processes and hash seeds.


## Compatibility

Python 3.11+. The core imports only the standard library; YAML support arrives with `pip install "confingo[yaml]"`.
