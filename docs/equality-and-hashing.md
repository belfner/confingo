[Documentation home](README.md)

# Equality and hashing

confingo owns equality and hashing on config dataclasses so that value comparison
and the run fingerprint agree by construction. This page is the authoritative
account of both: canonical equality (what `==` and `config_equal` mean on a schema
class) and stable run identity (what `config_hash` computes and why it is
reproducible). Schema-authoring basics live in [Schema design](schema-design.md);
the everyday persistence workflow lives in [Files, formats, and run identity](files-and-identity.md).


## Canonical equality

Two configs are `==` exactly when their compared fields serialize to the same canonical plain form, with `NotImplemented` for a different class. Equality compares the fields that are `init=True` and `compare=True` (the defaults); a [`field(compare=False)`](schema-design.md#field-options) still serializes through `to_dict`, and an `init=False` field holds runtime state. Canonical equality works uniformly for every supported field type, [array-valued fields](arrays-and-tensors.md) included, so the round-trip invariant `from_dict(cls, to_dict(config)) == config` reads literally at every level of the tree.

The comparison itself runs structurally: array and tensor pairs compare through the backends' vectorized equality wherever that is provably exact (same-kind dtypes, dense forms, elements present), so `==` on large arrays runs at native speed; a tensor meets a numpy array by converting through `detach().cpu().numpy()`; and pairs outside the provably-exact set (mixed integer/float dtypes, zero-size arrays) compare by their canonical JSON form, the same encoding `config_hash` uses, so equality tracks the fingerprint exactly. A cross-kind pair therefore compares equal only when it serializes to the same plain form: an integer and a float array of the same value are distinct (`3` versus `3.0`), while same-kind arrays of any width and float dtypes carrying the same value are equal. Runtime-only tensor state (device placement, `requires_grad`) compares equal, exactly as it serializes equal.

```python
from dataclasses import dataclass

from confingo import ConfigNode


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4


@dataclass(frozen=True)
class RunConfig(ConfigNode):
    optimizer: OptimizerConfig
    seed: int = 0
```

Canonical equality reaches a schema class through two doors:

- A `ConfigNode` subclass carries it from class-creation time: `ConfigNode.__init_subclass__` plants the canonical `__eq__` and a raising `__hash__` ahead of the `@dataclass` decorator, which then keeps them in place of generating its own.
- Every other schema dataclass receives the same canonical `__eq__` at its first schema processing (the first `from_dict` or file load that touches the tree, including its schema preflight), replacing the generated `__eq__` it carried. Ahead of that, a node already compares canonically through `ConfigNode` (recursing into its sections structurally), and `config_equal` covers any tree.

confingo owns equality and hashing on config dataclasses. A class that hand-writes `__eq__` or `__hash__` is rejected, because a hand-written definition would disagree with `config_equal` and `config_hash`. A `ConfigNode` subclass is rejected at class creation, reporting both together when it defines both; a plain dataclass is rejected at its first schema touch. A `ConfigNode` subclass is also rejected when it inherits a hand-written `__eq__` or `__hash__` from a base, since the canonical methods land on the subclass ahead of the decorator and would otherwise resolve in place of the inherited definition; the message names the base that owns it. A plain dataclass that inherits a hand-written definition and generates its own through `@dataclass` keeps the generated one.

The same guard rejects `@dataclass` flags that conflict with that ownership, reported at first schema processing once decoration has run: `init=False` (the class needs its generated `__init__` to build), `unsafe_hash=True` (it installs a field-tuple hash that disagrees with the fingerprint and raises on array fields), `eq=False`, and `order=True` (ordering compares the raw field tuple). A `ConfigNode` subclass declared `unsafe_hash=True` is the one flag caught earlier: it fails at class creation with the standard-library `TypeError` for overwriting `__hash__`, since the node installs its own `__hash__` ahead of the decorator. `frozen=True`, `slots=True`, and `weakref_slot=True` are supported. Provenance is told from a hand-written method by matching its code object against dataclass codegen on the current interpreter; a method fabricated to be byte-identical to that codegen is treated as generated.


## Config objects are unhashable

`hash(config)`, a config used as a dictionary key, and a config placed in a set each raise `TypeError`. [`config_hash`](#stable-run-identity) is the value-identity operation: it is stable across processes, where a Python hash is randomized per process for `str` keys, and it ranges over the same fields as canonical equality.

```python
config = RunConfig.cfg.load_json("config.json")

runs = {config_hash(config): config}          # keyed by value identity
seen = {config_hash(section) for section in sections}
```

From its first schema processing a config class carries `__hash__ = None`, written on the class the engine touched so an untouched base keeps its own hash. That covers a frozen class, where `@dataclass(frozen=True)` otherwise generates a field-tuple hash that disagrees with canonical equality and raises on array fields. A `ConfigNode` subclass holds the same contract from class creation, through a `__hash__` that raises `TypeError: unhashable type: 'RunConfig'; use config_hash(config) for value identity`.

A collection annotation that would need to hash a section is rejected at schema preflight, so `frozenset[OptimizerConfig]` reports at load time rather than failing during construction. Use `list[OptimizerConfig]` or `tuple[OptimizerConfig, ...]` for the collection, and `config_hash(section)` as the key when uniqueness matters.

The same preflight settles every other `set` and `frozenset` element from its annotation, since a load rebuilds each element from the plain form the file carries. An element annotation is admitted when it is a scalar, or a `tuple` / `frozenset` whose own arguments recursively satisfy that same rule, which is what keeps a saved config able to rebuild its own set.

The `config_equal` free function compares two config objects by canonical value equality with the operator's same-class rule, ahead of any engine call and operating on the two instances alone. It evaluates the canonical relation directly, so it always gives the value-comparison answer.


## Field participation

The `compare` and `hash` flags on an `init=True` field scope where the field
lands. The full table lives in [field options](schema-design.md#field-options):

- `field()` (the default): loaded, in `to_dict`, in equality, in `config_hash`.
- `field(compare=False)`: loaded and in `to_dict`; equality and the digest cover the remaining fields.
- `field(hash=False)`: loaded, in `to_dict`, in equality; the digest covers the remaining fields.
- `field(init=False)`: runtime state present in the built object; `to_dict`, equality, and the digest cover the configured fields.


## Stable run identity

`config_hash` fingerprints the resolved config:

```python
run_id = config.cfg.hash()          # "8e6ea26c7116"
long_id = config.cfg.hash(length=32)
```

The hash is the leading `length` hex characters of SHA-256 over the canonical compact JSON (sorted mapping keys, deterministic set ordering) of the config's hashing fields: those that are `init=True`, `compare=True`, and hashed (the defaults). A [`field(compare=False)` or `field(hash=False)`](schema-design.md#field-options) still serializes through `to_dict` while the digest covers the hashing fields, and an `init=False` field holds runtime state. `length` defaults to 12; the useful range is 1-64, and the full digest is 64.

Two properties make it useful as a run identity:

- Equal resolved configs produce equal hashes across processes and `PYTHONHASHSEED` values, because the canonical JSON depends only on the resolved values. A `compare=False` field serializes through `to_dict` while equality and the digest track the same fields, so this holds through it. A `hash=False` field participates in equality while the digest covers the remaining fields, so two configs differing only in a `hash=False` field are unequal yet share a digest.
- A change to a hashing field's canonical encoded value changes that input, so configs differing in an encoded hashing value get distinct digests up to the collision resistance of the chosen prefix length. Longer prefixes buy more resistance; because the digest covers the hashing fields, two configs that differ only in a `hash=False` field can share one.

[Array and tensor fields](arrays-and-tensors.md#round-trips-equality-and-hashing) hash by their encoded values and nesting, exactly what the file records. Arrays that share those encoded values collide by design: bare-annotated arrays holding the same values at different dtype widths hash equal, integer and float widths alike, as do tensors that share values while differing in device, gradient state, or stride and storage arrangement, and zero-size arrays that share their retained encoded dimensions. A concrete dtype annotation is schema, so it shapes the rebuilt value while the hash tracks the encoded values, the same way `tuple`-ness already works.

### Subtree digests

`config_hash` covers exactly the object it is called on, so a nested [config node](schema-design.md#config-nodes) fingerprints its own section:

```python
run_id = config.cfg.hash()              # covers the whole config
optimizer_id = config.optimizer.cfg.hash()   # covers the optimizer section
```

Three consequences follow from the digest being a content fingerprint:

- Each dataclass level applies its own field projection, so an enclosing field declared `hash=False` keeps that whole section out of the enclosing digest while the section's own `section.cfg.hash()` still tracks its fields. Two configs differing only inside such a section share an enclosing digest and carry distinct section digests.
- The canonical JSON records values and structure, so two classes whose hashing fields encode identically produce the same digest. A section digest identifies content rather than the class that held it.
- Prefix collision resistance is governed by `length` at every level, so a short section digest carries the same trade-off a short run id does.

That makes the hash a natural run-directory name for sweeps:

```python
for overrides in sweep:
    config = ExperimentConfig.cfg.from_dict({**base, **overrides})
    run_dir = Path("runs") / config.cfg.hash()
    run_dir.mkdir(parents=True, exist_ok=True)
    config.cfg.save_json(run_dir / "resolved.json")
```


---

Exact reference: [Schema design](schema-design.md) | [Types and coercion](types-and-coercion.md) | [Validation and errors](validation-and-errors.md) | [API reference](api-reference.md) | [Documentation home](README.md)
