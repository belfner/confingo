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

from confingo import ConfigRoot


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4


@dataclass(frozen=True)
class RunConfig(ConfigRoot):
    optimizer: OptimizerConfig
    seed: int = 0
```

Canonical equality reaches a schema class through two doors:

- A `ConfigRoot` subclass carries it from class-creation time: `ConfigRoot.__init_subclass__` plants the canonical `__eq__` and identity `__hash__` ahead of the `@dataclass` decorator, which then keeps them in place of generating its own.
- Every other schema dataclass receives the same canonical `__eq__` at its first schema processing -- the first `from_dict` or file load that touches the tree, including its schema preflight -- replacing the generated `__eq__` it carried, with identity hashing restored where generating `__eq__` had disabled it. Ahead of that, a root already compares canonically through `ConfigRoot` (recursing into its sections structurally), and `config_equal` covers any tree.

confingo owns equality and hashing on config dataclasses. A class that hand-writes `__eq__` or `__hash__` is rejected -- a root at class creation (both reported together when it defines both), a section at its first schema touch -- because a hand-written definition would disagree with `config_equal` and `config_hash`. The same guard rejects `@dataclass` flags that conflict with that ownership, reported at first schema processing once decoration has run: `init=False` (the class needs its generated `__init__` to build), `unsafe_hash=True` (it installs a field-tuple hash that disagrees with the fingerprint and raises on array fields), `eq=False`, and `order=True` (ordering compares the raw field tuple). A `ConfigRoot` subclass declared `unsafe_hash=True` is the one flag caught earlier: it fails at class creation with the standard-library `TypeError` for overwriting `__hash__`, since the root installs identity hashing ahead of the decorator. `frozen=True`, `slots=True`, and `weakref_slot=True` are supported; the generated hash of a frozen class, or one inherited by an undecorated dataclass subclass, is reduced to identity so it shares the same model as every other config. Provenance is told from a hand-written method by matching its code object against dataclass codegen on the current interpreter; a method fabricated to be byte-identical to that codegen is treated as generated.

With the canonical `__eq__` installed, `__hash__` stays object identity, so two equal configs are still distinct set members; [`config_hash`](#stable-run-identity) is the value-identity tool.

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
run_id = config.config_hash()          # "8e6ea26c7116"
long_id = config.config_hash(length=32)
```

The hash is the leading `length` hex characters of SHA-256 over the canonical compact JSON (sorted mapping keys, deterministic set ordering) of the config's hashing fields — those that are `init=True`, `compare=True`, and hashed (the defaults). A [`field(compare=False)` or `field(hash=False)`](schema-design.md#field-options) still serializes through `to_dict` while the digest covers the hashing fields, and an `init=False` field holds runtime state. `length` defaults to 12; the useful range is 1-64, and the full digest is 64.

Two properties make it useful as a run identity:

- Equal resolved configs produce equal hashes across processes and `PYTHONHASHSEED` values, because the canonical JSON depends only on the resolved values. A `compare=False` field serializes through `to_dict` while equality and the digest track the same fields, so this holds through it. A `hash=False` field participates in equality while the digest covers the remaining fields, so two configs differing only in a `hash=False` field are unequal yet share a digest.
- A change to a hashing field's canonical encoded value changes that input, so configs differing in an encoded hashing value get distinct digests up to the collision resistance of the chosen prefix length. Longer prefixes buy more resistance; because the digest covers the hashing fields, two configs that differ only in a `hash=False` field can share one.

[Array and tensor fields](arrays-and-tensors.md#round-trips-equality-and-hashing) hash by their encoded values and nesting, exactly what the file records. Arrays that share those encoded values collide by design: bare-annotated arrays holding the same values at different dtype widths hash equal, integer and float widths alike, as do tensors that share values while differing in device, gradient state, or stride and storage arrangement, and zero-size arrays that share their retained encoded dimensions. A concrete dtype annotation is schema, so it shapes the rebuilt value while the hash tracks the encoded values, the same way `tuple`-ness already works.

That makes the hash a natural run-directory name for sweeps:

```python
for overrides in sweep:
    config = ExperimentConfig.from_dict({**base, **overrides})
    run_dir = Path("runs") / config.config_hash()
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save_json(run_dir / "resolved.json")
```


---

[Schema design](schema-design.md) | [Home](README.md) | [Files, formats, and run identity](files-and-identity.md)
