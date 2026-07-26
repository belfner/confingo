[Documentation home](README.md)

# API reference

Compact reference for the public surface. Signatures come first and are exact; the class contracts confingo installs on a schema class close the page. Behavior details live on the linked guide pages.


## Import surface

```python
from confingo import (
    ConfigError,
    ConfigIssue,
    ConfigNode,
    config_equal,
    config_hash,
    dumps_json,
    from_dict,
    from_file,
    load_json,
    save_json,
    to_dict,
    to_file,
)
from confingo import dumps_yaml, load_yaml, save_yaml
```

`confingo.__version__` carries the package version.


## Construction and conversion

Learn more: [Schema design](schema-design.md), [Types and coercion](types-and-coercion.md).

### `from_dict(config_cls, data, *, context="config") -> T`

Builds `config_cls` (a dataclass type) from a mapping, coercing each value toward its annotation and collecting every issue before raising `ConfigError`. `context` names the source in error messages.

### `to_dict(value) -> Any`

Converts a config object to plain serializable data in field-declaration order. Raises `ConfigError` for values outside the plain-data model.

### `config_hash(config, *, length=12) -> str`

SHA-256 fingerprint of the config's canonical JSON over its hashing fields (`init=True`, `compare=True`, effective hash enabled), so a `compare=False` or `hash=False` field is carried by `to_dict` while the digest covers the hashing fields: the digest's leading `length` hex characters (useful range 1-64; the full digest is 64). Stable across processes and hash seeds. Learn more: [stable run identity](files-and-identity.md#stable-run-identity).


## JSON functions

Learn more: [Files, formats, and run identity](files-and-identity.md#json).

| Function | Behavior |
| --- | --- |
| `load_json(config_cls, path) -> T` | Reads UTF-8 JSON, builds the config; read, parse, preflight, and construction failures raise `ConfigError` with `config file <path>` context ([annotation-resolution failures use `config schema`](validation-and-errors.md#two-validation-phases)) |
| `dumps_json(config, *, indent=2) -> str` | JSON text, ASCII-escaped, trailing newline |
| `save_json(config, path, *, indent=2) -> Path` | Atomic write; returns the destination path |


## YAML functions

Built into the base install. Learn more: [YAML](files-and-identity.md#yaml).

| Function | Behavior |
| --- | --- |
| `load_yaml(config_cls, path) -> T` | `safe_load`, then builds the config; failures raise `ConfigError` |
| `dumps_yaml(config, *, indent=2, sort_keys=False) -> str` | `safe_dump` text; declaration order by default |
| `save_yaml(config, path, *, indent=2, sort_keys=False) -> Path` | Atomic write; returns the destination path |


## Format-dispatch functions

Route by suffix: `.json`, `.yaml`, `.yml`, case-insensitive. A missing or unrecognized suffix raises `ConfigError`. Learn more: [Extension dispatch](files-and-identity.md#extension-dispatch).

| Function | Behavior |
| --- | --- |
| `from_file(config_cls, path) -> T` | Loads via the format matching the suffix |
| `to_file(config, path, *, indent=2) -> Path` | Saves via the format matching the suffix |


## Error types

Learn more: [Validation and errors](validation-and-errors.md).

### `ConfigIssue(path, message)`

Frozen dataclass; one problem at one dotted path. `str(issue)` renders `path: message`, with an empty path shown as `<root>`.

### `ConfigError(issues, *, context)`

`ValueError` subclass carrying `.issues` (tuple, discovery order) and `.context` (str). The rendered message summarizes the count and lists every issue. `ConfigError.single(message, *, context, path="")` builds a one-issue error.


## `ConfigNode` method map

`ConfigNode` is a mixin any config dataclass may carry, at any depth in the tree; each method delegates to its free-function twin, and subclassing installs [canonical equality](schema-design.md#canonical-equality) at class-creation time. Subclasses still carry the `@dataclass` decorator.

The receiver defines the operation scope: a method called on a nested node builds, exports, fingerprints, or writes that node's own subtree, and issue paths are relative to it.

Subclassing reserves the eleven method names below. A node declares none of them: any annotation or class-body binding under one of these names is rejected at class creation with a `config schema` error, as is one supplied by a base ahead of `ConfigNode` in the MRO, inherited as a field, or supplied as a metaclass data descriptor. The same names carry no restriction on a plain dataclass.

| Method | Free function |
| --- | --- |
| `Config.from_dict(data, *, context="config")` | `from_dict(Config, data, *, context="config")` |
| `Config.load_json(path)` | `load_json(Config, path)` |
| `Config.load_yaml(path)` | `load_yaml(Config, path)` |
| `Config.from_file(path)` | `from_file(Config, path)` |
| `config.to_dict()` | `to_dict(config)` |
| `config.dumps_json(*, indent=2)` | `dumps_json(config, *, indent=2)` |
| `config.save_json(path, *, indent=2)` | `save_json(config, path, *, indent=2)` |
| `config.dumps_yaml(*, indent=2, sort_keys=False)` | `dumps_yaml(config, *, indent=2, sort_keys=False)` |
| `config.save_yaml(path, *, indent=2, sort_keys=False)` | `save_yaml(config, path, *, indent=2, sort_keys=False)` |
| `config.to_file(path, *, indent=2)` | `to_file(config, path, *, indent=2)` |
| `config.config_hash(*, length=12)` | `config_hash(config, *, length=12)` |


## Choosing a surface

The method style suits a root class that owns its schema (`TrainingConfig.load_json(path)` reads naturally at call sites). The free functions suit plain dataclass roots and library code that receives the config class as a parameter. Both surfaces are equivalent and stay in sync by construction.


## Class contracts

What confingo installs on a schema class, and what it rejects. Learn more: [canonical equality](schema-design.md#canonical-equality).

Schema classes are ordinary `@dataclass` declarations. Every schema class carries canonical equality: two configs are `==` exactly when their compared fields (`init=True` and `compare=True`) serialize to the same plain form (`NotImplemented` for a different class), with array and tensor fields compared through the backends' vectorized operations. A `ConfigNode` subclass carries canonical equality from class-creation time; every other schema dataclass receives it at first schema processing, replacing the generated `__eq__` it carried. Config objects are unhashable and `config_hash` carries value identity: a class carries `__hash__ = None` from its first schema processing, and a `ConfigNode` subclass holds the same contract from class creation through a `__hash__` that raises `TypeError` naming `config_hash`. confingo owns equality and hashing. A class that hand-writes `__eq__` or `__hash__` is rejected: a `ConfigNode` subclass at class creation, including one that inherits a hand-written definition from a base, and a plain dataclass at first schema touch. A conflicting `@dataclass` flag (`init=False`, `unsafe_hash=True`, `eq=False`, `order=True`) raises a `ConfigError` at first schema processing, with the one exception that a `ConfigNode` subclass declared `unsafe_hash=True` fails at class creation with the standard-library `TypeError` for overwriting `__hash__`. `frozen`, `slots`, and `weakref_slot` are supported.

### `config_equal(left, right) -> bool`

Compares two config objects by canonical value equality over their compared fields (`init=True` and `compare=True`), same-class rule included, ahead of any engine call and operating on the two instances alone. Evaluates the canonical relation directly. Requires `left` to be a dataclass instance, and raises `TypeError` for any other value.


---

Exact reference: [Schema design](schema-design.md) | [Types and coercion](types-and-coercion.md) | [Validation and errors](validation-and-errors.md) | [Equality and hashing](equality-and-hashing.md) | [Documentation home](README.md)
