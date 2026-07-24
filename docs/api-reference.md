[Documentation home](README.md)

# API reference

Compact reference for the public surface. Signatures are exact; behavior details live on the linked guide pages.


## Import surface

```python
from confingo import (
    ConfigError,
    ConfigIssue,
    ConfigRoot,
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


## Schema declaration and equality

Learn more: [canonical equality](schema-design.md#canonical-equality).

Schema classes are ordinary `@dataclass` declarations. Every schema class carries canonical equality: two configs are `==` exactly when their compared fields (`init=True` and `compare=True`) serialize to the same plain form (`NotImplemented` for a different class), with array and tensor fields compared through the backends' vectorized operations, and `__hash__` kept as object identity. A `ConfigRoot` subclass carries this from class-creation time; every other schema dataclass receives it at first schema processing, replacing the generated `__eq__` it carried. confingo owns equality and hashing: a class that hand-writes `__eq__` or `__hash__` is rejected -- a root at class creation, a section at first schema touch -- and a conflicting `@dataclass` flag (`init=False`, `unsafe_hash=True`, `eq=False`, `order=True`) raises a `ConfigError` at first schema processing, with the one exception that a `ConfigRoot` subclass declared `unsafe_hash=True` fails at class creation with the standard-library `TypeError` for overwriting `__hash__`. `frozen`, `slots`, and `weakref_slot` are supported.

### `config_equal(left, right) -> bool`

Compares two config objects by canonical value equality over their compared fields (`init=True` and `compare=True`), same-class rule included, ahead of any engine call and operating on the two instances alone. Evaluates the canonical relation directly. Raises `TypeError` when `left` is anything other than a dataclass instance.


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


## `ConfigRoot` method map

`ConfigRoot` is a mixin for the root dataclass; each method delegates to its free-function twin, and subclassing installs [canonical equality](schema-design.md#canonical-equality) at class-creation time. Subclasses still carry the `@dataclass` decorator, and nested sections are plain dataclasses.

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


---

[Previous: Files, formats, and run identity](files-and-identity.md) | [Home](README.md)
