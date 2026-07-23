[Documentation home](README.md)

# API reference

Compact reference for the public surface. Signatures are exact; behavior details live on the linked guide pages.


## Import surface

```python
from confingo import (
    ConfigError,
    ConfigIssue,
    ConfigRoot,
    ConfigWarning,
    config_hash,
    configclass,
    dumps_json,
    from_dict,
    from_file,
    load_json,
    save_json,
    to_dict,
    to_file,
)
from confingo import dumps_yaml, load_yaml, save_yaml  # confingo[yaml]
```

The YAML helpers resolve lazily on first attribute access, so the core import works on a stdlib-only install. `confingo.__version__` carries the package version.


## Schema declaration

Learn more: [configclass and equality](schema-design.md#configclass-and-equality).

### `@configclass` / `@configclass(**dataclass_kwargs)`

Declares a config dataclass: fields, defaults, and `__init__` generate exactly as `@dataclass` generates them, and the `dataclass()` keywords `init`, `repr`, `frozen`, `match_args`, `kw_only`, `slots`, and `weakref_slot` forward. Installs canonical `__eq__` (`to_dict(self) == to_dict(other)`, `NotImplemented` for a different class) with `__hash__` kept as object identity. A user-defined `__eq__` in the class body is respected and carries standard Python hashing semantics: define `__hash__` alongside it to keep instances hashable. Passing `eq`, `order=True`, or `unsafe_hash=True` raises `TypeError`.

### `ConfigWarning`

`UserWarning` subclass carrying confingo schema advisories. Emitted once per class per process when a schema dataclass lacks the `@configclass` marker; target it with `warnings.filterwarnings` to silence or escalate precisely.


## Construction and conversion

Learn more: [Schema design](schema-design.md), [Types and coercion](types-and-coercion.md).

### `from_dict(config_cls, data, *, context="config") -> T`

Builds `config_cls` (a dataclass type) from a mapping, coercing each value toward its annotation and collecting every issue before raising `ConfigError`. `context` names the source in error messages.

### `to_dict(value) -> Any`

Converts a config object to plain serializable data in field-declaration order. Raises `ConfigError` for values outside the plain-data model.

### `config_hash(config, *, length=12) -> str`

SHA-256 fingerprint of the resolved config's canonical JSON: the digest's leading `length` hex characters (useful range 1-64; the full digest is 64). Stable across processes and hash seeds. Learn more: [stable run identity](files-and-identity.md#stable-run-identity).


## JSON functions

Learn more: [Files, formats, and run identity](files-and-identity.md#json).

| Function | Behavior |
| --- | --- |
| `load_json(config_cls, path) -> T` | Reads UTF-8 JSON, builds the config; read, parse, preflight, and construction failures raise `ConfigError` with `config file <path>` context ([annotation-resolution failures use `config schema`](validation-and-errors.md#two-validation-phases)) |
| `dumps_json(config, *, indent=2) -> str` | JSON text, ASCII-escaped, trailing newline |
| `save_json(config, path, *, indent=2) -> Path` | Atomic write; returns the destination path |


## YAML functions

Require the `yaml` extra; on a core-only install these raise `ImportError` with the install hint. Learn more: [YAML extra](files-and-identity.md#yaml-extra).

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

`ConfigRoot` is a mixin for the root dataclass; each method delegates to its free-function twin. Subclasses still carry the `@configclass` (or `@dataclass`) decorator, and nested sections carry the decorator alone.

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
