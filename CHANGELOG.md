# Changelog

All notable changes to confingo are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Canonical equality on every schema dataclass: two configs are `==` exactly
  when their compared fields (`init=True` and `compare=True`) serialize to the
  same plain form at every tree level, array fields included. Array and tensor pairs compare through the backends'
  vectorized equality wherever that comparison is provably exact (with a
  tensor meeting a numpy array via `detach().cpu().numpy()`), and every
  other pair compares by its serialized form, so `==` on large arrays runs
  at native speed with exact value semantics. A `ConfigRoot` subclass
  carries the canonical `__eq__` and identity `__hash__` from class-creation
  time, installed by `__init_subclass__` ahead of the `@dataclass`
  decorator, and a body-defined `__eq__` there is respected with standard
  Python hashing semantics. Every other schema dataclass receives canonical
  equality at its first schema processing, replacing the `__eq__` it
  carried, with identity hashing restored where generating `__eq__` had
  disabled it. The new `config_equal(left, right)` free function exposes the
  same relation ahead of any engine call.
- NumPy array and PyTorch tensor fields, presence-detected: the backends
  install with the application, confingo's core stays stdlib-only, and
  `import confingo` loads neither. Supported annotations: bare `np.ndarray`,
  `npt.NDArray[...]` with concrete dtypes or abstract families, shape-typed
  `np.ndarray[tuple[int, int], np.dtype[...]]` with dimensionality
  enforcement, bare `torch.Tensor` (rebuilt with value-stable pinned dtypes
  bool / int64 / float64), and `Annotated[torch.Tensor, torch.dtype]`, where a
  fixed-arity shape tuple in the metadata
  (`Annotated[torch.Tensor, torch.float32, tuple[int, int]]`) enforces
  dimensionality exactly as the numpy shape spelling does. Values
  serialize as the validated `tolist()` form, detached and copied to the CPU
  for tensors; plain input validates leaf by leaf with indexed issue paths
  (`weights.2.0`), supplied arrays validate with vectorized masks, and every
  array field is capped at one million elements. Arrays under `Any` validate
  inbound and serialize as plain scalars and lists. Supported numpy scalars
  feed ordinary
  scalar fields as their exact Python equivalents.
- Dataclass `field()` options, with `init` as the master switch. An
  `init=False` field is runtime state: it is excluded from loading, export,
  equality, and the `config_hash` fingerprint, its `compare` / `hash` flags are
  inert, and its annotation is exempt from the supported-type boundary so it may
  hold any resolvable runtime object. It is populated by its default or in
  `__post_init__`, and every `init=False` field is checked for population after
  construction (before `__validate__`), so one left unset is reported as
  `init=False field was not set during __post_init__` rather than surfacing later
  as an `AttributeError`. Supplying an `init=False` field's key in the input is
  reported as `field is not configurable (init=False)`. On an `init=True` field,
  `compare=False` drops the field from equality and therefore from the
  fingerprint while `to_dict` still carries it, and `hash=False` drops it from
  the fingerprint alone while equality keeps it. `field(hash=True, compare=False)`
  is reported as a contradiction, since a field in the fingerprint must
  participate in equality.
- `config_hash` fingerprints the hashing fields (`init=True`, `compare=True`,
  effective hash enabled) rather than the full `to_dict` output, so a
  `compare=False` or `hash=False` field is serialized yet excluded from the
  digest.

### Changed

- PyYAML (`>=6.0`) is now a core runtime dependency, so YAML file IO
  (`load_yaml`, `save_yaml`, `dumps_yaml`, and the matching `ConfigRoot`
  methods) and extension dispatch work from the base install. The three YAML
  helpers are importable directly from `confingo` and resolve at import time.
  The `yaml` optional extra is removed; `pip install confingo` now carries YAML
  support.
- Python 3.11 is the minimum supported version.
- `to_dict` collects every serialization problem in one pass, each tagged with
  its dotted path, matching the collect-all model `from_dict` already follows.
- Dataclass sections instantiate implicitly. A dataclass-typed field with no
  default builds from an empty mapping when the input omits it, recursively
  through nested sections, so a required value inside an omitted section is
  reported at its nested dotted path (`optimizer.name: missing required value`).
  Every other undefaulted field stays required when absent -- scalars, unions,
  `Any`, and containers -- keeping a forgotten container distinct from an
  intentionally empty one authored as `field(default_factory=list)`. A
  self-referential section terminates with a missing-value issue at the point of
  recursion. Explicit defaults and `default_factory` values take precedence and
  are used as authored.

## [0.2.0] - 2026-07-22

### Added

- Optional YAML file IO behind the `yaml` extra (`pip install confingo[yaml]`):
  `load_yaml`, `save_yaml` (atomic write), and `dumps_yaml`, plus matching
  `ConfigRoot.load_yaml` / `save_yaml` / `dumps_yaml` methods. The helpers move
  through the same JSON-compatible data model as the JSON loaders, so a config
  round-trips across both formats. PyYAML is imported lazily on first use, so
  importing confingo needs only the standard library.
- Extension-dispatching file IO: `from_file` / `to_file` (and matching
  `ConfigRoot.from_file` / `to_file` methods) select JSON or YAML from the path
  extension (`.json`, `.yaml`, `.yml`), raising `ConfigError` when the extension
  names no supported format.
- `datetime`, `date`, and `time` fields: ISO 8601 strings and native
  `datetime` / `date` / `time` objects load into the annotated type, and
  `to_dict` renders them back as ISO 8601 strings.

### Changed

- The supported field-type set is an explicit boundary, validated against the
  schema itself so an unsupported annotation is reported even when the field is
  omitted. Enforced: allowed leaf and container types only, `str` mapping keys
  (including bare `dict`), primitive `Enum` / `Literal` values, finite floats,
  constructor-settable (`init=True`) fields, and nested dataclasses recursively.
  `to_dict` raises a `ConfigError` for a value it cannot render as plain data.
- `config_hash` orders set elements by their canonical JSON text, so the digest is
  stable across processes for mixed-type sets. Documented as stable across
  processes and independent of mapping key order and set iteration order.
- File loaders report a `ConfigError` (rather than a raw exception) for invalid
  UTF-8, non-finite floats, unhashable set elements, and out-of-range numbers.
  Atomic writes use a uniquely named temporary that preserves the destination's
  file mode.

## [0.1.0] - 2026-07-22

### Added

- Marshal / unmarshal core over dataclasses: `from_dict` builds a validated
  dataclass tree from a nested mapping, and `to_dict` renders a config object
  back to plain data in field-declaration order. The pair round-trips:
  `from_dict(cls, to_dict(config)) == config`.
- Type coercion toward each field's annotation: enums by value or name, `Literal`
  membership, unions, fixed-length and variadic tuples, sets and frozensets,
  `dict[str, X]`, `Path`, and integral-float-to-int, with `bool` held off `int`
  and `float` fields.
- Collect-all validation: one pass walks the whole tree and reports every issue,
  each tagged with a dotted path such as `training.trainer.lr`, surfaced through
  `ConfigError` and `ConfigIssue`. Custom `__validate__` messages and a
  `ValueError` or `TypeError` from `__post_init__` fold into the same report.
- JSON file IO: `load_json`, `save_json` (atomic write), and `dumps_json`.
- `config_hash` for a stable SHA-256 fingerprint over the canonical JSON form.
- `ConfigRoot` mixin exposing the same operations as methods on a root config
  dataclass (`Config.load_json(path)`, `config.save_json(path)`).
- `py.typed` marker; the package ships with inline type annotations and runs on
  the Python standard library alone, targeting Python 3.10 and newer.

[0.2.0]: https://github.com/belfner/confingo/releases/tag/v0.2.0
[0.1.0]: https://github.com/belfner/confingo/releases/tag/v0.1.0
