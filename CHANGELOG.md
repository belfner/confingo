# Changelog

All notable changes to confingo are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `ConfigNode` replaces `ConfigRoot` as the name of the method facade, and any
  dataclass in a config tree may subclass it rather than the root alone. Each
  method is scoped to the node it is called on, so a nested node builds,
  exports, fingerprints, and writes its own subtree, and issue paths from a
  subtree load are relative to that node. Attaching the base to a section
  changes its method surface alone: the engine reaches a nested section through
  the same generic recursion either way, so an enclosing load produces identical
  values, exported data, digests, and issue paths. Update imports and base
  classes from `ConfigRoot` to `ConfigNode`.
- **Breaking.** Config dataclasses are unhashable. From its first schema
  processing a class carries `__hash__ = None`, so `hash(config)`, a config as a
  mapping key, and a config in a set each raise `TypeError`;
  `config_hash(config)` is the stable value-identity operation. This covers
  frozen configs, where Python otherwise generates a field-tuple hash that
  disagrees with canonical equality and raises on array fields. A `ConfigNode`
  subclass holds the same contract from class creation through a `__hash__` that
  raises `TypeError: unhashable type: 'RunConfig'; use config_hash(config) for
  value identity`. Replace `hash(config)` and set or mapping membership with
  `config_hash(config)`.
- **Breaking.** Set annotations whose elements carry a config section --
  directly, through a union, or inside an immutable `tuple` / `frozenset` shape
  -- are rejected at schema preflight, naming the annotation as written and
  pointing at a list or tuple for the collection plus `config_hash(section)` as
  the value-identity key. Sets of hashable values are unaffected. Rewrite
  `frozenset[Section]` as `list[Section]` or `tuple[Section, ...]`.
- **Breaking.** Authored defaults are validated rather than trusted. A default
  has to already carry the runtime type its annotation names and to have a plain
  serializable form, so `output_dir: Path = "runs"` is reported as an authoring
  error rather than promoted to `Path("runs")`, as are a list default for a tuple
  field, an integral float for an `int` field, a mapping for a section, and a
  value under `Any` with no plain form. A direct `field(default=...)` is checked
  during schema preflight, whether or not the input supplies the field, so a
  wrong default surfaces even where it is always overridden. A `default_factory`
  runs once at the build that selects it; its one product is validated and passed
  on unchanged, and a factory that raises reports as
  `items: default_factory raised ValueError: <message>`. Write defaults in the
  annotated type: `Path("runs")`, `("a", "b")`, `Optimizer(lr=1e-3)`.
- The type-boundary message names the supported annotation categories and the
  `init=False` remedy: `unsupported field type Decimal; choose a supported
  annotation (bool, int, float, str, Path, date/time, Enum/Literal, dataclass,
  container/union, array/tensor, or Any) and derive other runtime values in an
  init=False field`. One builder produces it, so schema preflight and
  construction word the boundary identically.
- A class named as a field type that declares its own annotations without being
  a dataclass reports the decorator remedy at that field's path, `server: Forgot
  is not a dataclass, so it carries no config schema. Declare it with
  @dataclass.`, matching what an entry class of the same shape already reported.
  It reaches the field through a section, a container, a union, or a mapping
  value alike. A `TypedDict`, a `NamedTuple`, and a class with no annotations of
  its own stay on the type-boundary message.

### Added

- A `ConfigNode` subclass reserves the eleven facade names (`from_dict`,
  `load_json`, `load_yaml`, `from_file`, `to_dict`, `dumps_json`, `save_json`,
  `dumps_yaml`, `save_yaml`, `to_file`, `config_hash`). A node declares none of
  them: any annotation or class-body binding under one of these names is
  rejected at class creation, as is one supplied by a base ahead of `ConfigNode`
  in the MRO, inherited as a field, or supplied as a metaclass data descriptor,
  with every collision on the class reported together. Declarations are read
  rather than resolved, so a descriptor under a reserved name is never run. The
  same names carry no restriction on a plain dataclass.
- A `ConfigNode` subclass that inherits a hand-written `__eq__` or `__hash__`
  from a base is rejected at class creation, naming the base that owns it, since
  the canonical methods land on the subclass ahead of the decorator.
- A `ConfigNode` subclass that declares annotations without carrying
  `@dataclass` is reported as a schema error at its schema path: its own names
  stay outside the schema while the inherited fields load. `ClassVar`
  annotations raise no such error.
- An entry class that is not a dataclass is reported as a schema issue carrying
  the calling operation's context, in place of the standard library's bare
  `TypeError` from `dataclasses.fields`.

- Canonical equality on every schema dataclass: two configs are `==` exactly
  when their compared fields (`init=True` and `compare=True`) serialize to the
  same plain form at every tree level, array fields included. Array and tensor pairs compare through the backends'
  vectorized equality wherever that comparison is provably exact (with a
  tensor meeting a numpy array via `detach().cpu().numpy()`), and every
  other pair compares by its serialized form, so `==` on large arrays runs
  at native speed with exact value semantics. A `ConfigNode` subclass
  carries the canonical `__eq__` from class-creation time, installed by
  `__init_subclass__` ahead of the `@dataclass` decorator. Every other schema
  dataclass receives canonical equality at its first schema processing,
  replacing the generated `__eq__` it carried. The new
  `config_equal(left, right)` free function exposes the same relation ahead of
  any engine call.
- confingo owns equality and hashing on config dataclasses: a class that
  hand-writes `__eq__` or `__hash__` is rejected -- a `ConfigNode` subclass at
  class creation (both reported together when it defines both), a section at its
  first schema touch -- and a `@dataclass` flag confingo cannot honor
  (`init=False`, `unsafe_hash=True`, `eq=False`, `order=True`) raises a
  `ConfigError` at first schema processing, every violation on one class reported
  together. A `ConfigNode` subclass declared `unsafe_hash=True` is the exception:
  it fails at class creation with the standard-library `TypeError` for
  overwriting `__hash__`, since the node installs its own `__hash__` ahead of the
  decorator. `frozen=True`, `slots=True`, and `weakref_slot=True` are supported;
  a generated or inherited `__hash__` becomes `None` at first schema processing,
  written on the touched class so an untouched base keeps its own.
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
  (`load_yaml`, `save_yaml`, `dumps_yaml`, and the matching `ConfigNode`
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
- Internal restructure of the engine: the marshal / unmarshal core is split into
  focused modules (`_errors`, `_schema`, `_core`, `_serialize`), the NumPy and
  PyTorch array paths share one validation and indexed issue-reporting kernel,
  scalar coercion routes through a shared ISO temporal parser, and equality and
  hashing ownership resolves through one method-contract helper. Every docstring
  follows the project's Google style. The public API (`from confingo import ...`)
  and observable behavior are preserved.
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
