[Documentation home](README.md)

# Validation and errors

confingo walks the whole config tree in one pass and reports every problem it finds in a single exception, each issue tagged with a dotted path. This page covers the error objects, the two validation phases, the built-in issue sources, and how custom invariants join the same report.


## Read an error in three steps

Each rendered report answers three questions in order:

1. **Context.** The summary line names the source: `config file train.json` from a file loader, `config` from a direct `from_dict`, or `config schema` for an annotation problem.
2. **Path.** The dotted path locates the value in the tree (`optimizer.lr`, `stages.0.name`), with `<root>` for a whole-object issue.
3. **Message.** The text names what the value needs.

Common messages and where their rule lives:

| Message | Condition that produces it | Where the rule lives |
| --- | --- | --- |
| `unknown key (known keys: ...)` | an input key beyond the schema's fields | [Schema design](schema-design.md) |
| `missing required value` | a required leaf awaiting a value | [implicit sections](schema-design.md#implicit-sections-and-leaf-level-requirements) |
| `expected <type>, got <type>` | a value whose shape differs from the annotation | [Types and coercion](types-and-coercion.md) |
| `expected one of 'cpu' \| 'cuda', got 'tpu'` (enums add `for enum <Name>`) | a value differing from every declared `Literal` or enum option | [enums and literals](types-and-coercion.md#enums-and-literals) |
| `field is not configurable (init=False)` | a key supplied for a runtime field | [field options](schema-design.md#field-options) |
| `ragged array: expected N items, got M` | an array input with rows of differing length | [Arrays and tensors](arrays-and-tensors.md) |
| a custom message | a `__post_init__` or `__validate__` invariant | [dataclass invariants](#dataclass-invariants) |


## One exception, every discovered issue

A file with several mistakes produces one `ConfigError` listing all of them:

```
confingo.ConfigError: config file train.json has 4 issues:
  - sed: unknown key (known keys: data, device, model, optimizer, output_dir, seed)
  - data.batch_sizes: unknown key (known keys: augmentations, batch_size, dataset_path, num_workers)
  - optimizer.name: expected one of 'adamw' | 'sgd', got 'adam'
  - optimizer.lr: expected float, got str
```

Paths use dots for fields, indexes for sequence elements, and keys for mapping entries: `stages.0.name`, `datasets.train.path`.

Paths are relative to the object the operation was entered on, and `<root>` names that object. Loading through the whole config reports a leaf as `optimizer.lr`; loading the same section through `OptimizerConfig.cfg.from_dict(...)` reports it as `lr`, since the mapping supplied to that call is the document the path locates values in. Pass `context="optimizer override"` to label which source a subtree load was reading.


## Pending lifecycle work

A build reaches a node's lifecycle after its fields build, so an issue in a field leaves that node's `__post_init__`, `init=False` completeness check, and `__validate__` for a later load. `ConfigError.pending_lifecycle_paths` names where that work waits, and a closing line states it:

```
confingo.ConfigError: config file train.json has 1 issue:
  - model.optimizer.lr: expected float, got str
  Pending lifecycle work at model.optimizer, model, <root>: fix the issues above, then load the config again to run the applicable callbacks and checks.
```

The attribute is a `tuple[str, ...]` in discovery order, deepest node first, with the root as the empty string and `<root>` in the rendering. The rendered sentence names the first five paths and counts the rest; the attribute carries them all.

An entry is one of two things. A node entry names a node with stages still ahead of it. A barrier entry names a path where an authored default was set aside with its lifecycle unvisited, and covers that path together with anything beneath it.

The reading is deliberately generous: a listed callback may run and return an empty list, and a barrier may cover a subtree whose sections are all quiet. An entry marks work a repair reaches. Naming a path that turns out quiet costs one entry, and staying quiet about a path that later reports is the surprise the signal exists to end. Hook discovery reads statically visible class declarations along the MRO, taking the nearest binding, so a `__validate__ = None` in a subclass answers for the base method it shadows. A hook that an instance `__getattr__` supplies becomes visible once an instance exists, which is past the point this reading is taken.

Trials stay private: a union member is probed with its own collector, so the report carries the pending paths of the member it names, alongside that member's issues.


## Handling errors in a training CLI

Catch `ConfigError` at startup and fail fast, before allocating accelerators:

```python
def main() -> int:
    try:
        config = TrainingConfig.cfg.from_file(sys.argv[1])
    except ConfigError as err:
        print(err, file=sys.stderr)
        return 2
    run(config)
    return 0
```

The rendered error already includes the context, the issue count, and one line per issue, so printing the exception gives users an actionable report.


## Dataclass invariants

Custom checks join the report through two hooks on any dataclass in the tree:

- **`__post_init__`**: whatever it raises becomes one issue at the dataclass's path, reading `constructing <Class> raised ValueError: <message>`.
- **`__validate__`**: returns an iterable of messages, and an empty list when the config is valid; each message becomes its own issue at that path, so one hook can report several independent problems. Whatever the hook raises becomes one issue, reading `validating <Class> raised ValueError: <message>`.

Both hooks belong to the config author, so what they raise describes the config and arrives beside the issues its siblings reported. A `MemoryError` or `SystemError` travels to the caller unchanged, since it describes the interpreter rather than the config.

`__validate__`'s return is read against its contract before it is consumed. A bare string satisfies "an iterable" while iterating to one issue per character, and `None` is what an `if bad: return [...]` with no trailing return hands back; each is reported as `<Class>.__validate__ returned str; __validate__ returns an iterable of messages; return a list of strings, or an empty list when the config is valid` rather than acted on.

Per node the order is `__init__` (which runs `__post_init__` as its final step), then the [`init=False`](schema-design.md#field-options) completeness check, then `__validate__`. The completeness check sits between them so `__post_init__` has populated the runtime fields first, and `__validate__` runs only on a fully populated instance, so a node whose `init=False` field still awaits a value contributes that completeness issue in its place.

```python
@dataclass
class CheckpointConfig:
    keep_last: int = 3

    def __post_init__(self) -> None:
        if self.keep_last < 1:
            raise ValueError("keep_last must be >= 1")


@dataclass
class TrainerConfig:
    lr: float
    hidden_widths: tuple[int, ...]

    def __validate__(self) -> list[str]:
        messages = []
        if self.lr <= 0:
            messages.append("lr must be positive")
        if len(self.hidden_widths) == 0:
            messages.append("at least one hidden layer is required")
        return messages
```

Loading `{"lr": -1.0, "hidden_widths": []}` against `TrainerConfig` renders both `__validate__` messages at the node's path, `<root>` here since `TrainerConfig` is the root:

```
confingo.ConfigError: config has 2 issues:
  - <root>: lr must be positive
  - <root>: at least one hidden layer is required
```


## Exact contracts

The shape of the exception, when each phase runs, every issue confingo itself can report, and where aggregation stops.


## `ConfigError` and `ConfigIssue`

`ConfigError` subclasses `ValueError`, so existing `except ValueError` handlers catch it. It carries:

- `.issues`: a tuple of `ConfigIssue` objects in discovery order.
- `.context`: the source being validated (`"config"` by default; file loaders use `config file <path>`).

Each `ConfigIssue` is a frozen dataclass with `.path` and `.message`. `str(issue)` renders `path: message`, with an empty path shown as `<root>`.

```python
try:
    config = TrainingConfig.cfg.load_json(path)
except ConfigError as err:
    for issue in err.issues:
        print(issue.path, "->", issue.message)
```


## Two validation phases

1. **Schema preflight.** Every annotation in the tree is checked against the [accepted boundary](types-and-coercion.md#accepted-schema-boundary), including fields that will use defaults. Schema problems raise immediately.
2. **Construction.** The input mapping is walked against the schema, coercing values and collecting every discoverable issue before raising.

The context on the raised error tells you where the problem came from:

- File loaders attach the file path (`config file train.json`) to read, parse, preflight, and construction issues.
- An annotation that fails to resolve raises with the `config schema` context.
- Direct callers can set their own via `from_dict(cls, data, context="sweep entry 3")`.


## Built-in issue sources

- Unknown keys, with the sorted known-key list in the message.
- A key supplied for an [`init=False`](schema-design.md#field-options) field: `field is not configurable (init=False)`, since `__post_init__` populates runtime fields.
- Missing required values: required fields awaiting a value from the input. Dataclass sections build implicitly, so a required value reached through an implicit build is reported at its nested dotted path ([details](schema-design.md#implicit-sections-and-leaf-level-requirements)).
- A value other than a mapping supplied for a dataclass section or document root.
- Type mismatches and tuple-arity mismatches from [coercion](types-and-coercion.md).
- Enum and `Literal` values outside the declared options.
- A set element supplied as a `str`, `int`, or temporal subclass whose hashing or equality fails, reported under the annotation as written: `cannot build set[str]: unhashable type: 'NoHashStr'`. Building a set runs both, so a collision that reaches a failing `__eq__` reports the same way.
- A `set` or `frozenset` element annotation that rebuilds a value a set cannot hold, reported at preflight with the annotation as written: `set[list[int]] cannot be built: a set element must rebuild hashable when a file is loaded`. The message names a scalar element, a tuple of scalars, or a list as the remedy. An element is admitted when it is a scalar, or a `tuple` / `frozenset` whose own arguments recursively satisfy that same rule.
- A `set` or `frozenset` annotation whose elements carry a config section, reported the same way with a remedy of its own: `config sections are unhashable, so frozenset[Section] cannot be built`. The message points at a list or tuple for the collection and `config_hash(section)` as the value-identity key. Sections reached through a union or an immutable `tuple` / `frozenset` shape report the same way, and defects inside the section aggregate alongside it.
- Non-finite floats anywhere in supplied data.
- A contradictory field declaration, `field(hash=True, compare=False)`, reported at preflight: a field in the fingerprint must participate in equality.
- An authored default outside its annotation or without a plain form, since [defaults are validated rather than coerced](schema-design.md#leaf-defaults-and-precedence): `invalid authored default: expected a value already matching Path, got str; defaults are validated as written`. A direct `field(default=...)` reports at preflight whether or not the input supplies the field; a `default_factory` reports at the build that selects it, under `invalid default_factory value`, and a factory that raises reports as `default_factory raised ValueError: <message>`.
- An `init=False` field still awaiting a value after `__post_init__`: `init=False field was not set during __post_init__`. A section an authored default supplied reaches this check too, so the same section reports the same way whether the input carried it or the default did.
- An invariant a section an authored default supplied reports from `__validate__`, at that section's own path. The default arrives already constructed, so its `__post_init__` has run and the build that selects it runs the rest of the lifecycle, which keeps the report from depending on whether the file mentioned the section.
- A schema class that is also one of the kinds a walk dispatches on, reported at that class's path: `<Class> is a config section and also a mapping, and every walk over a value reads those as two different things, so what the section writes and what a file rebuilds it as follow from which reading a walk reaches first; declare the section as a plain dataclass, and carry the mapping on an object held in an init=False field`. The named kinds are `Mapping`, `Sequence`, `set`, `frozenset`, `Enum`, `Path`, `date`, and `time` ([the rule](schema-design.md#what-a-schema-class-may-be)).
- A constructor that cannot be handed the class's own fields. One that will not receive a field reads `<Class>.__init__ takes no value argument for the field of that name, and confingo builds a config object by calling the class with its field names; leave the generated __init__ in place, and derive anything else in __post_init__ or an init=False field`. One that demands something else, a required `InitVar` among them, reads `<Class>.__init__ requires the raw argument, which names no field a config file can supply; give it a default, or declare it as an ordinary field when the file carries the value`.
- An annotation that names no element type, at the field carrying it. `Any` reads `Any leaves the values it holds undescribed; annotate the field ConfigValue (from confingo) for plain data of any shape, or name the type the field holds`, and an argument-free container reads `list carries no element type; write list[ConfigValue] for plain data of any shape, or name the element type` with the form matching the container written.
- A subclass of `Path`, `datetime`, `date`, or `time`, since a load builds the base class: `SpecialDate is a date subclass, and a load builds date itself; annotate the field date, and derive the subclass in an init=False field`.
- A `set` or `frozenset` whose element names a union: `set[str | Path] cannot be built: a set element names one type, and str | Path names several, so a load can hand one member's plain form to another and rebuild an element the file did not carry; name the one type the elements carry, write T | None for an optional element, or hold the values in a list`.
- A value nested deeper than 64 levels of plain data, or one that reaches itself: `nesting reaches the 64 level limit for plain data` and `value holds itself, so it has no plain form; supply a structure that terminates`. One budget covers the whole plain document, so a value the load declines is one no authored default or factory can carry past export.
- A schema class that hand-writes `__eq__` or `__hash__`, since [confingo owns equality and hashing](schema-design.md#canonical-equality): `<Class> defines a custom __eq__` / `__hash__`. A `ConfigNode` subclass reports this at class creation, both together when it defines both, and reports the same way when it inherits a hand-written definition from a base, naming the base that owns it; a plain dataclass reports it at its first schema touch.
- A schema class declared with a `@dataclass` flag that conflicts with confingo's ownership of equality and hashing (`init=False`, `unsafe_hash=True`, `eq=False`, or `order=True`), each named in the message, with every violation on one class reported together at first schema processing. A `ConfigNode` subclass declared `unsafe_hash=True` is the exception: it fails at class creation with the standard-library `TypeError` for overwriting `__hash__`, because the node installs its own `__hash__` ahead of the decorator.
- A `ConfigNode` subclass that declares or inherits the reserved name `cfg`: `<Class>.cfg is declared as a field, which shadows the ConfigNode method of the same name`. Reported at class creation. The message names where the shadowing member comes from, so a class-body binding reads `is bound in the class body`, an inherited field reads `is inherited as a field from <Base>`, a base member reads `is supplied by base <Base>`, and a metaclass descriptor reads `is supplied as a data descriptor by metaclass <Meta>`.
- A `ConfigNode` subclass that declares annotations without carrying `@dataclass`: its own names stay outside the schema and only the inherited fields load, so the class is reported at its schema path with the remedy to decorate it. A `ClassVar` annotation passes, since it declares a class attribute rather than a field.
- A class that is not a dataclass, reported instead of the bare `TypeError` the standard library would raise: `<Class> is not a dataclass, so it carries no config schema. Declare it with @dataclass.` An entry class reports at the root, carrying the calling operation's context, so a file load still reports against the file it was reading. A class named as a field type reports at that field's path, and reaches it through a section, a container, a union, or a mapping value alike. The route is chosen by the class declaring its own annotations without carrying dataclass fields, which is what a forgotten decorator looks like; a `TypedDict`, a `NamedTuple`, and a class with no annotations of its own each stay on the type-boundary message below.
- An annotation outside the supported type set, at the field carrying it: `unsupported field type Decimal; choose a supported annotation (bool, int, float, str, Path, date/time, Enum/Literal, dataclass, container/union, array/tensor, or ConfigValue/ConfigScalar for plain data) and derive other runtime values in an init=False field`. One builder produces this wherever the boundary is reached, so preflight and construction word it identically. The [`init=False`](schema-design.md#field-options) remedy is the supported way to hold a runtime object such as an open connection or a compiled model beside the configured values.


## Aggregation boundaries

Collection is exhaustive across siblings: every field, sequence element, and mapping entry is visited even after earlier ones fail. Two boundaries shape what appears in a single report:

- A dataclass node whose fields have issues stays unconstructed for that load, so its `__post_init__` / `__validate__` hooks run on the next load, once the field issues are fixed. Fixing a config can therefore surface a deeper round of messages. [`pending_lifecycle_paths`](#pending-lifecycle-work) names where that work waits, so the later round is stated ahead of time.
- The same holds wherever a node is set aside before its lifecycle: a constructor that raised, a completeness check that failed, a section receiving another runtime shape, a nesting or cycle limit, and an authored default whose product was declined. Each leaves the stages after it for a later load.
- A union member is taken when its trial converts cleanly, so a member that builds and then reports an invariant leaves the whole union unmatched and the enclosing node unconstructed. A config can therefore take three loads to settle: one for a field type, one for the member's invariant, one for the enclosing node's own.
- A multi-member union tries each member privately. When every member fails, the field reports a summary, `expected AdamW | SGD; best match SGD failed with 1 issue`, followed by that one member's own issues at their own paths. The closest member is the one whose attempt collected the fewest issues, with an equal count going to the first declared member. A single-type optional (`T | None`) coerces straight through `T`, preserving detailed child paths and running hooks exactly once.


## Related pages

- [Schema design](schema-design.md) for structural fixes and hook placement.
- [Types and coercion](types-and-coercion.md) for the conversion rules behind type-mismatch issues.
- [Files, formats, and run identity](files-and-identity.md#document-and-read-rules) for parse and read failures.
- [API reference](api-reference.md#error-types) for exact signatures.

---

Exact reference: [Schema design](schema-design.md) | [Types and coercion](types-and-coercion.md) | [Equality and hashing](equality-and-hashing.md) | [API reference](api-reference.md) | [Documentation home](README.md)
