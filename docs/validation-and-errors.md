[Documentation home](README.md)

# Validation and errors

confingo walks the whole config tree in one pass and reports every problem it finds in a single exception, each issue tagged with a dotted path. This page covers the error objects, the two validation phases, the built-in issue sources, and how custom invariants join the same report.


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

Paths are relative to the object the operation was entered on, and `<root>` names that object. Loading through the whole config reports a leaf as `optimizer.lr`; loading the same section through `OptimizerConfig.from_dict(...)` reports it as `lr`, since the mapping supplied to that call is the document the path locates values in. Pass `context="optimizer override"` to label which source a subtree load was reading.


## Read an error in three steps

Each rendered report answers three questions in order:

1. **Context** — the summary line names the source: `config file train.json` from a file loader, `config` from a direct `from_dict`, or `config schema` for an annotation problem.
2. **Path** — the dotted path locates the value in the tree (`optimizer.lr`, `stages.0.name`), with `<root>` for a whole-object issue.
3. **Message** — the text names what the value needs.

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


## `ConfigError` and `ConfigIssue`

`ConfigError` subclasses `ValueError`, so existing `except ValueError` handlers catch it. It carries:

- `.issues`: a tuple of `ConfigIssue` objects in discovery order.
- `.context`: the source being validated (`"config"` by default; file loaders use `config file <path>`).

Each `ConfigIssue` is a frozen dataclass with `.path` and `.message`. `str(issue)` renders `path: message`, with an empty path shown as `<root>`.

```python
try:
    config = TrainingConfig.load_json(path)
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
- Unhashable elements supplied for set fields.
- A `set` or `frozenset` annotation whose elements carry a config section, reported at preflight with the annotation as written: `config sections are unhashable, so frozenset[Section] cannot be built`. The message points at a list or tuple for the collection and `config_hash(section)` as the value-identity key. Sections reached through a union or an immutable `tuple` / `frozenset` shape report the same way, and defects inside the section aggregate alongside it.
- Non-finite floats anywhere in supplied data.
- A contradictory field declaration, `field(hash=True, compare=False)`, reported at preflight: a field in the fingerprint must participate in equality.
- An authored default outside its annotation or without a plain form, since [defaults are validated rather than coerced](schema-design.md#leaf-defaults-and-precedence): `invalid authored default: expected a value already matching Path, got str; defaults are validated as written`. A direct `field(default=...)` reports at preflight whether or not the input supplies the field; a `default_factory` reports at the build that selects it, under `invalid default_factory value`, and a factory that raises reports as `default_factory raised ValueError: <message>`.
- An `init=False` field still awaiting a value after `__post_init__`: `init=False field was not set during __post_init__`.
- A schema class that hand-writes `__eq__` or `__hash__`, since [confingo owns equality and hashing](schema-design.md#canonical-equality): `<Class> defines a custom __eq__` / `__hash__`. A `ConfigNode` subclass reports this at class creation, both together when it defines both, and reports the same way when it inherits a hand-written definition from a base, naming the base that owns it; a plain dataclass reports it at its first schema touch.
- A schema class declared with a `@dataclass` flag that conflicts with confingo's ownership of equality and hashing -- `init=False`, `unsafe_hash=True`, `eq=False`, or `order=True` -- each named in the message, with every violation on one class reported together at first schema processing. A `ConfigNode` subclass declared `unsafe_hash=True` is the exception: it fails at class creation with the standard-library `TypeError` for overwriting `__hash__`, because the node installs its own `__hash__` ahead of the decorator.
- A `ConfigNode` subclass that declares or inherits one of the eleven reserved method names: `<Class>.<name> is declared as a field, which shadows the ConfigNode method of the same name`. Reported at class creation, with every collision on one class together. The message names where the shadowing member comes from, so a class-body binding reads `is bound in the class body`, an inherited field reads `is inherited as a field from <Base>`, a base member reads `is supplied by base <Base>`, and a metaclass descriptor reads `is supplied as a data descriptor by metaclass <Meta>`.
- A `ConfigNode` subclass that declares annotations without carrying `@dataclass`: its own names stay outside the schema and only the inherited fields load, so the class is reported at its schema path with the remedy to decorate it. `ClassVar` annotations raise no such error, since they are not fields in the first place.
- A class that is not a dataclass, reported instead of the bare `TypeError` the standard library would raise: `<Class> is not a dataclass, so it carries no config schema. Declare it with @dataclass.` An entry class reports at the root, carrying the calling operation's context, so a file load still reports against the file it was reading. A class named as a field type reports at that field's path, and reaches it through a section, a container, a union, or a mapping value alike. The route is chosen by the class declaring its own annotations without carrying dataclass fields, which is what a forgotten decorator looks like; a `TypedDict`, a `NamedTuple`, and a class with no annotations of its own each stay on the type-boundary message below.
- An annotation outside the supported type set, at the field carrying it: `unsupported field type Decimal; choose a supported annotation (bool, int, float, str, Path, date/time, Enum/Literal, dataclass, container/union, array/tensor, or Any) and derive other runtime values in an init=False field`. One builder produces this wherever the boundary is reached, so preflight and construction word it identically. The [`init=False`](schema-design.md#field-options) remedy is the supported way to hold a runtime object such as an open connection or a compiled model beside the configured values.


## Dataclass invariants

Custom checks join the report through two hooks on any dataclass in the tree:

- **`__post_init__`**: a raised `ValueError` or `TypeError` becomes one issue at the dataclass's path.
- **`__validate__`**: returns an iterable of messages; each message becomes its own issue at that path, so one hook can report several independent problems.

Per node the order is `__init__` (which runs `__post_init__` as its final step) → the [`init=False`](schema-design.md#field-options) completeness check → `__validate__`. The completeness check sits between them so `__post_init__` has populated the runtime fields first, and `__validate__` runs only on a fully populated instance, so a node whose `init=False` field still awaits a value contributes that completeness issue in its place.

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


## Aggregation boundaries

Collection is exhaustive across siblings: every field, sequence element, and mapping entry is visited even after earlier ones fail. Two boundaries shape what appears in a single report:

- A dataclass node whose fields have issues stays unconstructed for that load, so its `__post_init__` / `__validate__` hooks run on the next load, once the field issues are fixed. Fixing a config can therefore surface a second, deeper round of invariant messages.
- A multi-member union tries each member privately and, when every member fails, reports one union mismatch for the field. A single-type optional (`T | None`) coerces straight through `T`, preserving detailed child paths and running hooks exactly once.


## Handling errors in a training CLI

Catch `ConfigError` at startup and fail fast, before allocating accelerators:

```python
def main() -> int:
    try:
        config = TrainingConfig.from_file(sys.argv[1])
    except ConfigError as err:
        print(err, file=sys.stderr)
        return 2
    run(config)
    return 0
```

The rendered error already includes the context, the issue count, and one line per issue, so printing the exception gives users an actionable report.


## Related pages

- [Schema design](schema-design.md) for structural fixes and hook placement.
- [Types and coercion](types-and-coercion.md) for the conversion rules behind type-mismatch issues.
- [Files, formats, and run identity](files-and-identity.md#document-and-read-rules) for parse and read failures.
- [API reference](api-reference.md#error-types) for exact signatures.


---

[Previous: Types and coercion](types-and-coercion.md) | [Home](README.md) | [Next: Files, formats, and run identity](files-and-identity.md)
