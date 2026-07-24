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

1. **Schema preflight.** Every annotation in the tree is checked against the [accepted boundary](types-and-coercion.md#accepted-schema-boundary), including fields the input omits. Schema problems raise immediately.
2. **Construction.** The input mapping is walked against the schema, coercing values and collecting every discoverable issue before raising.

The context on the raised error tells you where the problem came from:

- File loaders attach the file path (`config file train.json`) to read, parse, preflight, and construction issues.
- An annotation that fails to resolve raises with the `config schema` context.
- Direct callers can set their own via `from_dict(cls, data, context="sweep entry 3")`.


## Built-in issue sources

- Unknown keys, with the sorted known-key list in the message.
- A key supplied for an [`init=False`](schema-design.md#field-options) field: `field is not configurable (init=False)`, since runtime fields are populated in `__post_init__`, not loaded.
- Missing required values: undefaulted fields absent from the input. Dataclass sections build implicitly, so a required value inside an omitted section is reported at its nested dotted path ([details](schema-design.md#implicit-sections-and-leaf-level-requirements)).
- A value other than a mapping supplied for a dataclass section or document root.
- Type mismatches and tuple-arity mismatches from [coercion](types-and-coercion.md).
- Enum and `Literal` values outside the declared options.
- Unhashable elements supplied for set fields.
- Non-finite floats anywhere in supplied data.
- A contradictory field declaration, `field(hash=True, compare=False)`, reported at preflight: a field in the fingerprint must participate in equality.
- An `init=False` field left unset by `__post_init__`: `init=False field was not set during __post_init__`.


## Dataclass invariants

Custom checks join the report through two hooks on any dataclass in the tree:

- **`__post_init__`**: a raised `ValueError` or `TypeError` becomes one issue at the dataclass's path.
- **`__validate__`**: returns an iterable of messages; each message becomes its own issue at that path, so one hook can report several independent problems.

Per node the order is `__init__` (which runs `__post_init__` as its final step) → the [`init=False`](schema-design.md#field-options) completeness check → `__validate__`. The completeness check sits between them so `__post_init__` has populated the runtime fields first, and `__validate__` runs only on a fully populated instance; a node with an unset `init=False` field reports that issue and skips `__validate__`.

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
