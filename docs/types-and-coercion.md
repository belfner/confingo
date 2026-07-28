[Documentation home](README.md)

# Types and coercion

This is the authoritative reference for the type boundary: which annotations a schema may use, which input shapes each accepts during `from_dict`, and which serialized form each produces under `to_dict`.

Two kinds of problems come out of this page's rules. An input outside an accepted shape produces a collected issue at that value's path. An annotation outside the accepted set produces a schema error before construction begins.


## Choose an annotation

Start from the value a field holds and read down to the rules that govern it:

| To configure | Annotation | Details |
| --- | --- | --- |
| a single value | `bool`, `int`, `float`, `str`, `Path`, `datetime`, `date`, `time` | [Scalars](#scalars) |
| one of a fixed set | `Literal[...]`, or an `Enum` subclass | [Enums and literals](#enums-and-literals) |
| a nullable value | `T \| None` | [Unions and optionals](#unions-and-optionals) |
| a first-match choice of types | `X \| Y` in declaration order, with a numeric pair settled by the value's own class | [Unions and optionals](#unions-and-optionals) |
| a nested section | a dataclass type | [Nested dataclasses](#nested-dataclasses); builds implicitly from its own leaves |
| an ordered collection | `list[T]`, `tuple[T, ...]`, `tuple[X, Y]`, `Sequence[T]` | [Sequences and tuples](#sequences-and-tuples) |
| a unique collection | `set[T]`, `frozenset[T]` | [Sequences and tuples](#sequences-and-tuples) |
| a keyed collection | `dict[str, T]`, `Mapping[str, T]` | [Mappings](#mappings); keys are strings |
| free-form data | `ConfigValue` | [open data](#open-data) |
| a numeric array | `np.ndarray`, `npt.NDArray[...]`, `torch.Tensor` | [Arrays and tensors](arrays-and-tensors.md); the backend is imported by the application |
| a value with a default fallback | `field(default=...)`, `field(default_factory=...)` | [leaf defaults and precedence](schema-design.md#leaf-defaults-and-precedence) |

The sections below are the authoritative rules for each row.


## How coercion works

- `from_dict` moves each supplied value toward its field's annotation, collecting an issue when the value's shape is outside what the annotation accepts.
- Defaults are validated against their annotation and their plain form, then used exactly as authored. Coercion applies to supplied values alone, so a default has to be written in the type its annotation names. See [defaults and precedence](schema-design.md#leaf-defaults-and-precedence).
- `to_dict` converts the built tree back to plain data: dicts, lists, strings, numbers, booleans, and `None`.

The everyday conversions are the ones you would guess: a JSON string on a `Path` field becomes a `Path`, a JSON array on a `tuple[int, ...]` field becomes a tuple of ints, and a nested object on a section field becomes that section. The sections below give the exact rule for each annotation, including the cases where confingo deliberately declines a conversion another library might make. For array and tensor fields, [Arrays and tensors](arrays-and-tensors.md) is the authoritative page.


## Scalars

| Annotation | Accepted inputs | Runtime value | Serialized form |
| --- | --- | --- | --- |
| `bool` | `bool` | `bool` | `true` / `false` |
| `int` | `int`; `float` with an integral value (`2e6` -> `2_000_000`) | `int` | number |
| `float` | `float`; `int` (converted, result must be finite) | `float` | number |
| `str` | `str` | `str` | string |
| `Path` | `str`; `Path` | `Path` | string (`str(path)`) |
| `datetime` | `datetime`; ISO 8601 `str` | `datetime` | ISO 8601 string |
| `date` | `date`; ISO 8601 `str` | `date` | ISO 8601 string |
| `time` | `time`; ISO 8601 `str` | `time` | ISO 8601 string |
| `None` (as `NoneType` or in unions) | `None` | `None` | `null` |

Booleans are exact. `bool` fields accept `bool` values only, and `int` / `float` fields report `bool` inputs as type mismatches, keeping `flag: bool` and `count: int` fully separate even though Python's `bool` subclasses `int`.

Strings are exact in both directions. `str` fields accept `str` only, and numeric fields report strings such as `"5"` as type mismatches: a config file states numbers as numbers.

```python
from_dict(DataConfig, {"dataset_path": "d", "batch_size": 2e6})
# batch_size == 2_000_000; the integral float landed on an int field
```


## Enums and literals

Enum members must carry primitive values whose runtime type is exactly `str`, `int`, or `bool`. A load reads one of those three from a file and hands it to the member lookup, and exact typing is what keeps that lookup single-valued: two members whose values differ only by a subclass of one of those three write the same plain form, so preflight names them and asks for exact values instead. An input matches an enum field by member value first, then by member name:

```python
class Optimizer(Enum):
    ADAMW = "adamw"
    SGD = "sgd"

# both inputs resolve to Optimizer.ADAMW:
#   "adamw"  (by value)
#   "ADAMW"  (by member name)
```

An enum leaves the member lookup to `EnumType`, which resolves a member's own value before anything else. A class whose metaclass binds `__call__` decides for itself what a value reaches, so a member's value could rebuild as a different member, and preflight reports it naming the metaclass. A `_missing_` hook is unaffected and stays the way to map spellings outside the member values, since a lookup reaches it only after those values miss.

`to_dict` serializes an enum as its `.value`. Error messages for a failed match list the valid values.

`Literal[...]` options must be primitives or `None`, and matching compares value and exact runtime type together: `Literal["cpu", "cuda"]` accepts exactly those strings, and `Literal[1, 2]` reports `True` as a mismatch even though `True == 1` in Python.


## Nested dataclasses

A mapping supplied for a dataclass-annotated field recursively constructs that section, with issues collected under the section's dotted path.

Dataclasses held in containers (`list[StageConfig]`, `dict[str, DatasetConfig]`) construct the same way, with the index or key in the path (`stages.0.name`).

A bare-annotated dataclass field defaults to an implicit build from an empty mapping, so its required leaves are reported at their nested paths. See [implicit sections](schema-design.md#implicit-sections-and-leaf-level-requirements).

The annotation drives construction, so a section annotated with a [config node](schema-design.md#config-nodes) class rebuilds as that class, in a direct field, in a container, or as a union member. Construction runs through the same recursion that builds a plain dataclass section, so each section's `__post_init__` and `__validate__` run once for the value that is kept.


## Sequences and tuples

A sequence-container annotation whose positions carry one shared meaning accepts a sequence, a `set`, or a `frozenset` as input. A fixed-arity `tuple[X, Y]` gives each position its own meaning, so it takes an ordered sequence:

| Annotation | Accepted inputs | Runtime value | Serialized form |
| --- | --- | --- | --- |
| `list[T]` | sequence/set of `T`-coercible items | `list` | array |
| `tuple[T, ...]` | sequence/set, each item `T`-coercible | `tuple` | array |
| `tuple[X, Y]` | ordered sequence of exactly that arity | `tuple` | array |
| `tuple[()]` / `typing.Tuple[()]` | empty sequence/set | `()` | `[]` |
| `set[T]` / `frozenset[T]` | sequence/set of `T`-coercible items, `T` a scalar or a tuple/frozenset whose arguments recursively satisfy the same rule | `set` / `frozenset` | array, deterministically ordered |
| `Sequence[T]` | sequence/set of `T`-coercible items | `list` | array |
| `list[ConfigValue]` / `tuple[ConfigValue, ...]` | sequence/set of [open data](#open-data) | matching container | array |

A few rules apply across every row of the table:

- Elements coerce individually, and element issues carry their index in the path (`hidden_widths.1`).
- A `set` or `frozenset` supplied to a sequence annotation is read in its own iteration order, which varies from run to run, so use one where the order carries no meaning. A fixed-arity `tuple[X, Y]` declines a set outright: each position carries its own meaning, and the message names a list as the form to write.
- String and bytes inputs follow scalar handling: a `list[str]` field reports a bare `"abc"` as one type mismatch.
- A `str`, `int`, or temporal value handed straight to `from_dict` already satisfies its annotation, so a subclass instance is kept as supplied and reaches a set with its own hashing and equality. Building the set runs both, and a failure in either reports as `cannot build set[str]: unhashable type: 'NoHashStr'` alongside any sibling issues. A subclass whose hashing and equality both succeed builds normally, and a `float` or `Path` value is rebuilt through its own type. A `MemoryError` or `SystemError` travels to the caller, since it describes the interpreter rather than the config.
- A `set[T]` / `frozenset[T]` element annotation is admitted when the plain data a file carries rebuilds hashable under it, which `T` settles on its own: a scalar, or a `tuple` / `frozenset` whose own arguments recursively satisfy that same rule. `set[str]`, `set[Color]`, `set[tuple[str, int]]`, `set[frozenset[str]]`, and deeper shapes such as `set[tuple[tuple[int, str], frozenset[int]]]` all qualify. An `Enum` qualifies when it leaves hashing to the implementation it inherits, and it carries the [member value and lookup requirements](#enums-and-literals) every enum annotation carries. Anything else is reported at preflight naming the annotation as written, with a scalar element, a tuple of scalars, or a list as the remedy.
- A `set[T]` / `frozenset[T]` element names one type. A union puts two readers behind one plain form, and the load hands that form to the first member accepting it, so a set of `str | Path` writes `"a"` for both `"a"` and `Path("a")` and rebuilds one element where the file carried two. `T | None` is the exception, and the only one, because `null` is a plain form no other reader accepts. The rule reaches into the `tuple` and `frozenset` positions an element can nest, so `set[tuple[int, str | Path]]` is reported and `set[tuple[int, str | None]]` is admitted. Every other position in a schema still takes a union.
- A `set[T]` / `frozenset[T]` whose element type carries a config section keeps a remedy of its own, since [config objects are unhashable](equality-and-hashing.md#config-objects-are-unhashable). Hold sections in a `list` or `tuple`, and key them by `config_hash(section)` when uniqueness matters.
- A container annotation carries the type arguments the engine builds from: one element type for a sequence or set, a key and a value type for a mapping, and `...` only as the variadic marker of `tuple[T, ...]`. Anything else is reported at preflight naming the annotation as written.
- A container field uses its default when it has one, and a required container needs a supplied value. An intentionally empty container is authored as `field(default_factory=list)`.

Typical ML shapes: `hidden_widths: tuple[int, ...]` for layer sizes, `tuple[int, int]` for a fixed `(warmup_steps, total_steps)` schedule pair, and `metrics: set[str]` for tracked metrics.


## Mappings

`dict[str, T]` and `Mapping[str, T]` accept mappings with `str` keys and construct a concrete `dict`. `dict[str, ConfigValue]` is the form for values whose shape the schema leaves to the file.

- Every key is checked as a string at load time, which catches YAML documents whose keys parsed as numbers.
- A document that repeats a key carries the value written last, which is what the JSON and YAML parsers hand confingo. Detecting the repetition belongs to the parser: supply your own loader and hand its result to `from_dict` when a repeated key should be reported.
- Split definitions like `datasets: dict[str, DatasetConfig]` construct each value against the annotated section type, with the key in the issue path (`datasets.train.path`).
- A mapping field uses its default when it has one, and a required mapping needs a supplied value; `field(default_factory=dict)` authors an intentionally empty mapping.


## Unions and optionals

Union members are tried in declaration order and the first member that coerces cleanly wins, so order unions deliberately: `int | str` sends `5` to `int` and `"5"` to `str`, while `OptimizerConfig | SchedulerConfig` tries `OptimizerConfig` first for every mapping.

One rule runs ahead of that order, and it applies to a union naming **two distinct classes** from `bool`, `int`, and `float`. A file states each of the three as itself, so a value's own class names which one the file carried, and the member naming that class is tried first. This is what makes `int | float` and `float | int` both round-trip: an `int` field reads an integral float so that `1e6` is accepted, which by declaration order alone would send a `1.0` to the `int` member and read it back as `1`. Once the rule applies, the member naming the carried class outranks every other member, an earlier member of another kind included, since that is what settles the conversion between the two numeric classes.

One numeric class keeps the declared order, however many members name it. `Number | int` reads `1` as the declared `Number`, and so does `Number | Annotated[int, "a"] | Annotated[int, "b"]`, where both `Annotated` members name the one class `int`. Where one plain form fits two members the declared order also still answers, as it does for `Path | str`: both are written as a string, so the form names neither.

A `float` field holds the float its annotation names. Assigning an `int` to one in Python is accepted by a type checker, since the numeric tower promotes it, and the value stays the `int` it was written as: `to_dict` renders `1` and the next load reads it back as `1.0`, which the fingerprint tokenizes differently. Write `1.0` where the field names a float, and a value that came from a file already carries it.

When every member fails, the field reports a summary naming the whole union and the member that came closest, followed by that one member's own issues at their own paths:

```text
config has 2 issues:
  - optimizer: expected AdamW | SGD; best match SGD failed with 1 issue
  - optimizer.lr: expected float, got str
```

"Closest" is the member whose attempt collected the fewest issues, and an equal count goes to the first declared member. That tie is what two structurally identical variants produce when only their discriminator `Literal` differs and the file carries a typo: each fails once, so the first declared member supplies the detail while the summary still names the whole union.

`T | None` is special-cased: `None` is accepted directly, and any other input coerces straight through `T`, preserving nested issue paths and running the section's construction hooks exactly once.


## Open data

Two annotations name plain data whose shape the schema leaves to the file. Both come from the package root:

```python
from confingo import ConfigScalar, ConfigValue
```

`ConfigValue` is the whole plain-data domain: `bool`, `int`, `float`, `str`, `None`, lists of those, and `str`-keyed mappings of those, nested up to 64 levels. `ConfigScalar` is the leaf half of it, for a field holding one value.

```python
@dataclass
class Experiment:
    notes: ConfigValue = None
    marker: ConfigScalar = None
```

Both are ordinary PEP 695 aliases, so a type checker reads them structurally and reports a value outside the domain at the assignment. confingo checks whatever the file supplied against the same domain.

A tuple supplied to a `ConfigValue` field rebuilds as a list, so the value round-trips to the plain form it was written as. Every leaf is checked for the JSON representation the fingerprint depends on, so a non-finite float is reported at its own path.

A value outside the domain is reported where it enters rather than after it has been written. A `Path`, a temporal value, a set, a config object, an array, and a numpy scalar each carry a shape the annotation does not name, and each is reported with the shapes that are inside it. Name the type instead: an [array annotation](#arrays-and-tensors) for an array, a [section](schema-design.md#config-nodes) for a config object, `Path` for a path.

A mapping key that is not a `str` is reported by its type, so a key whose `__str__` raises is named rather than run. Sibling entries still report together.

An authored default carries the same domain, checked as written rather than coerced. A mutable open-data default is authored as a factory annotated with its return type, which is what gives the literal an expected type for a checker to read it against:

```python
def default_notes() -> ConfigValue:
    return {"owner": "ml-platform", "tags": []}

@dataclass
class Experiment:
    notes: ConfigValue = field(default_factory=default_notes)
```


## Arrays and tensors

NumPy arrays and PyTorch tensors are field types whenever the host application has already imported the backend; the integration reads that application-loaded backend, while confingo's base runtime stays the standard-library core plus PyYAML-backed YAML I/O. A bare annotation (`np.ndarray`, `torch.Tensor`) rebuilds the array with a value-stable inferred dtype; a concrete annotation (`npt.NDArray[np.float32]`, `Annotated[torch.Tensor, torch.float32]`) pins the dtype, and a fixed-arity shape tuple pins the dimensionality. Values serialize as plain JSON (a scalar for a 0-d value, nested lists otherwise) and rebuild against the annotation on the next load.

The full contract lives in [Arrays and tensors](arrays-and-tensors.md): backend activation, the annotation table, accepted inputs, dtype normalization, serialization state, finiteness and size limits, and how arrays take part in round trips, equality, and hashing.


## Finite numbers and temporal exactness

JSON represents only finite numbers, so a non-finite float anywhere in supplied data produces an issue, and `to_dict` raises `ConfigError` when asked to serialize one. An integer too large for `float` conversion produces a collected overflow issue.

On temporal fields the subtype ordering is enforced: a `datetime` value supplied for a `date` field is reported as a type issue. The two types stay distinct through every load.


## Accepted schema boundary

The accepted annotation set is explicit and closed:

| Category | Accepted annotations |
| --- | --- |
| Scalars | `bool`, `int`, `float`, `str`, `Path`, `datetime`, `date`, `time`, `None` |
| Enums / literals | `Enum` subclasses with primitive member values; `Literal` with primitive or `None` options |
| Open data | `ConfigValue` for plain data of any shape, `ConfigScalar` for one plain leaf; see [open data](#open-data) |
| Containers | `list[T]`, `tuple[T, ...]`, `dict[str, T]`, `Sequence[T]`, `Mapping[str, T]`, each naming its element type; `set[T]` and `frozenset[T]` with an element that [rebuilds hashable](#sequences-and-tuples) |
| Structure | dataclasses (each `init=True` field boundary-checked; an `init=False` field holds runtime state and is exempt), unions of accepted members, `Optional[T]` |
| Arrays | `np.ndarray` forms and `torch.Tensor` forms from [arrays and tensors](#arrays-and-tensors), when the backend is loaded |
| Wrappers | `Annotated[T, ...]`, treated as `T`; on tensors, a `torch.dtype` entry pins the dtype and a fixed-arity all-`int` shape tuple such as `tuple[int, int]` enforces dimensionality, each usable alone or together; every other metadata entry passes through as ordinary annotation metadata |

An `init=True` annotation outside this set produces a `ConfigError` during schema preflight, even for a field that would have used its default. Rejected shapes include:

- `Decimal`, `TypedDict`, `Iterable[T]`, and `NewType`
- `Any`, which leaves the values it holds undescribed; write `ConfigValue` for plain data of any shape, or name the type the field holds
- an argument-free container (`list`, `dict`, `Sequence`, `typing.List`, `set[()]`, and every other spelling of the same), which names no element type; write the parameterized form the message gives. `tuple[()]` is the one exception: it names a tuple holding nothing, which is a shape confingo builds
- mappings with keys other than `str`, including `dict[int, T]`
- enums with object values, and enum-backed `Literal` options
- a schema class that owns type parameters, whether written `class Config[T]` or with the legacy `Generic[T]` / `Protocol[T]` spelling, and whether the parameters are its own, a base's, or its metaclass's. A config file carries concrete values, so a load builds the type an annotation names and a parameter names none; the class that declares them is the one reported, with the concrete types to write in their place. A `TypeVar` reached *through* an annotation is unaffected, which is how numpy spells `npt.NDArray`
- a subclass of `Path`, `datetime`, `date`, or `time`. A load builds the base class, so the annotation would name a type the value does not carry; annotate the base and derive the subclass in an [`init=False`](schema-design.md#field-options) field

An [`init=False`](schema-design.md#field-options) field holds runtime state populated in `__post_init__` and is exempt from this boundary; its annotation need only resolve. The preflight runs before any value is coerced. See [validation phases](validation-and-errors.md#two-validation-phases).


## Working near the boundary

For a value whose natural type sits outside the accepted set, store a supported primitive and validate its syntax in the schema:

```python
from decimal import Decimal, InvalidOperation


@dataclass
class BudgetConfig:
    max_cost: str  # decimal string, e.g. "12.50"

    def __post_init__(self) -> None:
        try:
            Decimal(self.max_cost)
        except InvalidOperation as exc:
            raise ValueError(f"max_cost must be a decimal string: {self.max_cost!r}") from exc

    @property
    def max_cost_decimal(self) -> Decimal:
        return Decimal(self.max_cost)
```

Raising `ValueError` from `__post_init__` routes the failure into the collect-all report. See [dataclass invariants](validation-and-errors.md#dataclass-invariants).

The stored field stays serializable and hashable, round trips through every format, and the application reads the parsed value through the property.


---

Exact reference: [Schema design](schema-design.md) | [Validation and errors](validation-and-errors.md) | [Equality and hashing](equality-and-hashing.md) | [API reference](api-reference.md) | [Documentation home](README.md)
