"""Contracts for the cfg accessor, user callbacks, scalars, and array export.

Four boundaries are pinned here. The ``cfg`` accessor is a data descriptor, so
the one reserved name cannot be shadowed away. Every user callback confingo runs
-- ``__post_init__``, ``__validate__``, and a selected ``default_factory`` --
reports what it raises as an issue beside its siblings, and ``__validate__``'s
return is read against its contract before it is consumed. A scalar annotation
names the class a load builds. And a numpy array reaching serialization carries a
plain form or says why it does not.
"""

from __future__ import annotations

import datetime as dt
import weakref
from dataclasses import (
    dataclass,
    field,
    make_dataclass,
)
from pathlib import Path
from typing import Any

import pytest

from confingo import (
    ConfigError,
    ConfigNode,
)
from confingo.functional import (
    config_equal,
    config_hash,
    from_dict,
    load_json,
    save_json,
    to_dict,
)


def issues_of(config_cls: type, data: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    """Build a config and return every issue as a (path, message) pair."""
    with pytest.raises(ConfigError) as info:
        from_dict(config_cls, {} if data is None else data)
    return [(issue.path, issue.message) for issue in info.value.issues]


# --- the cfg accessor is a data descriptor ------------------------------------


@dataclass
class Node(ConfigNode):
    lr: float = 1e-3


def test_assigning_over_the_accessor_names_what_it_carries():
    node = Node()
    with pytest.raises(AttributeError) as info:
        node.cfg = "shadowed"
    assert "cfg carries Node's config operations and cannot be assigned" in str(info.value)
    assert node.cfg.to_dict() == {"lr": 1e-3}


def test_deleting_the_accessor_names_what_it_carries():
    node = Node()
    with pytest.raises(AttributeError) as info:
        del node.cfg
    assert "cfg carries Node's config operations and cannot be deleted" in str(info.value)
    assert node.cfg.to_dict() == {"lr": 1e-3}


# --- the user-callback boundary -----------------------------------------------


@dataclass
class ValidateReturnsStr:
    lr: float = 2.0

    def __validate__(self) -> str:
        return "lr must be <= 1.0"


@dataclass
class ValidateReturnsNone:
    lr: float = 2.0

    def __validate__(self) -> None:
        return None


@dataclass
class ValidateRaises:
    lr: float = 2.0

    def __validate__(self) -> list[str]:
        raise ValueError("boom")


@dataclass
class PostInitRaises:
    lr: float = 2.0

    def __post_init__(self) -> None:
        raise RuntimeError("post boom")


@dataclass
class ValidateReports:
    lr: float = 2.0

    def __validate__(self) -> list[str]:
        return ["lr must be <= 1.0", "lr must be positive"]


HOOK_REMEDY = (
    "__validate__ returns an iterable of messages; return a list of strings, or an empty list when the config is valid"
)


def test_a_str_return_is_read_as_the_contract_slip_it_is():
    # A str satisfies "an iterable" while iterating to one issue per character.
    assert issues_of(ValidateReturnsStr) == [("", f"ValidateReturnsStr.__validate__ returned str; {HOOK_REMEDY}")]


def test_a_none_return_is_read_as_the_contract_slip_it_is():
    assert issues_of(ValidateReturnsNone) == [("", f"ValidateReturnsNone.__validate__ returned None; {HOOK_REMEDY}")]


def test_a_raising_validate_hook_becomes_one_issue():
    assert issues_of(ValidateRaises) == [("", "validating ValidateRaises raised ValueError: boom")]


def test_a_raising_post_init_becomes_one_issue():
    assert issues_of(PostInitRaises) == [("", "constructing PostInitRaises raised RuntimeError: post boom")]


def test_every_reported_message_still_becomes_its_own_issue():
    assert issues_of(ValidateReports) == [("", "lr must be <= 1.0"), ("", "lr must be positive")]


def raising_factory() -> list[int]:
    """Produce nothing; raise the way an author's factory can."""
    raise RuntimeError("factory boom")


@dataclass
class FactoryRaises:
    items: list[int] = field(default_factory=raising_factory)
    count: int = 1


def test_a_raising_factory_reports_beside_its_siblings():
    assert issues_of(FactoryRaises, {"count": "wrong"}) == [
        ("items", "default_factory raised RuntimeError: factory boom"),
        ("count", "expected int, got str"),
    ]


# --- a scalar annotation names the class a load builds ------------------------


class SpecialDate(dt.date):
    """A date subclass carrying behavior of its own."""


class SpecialPath(Path):
    """A Path subclass carrying behavior of its own."""


@dataclass
class HasSpecialDate:
    when: SpecialDate = None  # pyrefly: ignore[bad-assignment]


@dataclass
class HasSpecialPath:
    where: SpecialPath = None  # pyrefly: ignore[bad-assignment]


@pytest.mark.parametrize(
    ("config_cls", "path", "subclass", "base"),
    [
        (HasSpecialDate, "when", "SpecialDate", "date"),
        (HasSpecialPath, "where", "SpecialPath", "Path"),
    ],
    ids=["date", "path"],
)
def test_a_scalar_subclass_annotation_names_the_base_a_load_builds(
    config_cls: type, path: str, subclass: str, base: str
):
    assert issues_of(config_cls) == [
        (
            path,
            f"{subclass} is a {base} subclass, and a load builds {base} itself; "
            f"annotate the field {base}, and derive the subclass in an init=False field",
        )
    ]


@dataclass
class Temporal:
    day: dt.date = dt.date(2021, 5, 5)
    moment: dt.datetime = dt.datetime(2021, 5, 5, 12)
    clock: dt.time = dt.time(12)
    where: Path = Path("y")


def test_the_documented_base_types_build_the_class_they_name():
    built = from_dict(
        Temporal, {"day": "2020-01-02", "moment": "2020-01-02T03:04:05", "clock": "03:04:05", "where": "z"}
    )
    assert type(built.day) is dt.date
    assert type(built.moment) is dt.datetime
    assert type(built.clock) is dt.time
    assert isinstance(built.where, Path)


# --- numpy arrays carry a plain form or say why they do not -------------------


np = pytest.importorskip("numpy")
npt = pytest.importorskip("numpy.typing")


@dataclass
class Weights:
    w: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(1))


MASK_REMEDY = (
    "got a masked array, whose mask has no plain form; "
    "pass np.asarray(value) after deciding what each masked element holds"
)


def test_a_masked_array_is_declined_where_it_enters():
    masked = np.ma.masked_array([1.0, 2.0], mask=[False, True])
    assert issues_of(Weights, {"w": masked}) == [("w", f"only plain numpy arrays are supported; {MASK_REMEDY}")]


def test_a_masked_array_is_declined_on_the_way_out():
    masked = np.ma.masked_array([1.0, 2.0], mask=[False, True])
    with pytest.raises(ConfigError) as info:
        to_dict(Weights(w=masked))
    assert info.value.issues[0].message == f"only plain numpy arrays can be serialized; {MASK_REMEDY}"


def test_a_subclass_whose_tolist_raises_reports_instead_of_escaping():
    class Exploding(np.ndarray):
        """An ndarray subclass whose tolist raises."""

        def tolist(self) -> list[Any]:
            raise RuntimeError("tolist exploded")

    with pytest.raises(ConfigError) as info:
        to_dict(Weights(w=np.array([1.0, 2.0]).view(Exploding)))
    assert info.value.issues[0].message == "cannot convert numpy array to its plain form: tolist exploded"


# --- the callback boundary holds against a hostile class ----------------------


class RaisingMeta(type):
    """A metaclass whose name read raises."""

    # Replacing the plain attribute with a property is what makes reading the name
    # run code, which is the shape under test.
    @property
    # pyrefly: ignore[bad-override, missing-override-decorator]
    def __name__(cls) -> str:
        """Fail, so a reporter that reads it is visible.

        Raises:
          RuntimeError: Always.
        """
        raise RuntimeError("name read boom")


class UnnameableError(Exception, metaclass=RaisingMeta):
    """An exception whose class cannot be named."""


class UnrenderableText:
    """A value whose text cannot be rendered."""

    def __str__(self) -> str:
        """Fail, so a reporter that renders it is visible.

        Raises:
          RuntimeError: Always.
        """
        raise RuntimeError("str boom")


@dataclass
class HookLookupRaises:
    lr: float = 1.0

    def __getattribute__(self, name: str) -> Any:
        """Fail for the hook name alone, leaving every other read intact.

        Args:
          name (str): The attribute being read.

        Returns:
          Any: The attribute's value.

        Raises:
          RuntimeError: When the hook name is read.
        """
        if name == "__validate__":
            raise RuntimeError("hook lookup boom")
        return object.__getattribute__(self, name)


@dataclass
class HookRaisesUnnameable:
    lr: float = 1.0

    def __validate__(self) -> list[str]:
        """Fail with an exception whose class cannot be named.

        Raises:
          UnnameableError: Always.
        """
        raise UnnameableError("inner")


@dataclass
class HookReturnsUnrenderable:
    lr: float = 1.0

    def __validate__(self) -> Any:
        """Return a value outside the contract whose text cannot be rendered.

        Returns:
          Any: The unrenderable value.
        """
        return UnrenderableText()


def test_a_raising_hook_lookup_is_reported_rather_than_escaping():
    assert issues_of(HookLookupRaises) == [("", "validating HookLookupRaises raised RuntimeError: hook lookup boom")]


def test_an_exception_whose_class_cannot_be_named_still_reports():
    assert issues_of(HookRaisesUnnameable) == [
        ("", "validating HookRaisesUnnameable raised an exception whose class could not be named: inner")
    ]


def test_a_return_whose_text_cannot_be_rendered_still_reports():
    found = issues_of(HookReturnsUnrenderable)
    assert len(found) == 1, found
    assert "returned UnrenderableText" in found[0][1]


# --- an array's plain form is bounded and checked -----------------------------


def test_an_array_whose_rank_passes_the_budget_is_declined():
    with pytest.raises(ConfigError) as info:
        to_dict(Weights(w=np.zeros((1,) * 64)))
    assert "nesting reaches the 64 level limit for plain data" in info.value.issues[0].message


def test_an_array_whose_rank_fits_the_budget_serializes():
    assert to_dict(Weights(w=np.zeros((1,) * 62)))["w"] is not None


def test_a_subclass_plain_form_is_checked_the_way_a_supplied_value_is():
    class NotPlain(np.ndarray):
        """An ndarray subclass whose tolist hands back a value of its own."""

        def tolist(self) -> Any:
            """Return something other than plain lists.

            Returns:
              Any: A Path, which the marshal walk renders as the text it is.
            """
            return Path("not-plain")

    # to_dict's contract is plain data, so whatever the subclass produced travels
    # through the same conversion any supplied value does.
    assert to_dict(Weights(w=np.array([1.0]).view(NotPlain))) == {"w": "not-plain"}


# --- the minor findings -------------------------------------------------------


@pytest.mark.parametrize(
    "length", [0, -1, 65, 2.5, True, "12"], ids=["zero", "negative", "over", "float", "bool", "str"]
)
def test_a_digest_length_outside_the_documented_range_is_rejected(length: Any):
    with pytest.raises(ConfigError) as info:
        config_hash(Node(), length=length)
    assert "config_hash length must be an int from 1 to 64" in info.value.issues[0].message


@pytest.mark.parametrize("length", [1, 12, 64], ids=["one", "default", "full"])
def test_a_digest_length_inside_the_range_returns_that_many_characters(length: int):
    assert len(config_hash(Node(), length=length)) == length


@dataclass
class Span:
    window: tuple[int, int] = (0, 1)


def test_a_fixed_arity_tuple_declines_input_whose_order_is_not_expressed():
    assert issues_of(Span, {"window": {1, 2}}) == [
        (
            "window",
            "expected an ordered sequence for tuple[int, int], got set; each position of this tuple "
            "carries its own meaning, so write the items in a list",
        )
    ]


def test_a_variadic_tuple_still_accepts_a_set():
    @dataclass
    class Tags:
        tags: tuple[int, ...] = ()

    assert set(from_dict(Tags, {"tags": {1, 2}}).tags) == {1, 2}


@pytest.mark.parametrize("recursive", [False, True], ids=["plain", "self-referential"])
def test_a_schema_takes_its_cached_work_with_it(recursive: bool):
    # Every per-class result is stored on the class it describes, so nothing this
    # module roots outlives the class, a schema naming itself included.
    import gc  # noqa: PLC0415

    from confingo._schema import _classify_hint_by_id  # noqa: PLC0415

    if recursive:
        holder = make_dataclass("Transient", [("child", "Transient | None", field(default=None))])
        holder.__module__ = __name__
        globals()["Transient"] = holder
    else:
        holder = make_dataclass("Transient", [("lr", float, field(default=1.0))])
    from_dict(holder, {})
    reference = weakref.ref(holder)
    del holder
    globals().pop("Transient", None)
    _classify_hint_by_id.cache_clear()
    gc.collect()
    assert reference() is None


def test_a_saved_file_carries_the_whole_document(tmp_path: Path):
    # The write reaches the disk before the rename, so the destination holds the
    # complete text rather than an empty file.
    destination = save_json(Node(lr=0.5), tmp_path / "nested" / "run.json")
    assert destination.read_text(encoding="utf-8").strip().endswith("}")
    assert load_json(Node, destination) == Node(lr=0.5)


def test_the_element_cap_names_where_the_data_belongs():
    oversized = np.zeros(1_000_001, dtype=np.uint8)
    assert issues_of(Weights, {"w": oversized}) == [
        (
            "w",
            "array has 1000001 elements; a config field holds at most 1000000, since every element is "
            "written into the config file; store the data in its own file and configure that file's path",
        )
    ]


# --- an array spends the same budget its plain form costs ---------------------


ARRAY_DEPTHS: list[tuple[str, tuple[int, ...], bool]] = [
    ("rank-62", (1,) * 62, True),
    ("rank-63", (1,) * 63, True),
    ("rank-64", (1,) * 64, False),
    ("empty-first-axis", (0, *(1,) * 63), True),
    ("empty-second-axis", (1, 0, *(1,) * 62), True),
]


@pytest.mark.parametrize(
    ("shape", "admitted"),
    [(shape, admitted) for _name, shape, admitted in ARRAY_DEPTHS],
    ids=[name for name, _shape, _admitted in ARRAY_DEPTHS],
)
def test_an_array_costs_the_levels_its_plain_form_writes(shape: tuple[int, ...], admitted: bool):
    # An empty axis ends the encoding, so a rank-64 array whose first axis is
    # empty writes one level and is admitted where a full one is declined.
    held = Weights(w=np.zeros(shape))
    if not admitted:
        for operation in (to_dict, config_hash, lambda config: config_equal(config, config)):
            with pytest.raises(ConfigError):
                operation(held)
        return
    assert to_dict(held) is not None
    assert config_hash(held)
    assert config_equal(held, Weights(w=np.zeros(shape)))


def test_an_array_too_deep_to_write_is_declined_from_either_side_of_a_comparison():
    # The value with no plain form is reported whichever operand carries it, so
    # the two orders of one comparison answer the same way.
    from_dict(Weights, {})  # installs canonical equality, so == evaluates the same relation
    over = Weights(w=np.zeros((1,) * 64))
    under = Weights(w=np.zeros(1))
    for left, right in ((over, under), (under, over)):
        with pytest.raises(ConfigError, match="nesting reaches the 64 level limit"):
            config_equal(left, right)
        with pytest.raises(ConfigError, match="nesting reaches the 64 level limit"):
            assert left == right


def test_an_array_too_deep_to_write_is_declined_at_load():
    deep: Any = 1.0
    for _ in range(64):
        deep = [deep]
    assert issues_of(Weights, {"w": deep})[0][1].startswith("nesting reaches the 64 level limit")


# --- a direct default spends the budget where it is selected ------------------


def nested_tuple(levels: int) -> Any:
    """Build a value of exactly this many nested tuples around one scalar."""
    value: Any = 1
    for _ in range(levels):
        value = (value,)
    return value


def tuple_hint(levels: int) -> Any:
    """Build the annotation matching a value of that many nested tuples."""
    hint: Any = int
    for _ in range(levels):
        hint = tuple[hint]
    return hint


@pytest.mark.parametrize("levels", [62, 63], ids=["inside", "at-the-edge"])
def test_a_direct_default_at_a_root_field_agrees_with_every_walk(levels: int):
    child = make_dataclass("Child", [("payload", tuple_hint(levels), field(default=nested_tuple(levels)))])
    built = from_dict(child, {})
    assert to_dict(built) is not None
    assert config_hash(built)


def test_a_direct_default_is_declined_where_its_runtime_position_passes_the_budget():
    # The same default is admitted on a root field and declined one level deeper,
    # since the schema path cannot name where a container places the section.
    child = make_dataclass("Child", [("payload", tuple_hint(63), field(default=nested_tuple(63)))])
    root = make_dataclass("Root", [("children", list[child], field(default_factory=list))])
    assert from_dict(child, {}) is not None
    assert issues_of(root, {"children": [{}]})[0][1].endswith("or name the shape with a dataclass section")


def test_a_wide_shallow_default_is_not_rewalked_on_every_load():
    wide = tuple(range(50_000))
    config_cls = make_dataclass("Wide", [("xs", tuple[int, ...], field(default=wide))])
    assert from_dict(config_cls, {}).xs == wide
    assert from_dict(config_cls, {}).xs == wide
