"""Contracts for value identity, authored section defaults, and schema classes.

Five boundaries are pinned here. ``config_hash`` is the value-identity operation,
so it installs the ownership contract the digest's claim rests on. An authored
section default reaches the same lifecycle report as one built from supplied
data. Two rules say what a schema class may be: a section is not also one of the
kinds a walk dispatches on, and its constructor takes the class's own fields,
which is what a hand-written ``__init__`` and a required ``InitVar`` each break.
The subscripted empty shape tuple pins a scalar array. And the array-annotation
cache classifies against the backends it stores an answer under.
"""

from __future__ import annotations

from collections.abc import (
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import (
    InitVar,
    dataclass,
    field,
)
from typing import (
    Annotated,
    Any,
    override,
)

import pytest

from confingo import (
    ConfigError,
    ConfigNode,
)
from confingo._arrays import inspect_annotation
from confingo._backend import (
    BackendSnapshot,
    capture_backend_snapshot,
)
from confingo._schema import (
    _array_match_by_id,
    array_match,
)
from confingo.functional import (
    config_equal,
    config_hash,
    from_dict,
    to_dict,
    validate_schema,
)


np = pytest.importorskip("numpy")
npt = pytest.importorskip("numpy.typing")
torch = pytest.importorskip("torch")


def issues_of(config_cls: type, data: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    """Build a class and return the issues it reported, as ``(path, message)`` pairs.

    Args:
      config_cls (type): The entry dataclass to build.
      data (dict[str, Any] | None = None): The mapping to build from, empty by default.

    Returns:
      list[tuple[str, str]]: One pair per reported issue, in discovery order.
    """
    with pytest.raises(ConfigError) as info:
        from_dict(config_cls, {} if data is None else data)
    return [(issue.path, issue.message) for issue in info.value.issues]


# --- config_hash carries the ownership contract the digest claims -------------


@dataclass(frozen=True)
class SignedZero:
    x: float = 0.0


@dataclass
class OwnEquality:
    x: int = 1

    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = None  # type: ignore[assignment]


def test_config_hash_installs_the_ownership_contract():
    # Before the digest is taken the class still carries the generated __eq__,
    # which holds signed zeros equal where the canonical JSON keeps them apart.
    positive, negative = SignedZero(0.0), SignedZero(-0.0)
    assert config_hash(positive) != config_hash(negative)
    # Taking the digest is what settles identity, so equality now reads the same
    # plain form the digest does, and the class carries the unhashable contract.
    assert (positive == negative) is False
    assert config_equal(positive, negative) is False
    with pytest.raises(TypeError, match="unhashable type"):
        hash(positive)


def test_config_hash_rejects_a_class_that_owns_its_own_equality():
    with pytest.raises(ConfigError, match="defines a custom __eq__"):
        config_hash(OwnEquality())


# --- an authored section default reaches the same lifecycle report ------------


@dataclass
class Window:
    low: int = 5
    high: int = 2

    def __validate__(self) -> list[str]:
        return ["low must not exceed high"] if self.low > self.high else []


@dataclass
class Runtime:
    seed: int = 1
    handle: str = field(init=False)


@dataclass
class FactoryRoot:
    window: Window = field(default_factory=Window)


@dataclass(frozen=True)
class FrozenWindow:
    low: int = 5
    high: int = 2

    def __validate__(self) -> list[str]:
        return ["low must not exceed high"] if self.low > self.high else []


@dataclass
class DirectRoot:
    # A frozen section is the shape a direct default can hold: dataclasses
    # decline a default whose class withdrew hashing.
    window: FrozenWindow = field(default=FrozenWindow())


@dataclass
class HeldRoot:
    windows: list[Window] = field(default_factory=lambda: [Window()])


@dataclass
class RuntimeRoot:
    inner: Runtime = field(default_factory=Runtime)


@pytest.mark.parametrize("config_cls", [FactoryRoot, DirectRoot], ids=["default_factory", "direct default"])
def test_an_authored_section_default_reports_its_own_invariants(config_cls: type):
    # The same section reaches the same report whether the input carried it or
    # the default supplied it.
    supplied = issues_of(config_cls, {"window": {}})
    assert supplied == [("window", "low must not exceed high")]
    assert issues_of(config_cls) == supplied


def test_a_section_inside_a_container_default_reports_at_its_own_path():
    assert issues_of(HeldRoot) == [("windows.0", "low must not exceed high")]


def test_an_authored_default_reports_an_unset_runtime_field():
    assert issues_of(RuntimeRoot) == [("inner.handle", "init=False field was not set during __post_init__")]


# --- a constructor confingo cannot call is named at preflight -----------------


@dataclass
class RenamedInit:
    value: int = 0

    def __init__(self, raw: int = 0) -> None:
        self.value = raw


@dataclass
class RequiredInitVar:
    raw: InitVar[int]
    value: int = field(init=False)

    def __post_init__(self, raw: int) -> None:
        self.value = raw


@dataclass
class DefaultedInitVar:
    raw: InitVar[int] = 3
    value: int = field(init=False)

    def __post_init__(self, raw: int) -> None:
        self.value = raw


def test_a_hand_written_init_is_reported_by_the_field_it_cannot_receive():
    with pytest.raises(ConfigError) as info:
        validate_schema(RenamedInit)
    message = info.value.issues[0].message
    assert message.startswith("RenamedInit.__init__ takes no value argument for the field of that name")
    assert "leave the generated __init__ in place" in message


def test_a_required_init_var_is_reported_as_an_argument_no_file_can_supply():
    with pytest.raises(ConfigError) as info:
        validate_schema(RequiredInitVar)
    message = info.value.issues[0].message
    assert message.startswith("RequiredInitVar.__init__ requires the raw argument")
    assert "declare it as an ordinary field" in message


def test_an_init_var_carrying_a_default_builds():
    # The default answers the constructor, so the schema is complete as declared.
    validate_schema(DefaultedInitVar)
    assert from_dict(DefaultedInitVar, {}).value == 3


@dataclass
class BorrowedInit:
    x: int = 0

    __init__ = DefaultedInitVar.__init__


@dataclass
class NotAFunctionInit:
    x: int = 0

    __init__ = object.__init__


CONSTRUCTOR_CASES: list[tuple[str, type, str]] = [
    ("borrowed __init__", BorrowedInit, "BorrowedInit.__init__ takes no x argument"),
    ("__init__ that is not a function", NotAFunctionInit, "NotAFunctionInit carries an __init__ that is not"),
]


@pytest.mark.parametrize(
    ("config_cls", "opening"),
    [(config_cls, opening) for _name, config_cls, opening in CONSTRUCTOR_CASES],
    ids=[name for name, _config_cls, _opening in CONSTRUCTOR_CASES],
)
def test_a_constructor_that_cannot_take_the_fields_is_reported(config_cls: type, opening: str):
    # The rule reads what the constructor accepts rather than where its body came
    # from, so every shape confingo's own call cannot satisfy is named the same way.
    with pytest.raises(ConfigError) as info:
        validate_schema(config_cls)
    assert info.value.issues[0].message.startswith(opening)


@dataclass(kw_only=True)
class KeywordOnlyFields:
    a: int = 1
    b: str = "x"


@dataclass(frozen=True, slots=True, weakref_slot=True)
class SlottedFrozen:
    a: int = 1


@dataclass(match_args=False)
class NoMatchArgs:
    a: int = 1


@dataclass
class RequiredThenDefaulted:
    a: int
    b: int = 2


@dataclass
class WithNonInitField:
    a: int = 1
    derived: int = field(init=False, default=0)


@dataclass(frozen=True)
class InheritedFields(SlottedFrozen):
    b: int = 2


GENERATED_SHAPES: list[type] = [
    KeywordOnlyFields,
    SlottedFrozen,
    NoMatchArgs,
    RequiredThenDefaulted,
    WithNonInitField,
    InheritedFields,
    DefaultedInitVar,
    Window,
]


@pytest.mark.parametrize("config_cls", GENERATED_SHAPES, ids=[item.__name__ for item in GENERATED_SHAPES])
def test_a_generated_constructor_is_never_caught_by_the_rule(config_cls: type):
    # The generated __init__ takes the class's own fields whatever flags produced
    # it, so the rule reads as absent for every ordinary declaration.
    validate_schema(config_cls)


# --- a schema class is a section and nothing the walks also dispatch on -------


@dataclass
class SectionAndMapping(Mapping[str, int]):
    x: int = 1

    @override
    def __getitem__(self, key: str) -> int:
        if key == "x":
            return self.x
        raise KeyError(key)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(("x",))

    @override
    def __len__(self) -> int:
        return 1


@dataclass
class SectionAndSequence(Sequence[int]):
    x: int = 1

    @override
    def __getitem__(self, index: int) -> int:  # pyrefly: ignore[bad-override]
        return (self.x,)[index]

    @override
    def __len__(self) -> int:
        return 1


PROTOCOL_CASES: list[tuple[str, type, str]] = [
    ("mapping", SectionAndMapping, "a mapping"),
    ("sequence", SectionAndSequence, "a sequence"),
]


@pytest.mark.parametrize(
    ("config_cls", "described"),
    [(config_cls, described) for _name, config_cls, described in PROTOCOL_CASES],
    ids=[name for name, _config_cls, _described in PROTOCOL_CASES],
)
def test_a_section_that_is_also_a_dispatch_kind_is_reported(config_cls: type, described: str):
    with pytest.raises(ConfigError) as info:
        validate_schema(config_cls)
    message = info.value.issues[0].message
    assert message.startswith(f"{config_cls.__name__} is a config section and also {described}")
    assert "declare the section as a plain dataclass" in message


@dataclass
class ProtocolSectionRoot:
    section: SectionAndMapping = field(default_factory=SectionAndMapping)


def test_the_rule_reaches_a_dispatch_kind_named_as_a_nested_section():
    with pytest.raises(ConfigError) as info:
        from_dict(ProtocolSectionRoot, {})
    assert info.value.issues[0].path == "section"


# --- the subscripted empty shape tuple pins a scalar array --------------------


@dataclass
class NumpyScalar:
    x: np.ndarray[tuple[()], np.dtype[np.float64]] = field(default_factory=lambda: np.zeros(()))


@dataclass
class TorchScalar:
    x: Annotated[torch.Tensor, torch.float64, tuple[()]] = field(
        default_factory=lambda: torch.tensor(0.0, dtype=torch.float64)
    )


@pytest.mark.parametrize("config_cls", [NumpyScalar, TorchScalar], ids=["numpy", "torch"])
def test_the_empty_shape_tuple_names_zero_dimensions(config_cls: type):
    assert from_dict(config_cls, {"x": 1.5}).x.ndim == 0
    assert issues_of(config_cls, {"x": [1.5]}) == [("x", "expected a 0-dimensional array, got 1 dimensions")]


@dataclass
class UnclaimedShape:
    x: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(1))


def test_an_unsubscripted_shape_still_carries_no_dimensionality_claim():
    # A bare alias names no shape, so an array of any rank is admitted.
    assert from_dict(UnclaimedShape, {"x": [[1.0, 2.0]]}).x.ndim == 2
    assert from_dict(UnclaimedShape, {"x": 1.0}).x.ndim == 0


# --- the array-annotation cache answers per set of loaded backends ------------


ARRAY_HINTS: list[Any] = [
    int,
    list[int],
    npt.NDArray[np.float64],
    np.ndarray[tuple[int, int], np.dtype[np.float32]],
    Annotated[torch.Tensor, torch.float64],
]


@pytest.mark.parametrize("hint", ARRAY_HINTS, ids=[str(index) for index in range(len(ARRAY_HINTS))])
def test_the_cached_array_match_answers_what_the_classifier_answers(hint: Any):
    live = capture_backend_snapshot()
    direct = inspect_annotation(hint)
    cached = array_match(hint, live)
    assert cached.matched == direct.matched
    assert cached.spec == direct.spec
    assert cached.error == direct.error
    # Twice through the cache is the same answer.
    assert array_match(hint, live) == cached


def test_a_second_backend_snapshot_is_a_separate_cache_entry():
    # The snapshot names which backends the classification was made against, so a
    # different set of them recomputes rather than reading the first entry back.
    hint = npt.NDArray[np.float64]
    array_match(hint, capture_backend_snapshot())
    before = _array_match_by_id.cache_info().misses
    array_match(hint, BackendSnapshot(numpy=None, torch=None))
    assert _array_match_by_id.cache_info().misses == before + 1


# --- the whole set still round-trips ------------------------------------------


def test_the_fixed_schemas_round_trip():
    for config_cls in (NumpyScalar, TorchScalar, UnclaimedShape, DefaultedInitVar):
        built = from_dict(config_cls, {})
        assert config_equal(from_dict(config_cls, to_dict(built)), built)


@dataclass
class NodeWithHash(ConfigNode):
    lr: float = 0.1


def test_a_node_reaches_the_same_digest_through_its_accessor():
    # A node's generated constructor takes its own fields like any other, so the
    # constructor rule leaves it alone here too.
    validate_schema(NodeWithHash)
    node = NodeWithHash()
    assert node.cfg.hash() == config_hash(node)
