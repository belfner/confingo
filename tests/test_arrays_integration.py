"""End-to-end array support tests through ``from_dict`` / ``to_dict`` / file IO."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import (
    dataclass,
    field,
    make_dataclass,
)
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    TypeVar,
    override,
)


if TYPE_CHECKING:
    from pathlib import Path

import pytest


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
npt = pytest.importorskip("numpy.typing")

from confingo import (  # noqa: E402
    ConfigError,
    ConfigNode,
    ConfigValue,
)
from confingo.functional import (  # noqa: E402
    config_equal,
    config_hash,
    from_dict,
    load_json,
    save_json,
    to_dict,
    validate_schema,
)


def schema_for(hint: Any) -> type:
    """Build a one-field dataclass carrying a real (non-string) annotation."""
    return make_dataclass("ArrayHolder", [("x", hint)])


# --- module-level schemas -----------------------------------------------------


@dataclass
class ArraySection:
    weights: npt.NDArray[np.float32]
    bias: Annotated[torch.Tensor, torch.float64]
    scale: float = 1.0


@dataclass
class ArrayRoot(ConfigNode):
    section: ArraySection
    tags: npt.NDArray[np.int64] = field(default_factory=lambda: np.array([0], dtype=np.int64))


# --- per-dtype round trips ----------------------------------------------------


@pytest.mark.parametrize(
    ("scalar", "values"),
    [
        (np.bool_, [True, False]),
        (np.uint8, [0, 255]),
        (np.int16, [-32768, 32767]),
        (np.int32, [-(2**31), 2**31 - 1]),
        (np.int64, [-(2**63), 2**63 - 1]),
        (np.uint64, [0, 2**64 - 1]),
        (np.float16, [1.5, 65504.0, -0.0009765625]),
        (np.float32, [1.5, 3.25e38, -1.25]),
        (np.float64, [1.5, 1e300, 2**53 - 1.0]),
    ],
)
def test_numpy_concrete_round_trips_bit_exactly(scalar: Any, values: list[Any]):
    cls = schema_for(npt.NDArray[scalar])
    built = from_dict(cls, {"x": values})
    assert built.x.dtype == np.dtype(scalar)
    plain = to_dict(built)
    rebuilt = from_dict(cls, plain)
    assert rebuilt.x.dtype == built.x.dtype
    assert rebuilt.x.shape == built.x.shape
    assert np.array_equal(rebuilt.x, built.x)
    assert to_dict(rebuilt) == plain


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (torch.bool, [True, False]),
        (torch.uint8, [0, 255]),
        (torch.int8, [-128, 127]),
        (torch.int16, [-32768, 32767]),
        (torch.int32, [-(2**31), 2**31 - 1]),
        (torch.int64, [-(2**63), 2**63 - 1]),
        (torch.float16, [1.5, 65504.0]),
        (torch.bfloat16, [1.5, 2.25, -0.125]),
        (torch.float32, [1.5, -1.25]),
        (torch.float64, [1.5, 1e300]),
    ],
)
def test_torch_concrete_round_trips_bit_exactly(dtype: Any, values: list[Any]):
    cls = schema_for(Annotated[torch.Tensor, dtype])
    built = from_dict(cls, {"x": values})
    assert built.x.dtype is dtype
    plain = to_dict(built)
    rebuilt = from_dict(cls, plain)
    assert rebuilt.x.dtype is dtype
    assert torch.equal(rebuilt.x, built.x)
    assert to_dict(rebuilt) == plain


def test_bare_torch_is_immune_to_the_process_default_dtype():
    cls = schema_for(torch.Tensor)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        first = from_dict(cls, {"x": [1.5, 2.5]})
        hash_first = config_hash(first)
        torch.set_default_dtype(torch.float16)
        second = from_dict(cls, to_dict(first))
        hash_second = config_hash(second)
    finally:
        torch.set_default_dtype(previous)
    assert first.x.dtype is torch.float64
    assert second.x.dtype is torch.float64
    assert hash_first == hash_second


# --- canonical equality and file IO -------------------------------------------


def test_round_trip_equality_reads_literally_with_arrays():
    config = ArrayRoot(
        section=ArraySection(
            weights=np.array([[1.5, 2.5]], dtype=np.float32), bias=torch.tensor([0.5], dtype=torch.float64)
        ),
        tags=np.array([1, 2], dtype=np.int64),
    )
    assert from_dict(ArrayRoot, to_dict(config)) == config
    assert config.section == from_dict(ArrayRoot, to_dict(config)).section


def test_json_file_round_trip_with_arrays(tmp_path: Path):
    config = ArrayRoot(
        section=ArraySection(
            weights=np.array([[1.0]], dtype=np.float32), bias=torch.tensor([2.0], dtype=torch.float64)
        ),
    )
    path = save_json(config, tmp_path / "config.json")
    loaded = load_json(ArrayRoot, path)
    assert loaded == config
    assert config_hash(loaded) == config_hash(config)


def test_hash_collides_across_unencoded_state():
    bare = schema_for(np.ndarray)
    narrow = from_dict(bare, {"x": np.array([1, 2], dtype=np.int32)})
    wide = from_dict(bare, {"x": np.array([1, 2], dtype=np.int64)})
    assert config_hash(narrow) == config_hash(wide)

    tensor_cls = schema_for(torch.Tensor)
    plain_tensor = from_dict(tensor_cls, {"x": torch.tensor([1.0, 2.0], dtype=torch.float64)})
    grad_tensor = from_dict(tensor_cls, {"x": torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)})
    assert config_hash(plain_tensor) == config_hash(grad_tensor)


# --- programmatic input preservation ------------------------------------------


def test_supplied_tensors_keep_grad_state_until_serialization():
    cls = schema_for(torch.Tensor)
    tensor = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    built = from_dict(cls, {"x": tensor})
    assert built.x is tensor
    assert built.x.requires_grad
    assert to_dict(built) == {"x": [1.0, 2.0]}


# --- arrays inside unions, optionals, and containers --------------------------


def test_optional_array_fields():
    cls = schema_for(npt.NDArray[np.float64] | None)
    assert from_dict(cls, {"x": None}).x is None
    built = from_dict(cls, {"x": [1.0, 2.0]})
    assert built.x.dtype == np.dtype(np.float64)


def test_containers_of_arrays():
    list_cls = schema_for(list[npt.NDArray[np.float32]])
    built = from_dict(list_cls, {"x": [[1.0], [2.0, 3.0]]})
    assert [item.tolist() for item in built.x] == [[1.0], [2.0, 3.0]]
    assert to_dict(built) == {"x": [[1.0], [2.0, 3.0]]}

    dict_cls = schema_for(dict[str, Annotated[torch.Tensor, torch.int32]])
    built = from_dict(dict_cls, {"x": {"a": [1, 2]}})
    assert built.x["a"].dtype is torch.int32


def test_container_element_failures_carry_indexed_paths():
    cls = schema_for(list[npt.NDArray[np.float32]])
    with pytest.raises(ConfigError) as info:
        from_dict(cls, {"x": [[1.0], ["bad"]]})
    assert any(i.path == "x.1.0" and "got str" in i.message for i in info.value.issues)


# --- arrays under ConfigValue -------------------------------------------------


def test_config_value_fields_reject_arrays():
    cls = schema_for(ConfigValue)
    for value in (np.array([[1, 2]], dtype=np.int32), torch.tensor([1.0, 2.0])):
        with pytest.raises(ConfigError) as info:
            from_dict(cls, {"x": value})
        assert any(i.path == "x" and "expected plain data for ConfigValue" in i.message for i in info.value.issues)


def test_config_value_fields_report_nested_arrays_at_their_path():
    cls = schema_for(ConfigValue)
    with pytest.raises(ConfigError) as info:
        from_dict(cls, {"x": {"nested": [np.array([1.0])]}})
    assert any(i.path == "x.nested.0" and "expected plain data for ConfigValue" in i.message for i in info.value.issues)


def test_config_value_fields_reject_numpy_scalars():
    cls = schema_for(ConfigValue)
    with pytest.raises(ConfigError) as info:
        from_dict(cls, {"x": np.float32(1.5)})
    assert any(i.path == "x" and "expected plain data for ConfigValue" in i.message for i in info.value.issues)


def test_array_annotations_carry_arrays_a_config_value_field_declines():
    cls = schema_for(npt.NDArray[np.int32])
    value = np.array([[1, 2]], dtype=np.int32)
    built = from_dict(cls, {"x": value})
    assert built.x is value
    plain = to_dict(built)
    assert plain == {"x": [[1, 2]]}
    assert from_dict(cls, plain).x.tolist() == [[1, 2]]


def test_array_annotations_reject_nonfinite_elements_with_paths():
    cls = schema_for(dict[str, npt.NDArray[np.float64]])
    with pytest.raises(ConfigError) as info:
        from_dict(cls, {"x": {"nested": np.array([float("nan")])}})
    assert any(i.path == "x.nested.0" and "finite" in i.message for i in info.value.issues)


# --- numpy scalars on ordinary scalar fields ----------------------------------


def test_numpy_scalars_feed_scalar_fields():
    @dataclass
    class Scalars:
        f: float
        i: int
        b: bool

    built = from_dict(Scalars, {"f": np.float32(1.5), "i": np.int64(7), "b": np.bool_(True)})
    assert built == Scalars(f=1.5, i=7, b=True)
    built = from_dict(Scalars, {"f": np.int16(2), "i": np.float64(3.0), "b": np.bool_(False)})
    assert built.f == 2.0
    assert built.i == 3


def test_nonfinite_numpy_scalars_are_rejected():
    cls = schema_for(float)
    with pytest.raises(ConfigError) as info:
        from_dict(cls, {"x": np.float64("nan")})
    assert any("finite" in i.message for i in info.value.issues)


# --- schema preflight through from_dict ---------------------------------------


def test_bad_array_annotations_are_reported_even_when_omitted():
    for hint, fragment in [
        (npt.NDArray[np.complex64], "unsupported array dtype complex64"),
        (npt.NDArray[np.generic], "unsupported numpy dtype family numpy.generic"),
        (Annotated[torch.Tensor, torch.float32, torch.float64], "conflicting torch dtype metadata"),
        (Annotated[torch.Tensor, torch.complex64], "unsupported array dtype complex64"),
    ]:
        cls = schema_for(hint)
        with pytest.raises(ConfigError) as info:
            from_dict(cls, {})
        assert any(i.path == "x" and fragment in i.message for i in info.value.issues), info.value.issues


def test_array_annotations_inside_unions_are_preflighted():
    cls = schema_for(npt.NDArray[np.complex64] | None)
    with pytest.raises(ConfigError) as info:
        from_dict(cls, {})
    assert any("unsupported array dtype complex64" in i.message for i in info.value.issues)


# --- shape edge cases through the full stack ----------------------------------


def test_zero_dimensional_arrays_round_trip():
    cls = schema_for(npt.NDArray[np.float64])
    built = from_dict(cls, {"x": 2.5})
    assert built.x.shape == ()
    assert to_dict(built) == {"x": 2.5}
    rebuilt = from_dict(cls, to_dict(built))
    assert rebuilt.x.shape == ()


def test_fixed_ndim_zero_size_round_trips_via_padding():
    cls = schema_for(np.ndarray[tuple[int, int], np.dtype[np.float64]])
    original = from_dict(cls, {"x": np.zeros((0, 3))})
    plain = to_dict(original)
    assert plain == {"x": []}
    rebuilt = from_dict(cls, plain)
    assert rebuilt.x.shape == (0, 0)
    assert to_dict(rebuilt) == plain


def test_tensor_fixed_ndim_round_trips_through_from_dict():
    cls = schema_for(Annotated[torch.Tensor, torch.float32, tuple[int, int]])
    built = from_dict(cls, {"x": [[1.5, 2.5]]})
    assert built.x.shape == (1, 2)
    rebuilt = from_dict(cls, to_dict(built))
    assert torch.equal(rebuilt.x, built.x)
    with pytest.raises(ConfigError) as info:
        from_dict(cls, {"x": [1.5, 2.5]})
    assert any(i.message == "expected a 2-dimensional array, got 1 dimensions" for i in info.value.issues)


def test_fixed_ndim_mismatch_reports_through_from_dict():
    cls = schema_for(np.ndarray[tuple[int, int], np.dtype[np.float64]])
    with pytest.raises(ConfigError) as info:
        from_dict(cls, {"x": [1.0, 2.0]})
    assert any(i.message == "expected a 2-dimensional array, got 1 dimensions" for i in info.value.issues)


# --- Annotated behavior for ordinary types is unchanged -----------------------


@dataclass
class AnnotatedSection:
    y: int = 3


@dataclass
class AnnotatedHolder:
    count: Annotated[int, "a count"] = 0
    section: Annotated[AnnotatedSection, "a section"] = field(default_factory=AnnotatedSection)
    items: Annotated[list[int], "items"] = field(default_factory=list)


def test_annotated_ordinary_fields_behave_as_their_base():
    built = from_dict(AnnotatedHolder, {"count": 1.0, "section": {"y": 5}, "items": [1, 2]})
    assert built.count == 1
    assert built.section.y == 5
    assert built.items == [1, 2]
    assert to_dict(built) == {"count": 1, "section": {"y": 5}, "items": [1, 2]}


@dataclass
class AnnotatedImplicit:
    section: Annotated[AnnotatedSection, "implicit"]


def test_annotated_dataclass_sections_still_build_implicitly():
    assert from_dict(AnnotatedImplicit, {}).section.y == 3


# --- canonical equality on array fields ---------------------------------------


@dataclass
class PlainWithArray:
    weights: npt.NDArray[np.float32] = field(default_factory=lambda: np.zeros(1, dtype=np.float32))


@dataclass
class ArrayBox(ConfigNode):
    """Holds an array from either backend, so cross-backend pairs share a class."""

    value: np.ndarray | torch.Tensor


@dataclass
class PlainBox(ConfigNode):
    """Holds plain data, for the projection rules the scalar and key paths use."""

    value: ConfigValue


def test_plain_array_bearing_dataclass_compares_after_injection():
    left = from_dict(PlainWithArray, {"weights": [[1.0, 2.0], [3.0, 4.0]]})
    right = from_dict(PlainWithArray, {"weights": [[1.0, 2.0], [3.0, 4.0]]})
    assert left == right
    assert left != from_dict(PlainWithArray, {"weights": [[1.0, 2.0], [3.0, 9.0]]})


def test_tensor_and_ndarray_under_any_compare_by_value():
    assert ArrayBox(value=np.array([1.0, 2.0])) == ArrayBox(value=torch.tensor([1.0, 2.0]))
    assert ArrayBox(value=torch.tensor([[1, 2]])) == ArrayBox(value=np.array([[1, 2]]))
    assert ArrayBox(value=np.array([1.0, 2.0])) != ArrayBox(value=torch.tensor([1.0, 2.5]))
    assert ArrayBox(value=np.array([1.0, 2.0])) != ArrayBox(value=torch.tensor([[1.0, 2.0]]))


def test_cross_kind_pairs_compare_by_canonical_plain_form():
    huge = 2**63 - 1
    # Precision loss makes these serialize to different numbers -> unequal.
    assert ArrayBox(value=np.array([huge], dtype=np.int64)) != ArrayBox(value=np.array([float(huge)]))
    # int and float serialize to distinct plain forms (3 vs 3.0), exactly how
    # config_hash tokenizes them, so a cross-kind pair is unequal and the
    # equal-configs -> equal-fingerprint invariant holds.
    assert ArrayBox(value=torch.tensor([3], dtype=torch.int64)) != ArrayBox(value=torch.tensor([3.0]))


def test_signed_zero_arrays_compare_by_canonical_plain_form():
    # 0.0 and -0.0 are equal by value but serialize to distinct JSON tokens, so
    # the native float fast path must treat them as unequal to match config_hash.
    assert ArrayBox(value=np.array([0.0])) != ArrayBox(value=np.array([-0.0]))
    assert ArrayBox(value=torch.tensor([0.0])) != ArrayBox(value=torch.tensor([-0.0]))
    assert ArrayBox(value=np.array([0.0, 1.5])) != ArrayBox(value=np.array([-0.0, 1.5]))
    # Cross-backend and same-sign pairs still resolve correctly.
    assert ArrayBox(value=np.array([0.0])) != ArrayBox(value=torch.tensor([-0.0]))
    assert ArrayBox(value=np.array([-0.0])) == ArrayBox(value=torch.tensor([-0.0]))


def test_zero_size_arrays_keep_dimension_collapse_equality():
    assert ArrayBox(value=np.zeros((0, 3))) == ArrayBox(value=np.zeros((0, 4)))
    assert ArrayBox(value=torch.zeros((0, 3))) == ArrayBox(value=torch.zeros((0, 5)))


def test_grad_and_bfloat16_tensors_compare_by_value():
    plain = torch.tensor([1.5, 2.5])
    grad = torch.tensor([1.5, 2.5], requires_grad=True)
    assert ArrayBox(value=plain) == ArrayBox(value=grad)
    assert ArrayBox(value=torch.tensor([1.5], dtype=torch.bfloat16)) == ArrayBox(
        value=np.array([1.5], dtype=np.float32)
    )


def test_nan_bearing_arrays_compare_unequal_outside_the_finite_domain():
    from_dict(PlainWithArray, {})
    left = PlainWithArray(weights=np.array([np.nan], dtype=np.float32))
    right = PlainWithArray(weights=np.array([np.nan], dtype=np.float32))
    assert left != right


def test_scalar_shortcut_stays_exact_for_numpy_scalars():
    assert PlainBox(value=np.float64(2**53)) != PlainBox(value=2**53 + 1)
    assert PlainBox(value=np.float64(1.5)) == PlainBox(value=1.5)


def test_numpy_scalars_held_in_memory_serialize_and_fingerprint_plain():
    # np.float64 is a float to the type checker and to Python, so a directly
    # constructed value carries one; the marshal walk normalizes it through the
    # adapter, and the fingerprint reads the same plain form.
    held = PlainBox(value=np.float64(1.5))
    assert to_dict(held) == {"value": 1.5}
    assert config_hash(held) == config_hash(PlainBox(value=1.5))


def test_canonicalizing_dict_keys_compare_by_plain_form():
    from pathlib import Path  # noqa: PLC0415

    # A Path key sits outside ConfigValue, so the checker flags the literal; the
    # assertion is about the projection the comparison applies to whatever a key is.
    left = PlainBox(value={Path("x"): 1})  # pyrefly: ignore[bad-argument-type]
    assert left == PlainBox(value={"x": 1})
    assert PlainBox(value={"x": 1}) != PlainBox(value={"y": 1})


def test_negative_bit_tensor_views_compare_by_value():
    view = torch._neg_view(torch.tensor([1.0, -2.0]))
    assert ArrayBox(value=view) == ArrayBox(value=np.array([-1.0, 2.0]))
    assert ArrayBox(value=view) == ArrayBox(value=torch.tensor([-1.0, 2.0]))
    assert ArrayBox(value=view) != ArrayBox(value=np.array([1.0, -2.0]))


def test_tensor_subclasses_compare_by_serialized_value():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    assert ArrayBox(value=param) == ArrayBox(value=torch.tensor([1.0, 2.0]))
    assert ArrayBox(value=param) != ArrayBox(value=torch.tensor([1.0, 3.0]))


def test_array_mapping_keys_are_rejected_at_load():
    cls = schema_for(ConfigValue)
    with pytest.raises(ConfigError) as info:
        from_dict(cls, {"x": {torch.tensor([1, 2]): "v"}})
    assert any("expected a str mapping key, got Tensor" in i.message for i in info.value.issues)


def test_array_mapping_keys_collect_on_marshal():
    cls = schema_for(ConfigValue)
    built = cls(x={torch.tensor([1, 2]): "v"})
    with pytest.raises(ConfigError) as info:
        to_dict(built)
    assert any("cannot serialize Tensor as a mapping key" in i.message for i in info.value.issues)


def test_tuple_mapping_keys_collect_on_marshal():
    cls = schema_for(ConfigValue)
    built = cls(x={(1, 2): "v"})
    with pytest.raises(ConfigError) as info:
        to_dict(built)
    assert any("cannot serialize tuple as a mapping key" in i.message for i in info.value.issues)


# --- hidden backends through the full stack -----------------------------------


def test_hidden_numpy_turns_array_annotations_into_schema_errors(monkeypatch: pytest.MonkeyPatch):
    cls = schema_for(npt.NDArray[np.float32])
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ConfigError) as info:
        from_dict(cls, {"x": [1.0]})
    assert any("unsupported field type" in i.message for i in info.value.issues)


# --- import hygiene and cross-process stability -------------------------------


def run_python(code: str) -> str:
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd="/tmp")
    return result.stdout.strip()


def test_importing_confingo_loads_neither_backend():
    out = run_python("import sys, confingo; print('numpy' in sys.modules, 'torch' in sys.modules)")
    assert out == "False False"


def test_numpy_only_schemas_never_load_torch():
    out = run_python(
        "import sys\n"
        "import numpy as np\n"
        "import numpy.typing as npt\n"
        "from dataclasses import make_dataclass\n"
        "from confingo.functional import from_dict, to_dict\n"
        "cls = make_dataclass('S', [('x', npt.NDArray[np.float64])])\n"
        "import warnings\n"
        "warnings.simplefilter('ignore')\n"
        "built = from_dict(cls, {'x': [1.0, 2.0]})\n"
        "print(to_dict(built), 'torch' in sys.modules)\n"
    )
    assert out == "{'x': [1.0, 2.0]} False"


def test_config_hash_is_stable_across_processes():
    code = (
        "import warnings; warnings.simplefilter('ignore')\n"
        "import numpy as np, numpy.typing as npt, torch\n"
        "from dataclasses import make_dataclass\n"
        "from typing import Annotated\n"
        "from confingo.functional import config_hash, from_dict\n"
        "cls = make_dataclass('S', [('a', npt.NDArray[np.float32]), ('b', Annotated[torch.Tensor, torch.int32])])\n"
        "built = from_dict(cls, {'a': [1.5, 2.5], 'b': [1, 2]})\n"
        "print(config_hash(built))\n"
    )
    assert run_python(code) == run_python(code)


def test_an_array_element_in_a_set_is_rejected_at_preflight():
    """An array rebuilds as an ordinary unhashable array, so a set cannot hold one.

    A conforming array subclass that defines its own ``__hash__`` can be put in a
    Python set and would pass a value-level check, while the load that rebuilds it
    produces a plain array. The annotation settles it before any value is read.
    """
    holder = schema_for(set[npt.NDArray[np.float64]])
    with pytest.raises(ConfigError) as info:
        from_dict(holder, {})
    messages = [issue.message for issue in info.value.issues]
    assert any("cannot be built" in message for message in messages), messages
    assert any("hold the values in a list" in message for message in messages), messages


class DeepPlainArray(np.ndarray):
    """An ndarray subclass whose plain form is deeper than its shape.

    The shape carries one axis while ``tolist`` hands back 62 nested lists, which
    is what separates measuring the shape from measuring what the class writes.
    """

    __hash__ = object.__hash__

    def tolist(self) -> Any:
        """Build a plain form 62 levels deep.

        Returns:
          Any: Nested single-element lists around a float.
        """
        value: Any = 0.0
        for _ in range(62):
            value = [value]
        return value


DEEP_PLAIN_DEFAULT = np.zeros(1).view(DeepPlainArray)


@dataclass
class DeepPlainHolder:
    x: np.ndarray = DEEP_PLAIN_DEFAULT


@dataclass
class DeepPlainParent:
    children: list[DeepPlainHolder] = field(default_factory=list)


def _plain_depth(node: Any) -> int:
    """Count the list levels a plain form carries.

    Args:
      node (Any): The plain value to measure.

    Returns:
      int: The number of nested list levels.
    """
    levels = 0
    while isinstance(node, list) and len(node) > 0:
        levels += 1
        node = node[0]
    return levels


def test_an_array_subclass_default_is_measured_by_what_it_writes():
    # Its shape is (1,), so a shape-derived measure would call it one level and
    # let it through anywhere. Rendering it once is what charges the 62 levels
    # its tolist actually writes, and at a root field those levels fit.
    built = from_dict(DeepPlainHolder, {})
    assert _plain_depth(to_dict(built)["x"]) == 62


def test_the_measured_depth_follows_the_default_to_a_deeper_position():
    # The same default two levels down spends 62 levels from there, so the load
    # that selects it reports at the field's own path rather than leaving export
    # to raise from the bottom of the nesting.
    with pytest.raises(ConfigError) as info:
        from_dict(DeepPlainParent, {"children": [{}]})
    assert [issue.path for issue in info.value.issues] == ["children.0.x"]
    assert "nesting reaches the 64 level limit" in info.value.issues[0].message


class SelfReferentialPlain(np.ndarray):
    """An ndarray subclass whose plain form reaches the array again."""

    __hash__ = object.__hash__

    def tolist(self) -> Any:
        """Answer with the array itself.

        Returns:
          Any: This array, which reaches itself rather than terminating.
        """
        return self


SELF_REFERENTIAL_DEFAULT = np.zeros(1).view(SelfReferentialPlain)


@dataclass
class SelfReferentialHolder:
    x: np.ndarray = SELF_REFERENTIAL_DEFAULT


@pytest.mark.parametrize(
    "operation",
    [validate_schema, lambda config_cls: from_dict(config_cls, {})],
    ids=["validate_schema", "from_dict"],
)
def test_a_rendered_form_that_reaches_its_array_is_reported_at_its_path(operation: Any):
    # The rendered form comes from the array's own class, so the array joins the
    # branch before the walk follows it, and a graph pointing back arrives as the
    # ordinary cycle report rather than as a raw RecursionError.
    with pytest.raises(ConfigError) as info:
        operation(SelfReferentialHolder)
    (issue,) = info.value.issues
    assert issue.path == "x"
    assert "value holds itself, so it has no plain form" in issue.message


class FreshPlainArray(np.ndarray):
    """An ndarray subclass whose plain form is a new array of the same class."""

    __hash__ = object.__hash__

    def tolist(self) -> Any:
        """Answer with a fresh array rather than with lists and numbers.

        Returns:
          Any: A new array whose own rendering answers the same way.
        """
        return np.zeros(1).view(FreshPlainArray)


def _terminating_chain(hops: int) -> type:
    """Build an ndarray subclass whose rendering reaches a scalar after ``hops``.

    Args:
      hops (int): How many arrays the chain follows before answering with a number.

    Returns:
      type: The subclass to view an array as.
    """

    class Link(np.ndarray):
        __hash__ = object.__hash__

        def tolist(self) -> Any:
            if hops <= 0:
                return 7
            return np.zeros(1).view(_terminating_chain(hops - 1))

    return Link


@dataclass
class FreshSupplied:
    x: np.ndarray


@dataclass
class FreshFactory:
    x: np.ndarray = field(default_factory=lambda: np.zeros(1).view(FreshPlainArray))


FRESH_DEFAULT = np.zeros(1).view(FreshPlainArray)

SHORT_CHAIN_DEFAULT = np.zeros(1).view(_terminating_chain(60))


@dataclass
class FreshDefault:
    x: np.ndarray = FRESH_DEFAULT


@dataclass
class ShortChain:
    x: np.ndarray = SHORT_CHAIN_DEFAULT


HOP_LIMIT_TEXT = "rendering the plain form followed 64 arrays into one another"


def _render_operations() -> list[tuple[str, Any]]:
    """Build one entry per operation that walks a value's plain form.

    Returns:
      list[tuple[str, Any]]: An id and a callable per operation.
    """
    supplied = {"x": np.zeros(1).view(FreshPlainArray)}
    return [
        ("to_dict", lambda: to_dict(from_dict(FreshSupplied, supplied))),
        ("config_hash", lambda: config_hash(from_dict(FreshSupplied, supplied))),
        ("default_factory", lambda: from_dict(FreshFactory, {})),
        ("direct_default", lambda: from_dict(FreshDefault, {})),
        ("validate_schema", lambda: validate_schema(FreshDefault)),
    ]


@pytest.mark.parametrize(("label", "operation"), _render_operations(), ids=[name for name, _ in _render_operations()])
def test_a_render_chain_that_never_terminates_reports_the_hop_limit(label: str, operation: Any):
    # Each result carries a new identity, so the identity stack cannot end the
    # chain; the hop budget is what does, at the field's own path.
    with pytest.raises(ConfigError) as info:
        operation()
    assert any(HOP_LIMIT_TEXT in issue.message for issue in info.value.issues), info.value.issues
    assert all(issue.path == "x" for issue in info.value.issues), info.value.issues


def test_equality_reports_the_hop_limit_the_way_export_does():
    # Equality falls back to the plain form for a subclass, so it carries the
    # same budget and answers the same way export does.
    supplied = {"x": np.zeros(1).view(FreshPlainArray)}
    left = from_dict(FreshSupplied, supplied)
    right = from_dict(FreshSupplied, supplied)
    with pytest.raises(ConfigError) as info:
        config_equal(left, right)
    assert any(HOP_LIMIT_TEXT in issue.message for issue in info.value.issues), info.value.issues


def test_a_render_chain_that_reaches_a_number_is_carried():
    # Sixty hops sit inside the budget, and the number the chain reaches is what
    # the field writes.
    assert to_dict(from_dict(ShortChain, {}))["x"] == 7


class HopLink(np.ndarray):
    """An ndarray subclass that renders into another for a set number of hops.

    ``remaining`` counts the array-to-array transitions still to come; the render
    at zero answers with a number, which is the terminal render that follows into
    no further array.
    """

    __hash__ = object.__hash__
    remaining = 0

    def tolist(self) -> Any:
        """Answer with the next array in the chain, or with the terminal number.

        Returns:
          Any: A further array while hops remain, otherwise the number 7.
        """
        if self.remaining == 0:
            return 7
        child = np.zeros(()).view(HopLink)
        child.remaining = self.remaining - 1
        return child


def _hop_chain(transitions: int) -> Any:
    """Build an array whose rendering follows into ``transitions`` further arrays.

    Args:
      transitions (int): Array-to-array transitions before the terminal number.

    Returns:
      Any: The head of the chain.
    """
    head = np.zeros(()).view(HopLink)
    head.remaining = transitions
    return head


@dataclass
class HopBox:
    x: np.ndarray


@pytest.mark.parametrize("transitions", [0, 1, 63, 64], ids=["none", "one", "under", "at-the-limit"])
def test_a_chain_terminating_within_the_hop_limit_is_carried(transitions: int):
    # The limit counts transitions into a further array, so the render that
    # reaches a number follows into nothing and costs no hop. A chain making
    # exactly the limit's worth of transitions still reaches its number.
    assert to_dict(from_dict(HopBox, {"x": _hop_chain(transitions)}))["x"] == 7


@pytest.mark.parametrize("transitions", [65, 66], ids=["one-past", "two-past"])
def test_a_chain_past_the_hop_limit_reports_it(transitions: int):
    with pytest.raises(ConfigError) as info:
        to_dict(from_dict(HopBox, {"x": _hop_chain(transitions)}))
    (issue,) = info.value.issues
    assert issue.path == "x"
    assert HOP_LIMIT_TEXT in issue.message


class UnnameableLeafMeta(type):
    """A metaclass that raises when its class is named.

    Intercepting ``__getattribute__`` keeps the declaration one a type checker
    accepts, so the failing read is a case confingo contains rather than one the
    checker rules out.
    """

    @override
    def __getattribute__(cls, name: str) -> Any:
        if name == "__name__":
            raise RuntimeError("leaf name boom")
        return super().__getattribute__(name)


class UnnameableLeaf(metaclass=UnnameableLeafMeta):
    pass


@dataclass
class ArrayLeafBox:
    x: np.ndarray


@pytest.mark.parametrize(
    ("supplied", "path", "expected"),
    [
        ([UnnameableLeaf()], "x.0", "expected a number, got a class that could not be named"),
        (
            UnnameableLeaf(),
            "x",
            "expected an array-compatible scalar or sequence for ndarray, got a class that could not be named",
        ),
    ],
    ids=["inside-a-sequence", "supplied-directly"],
)
def test_an_array_leaf_whose_class_declines_to_be_named_is_reported(supplied: Any, path: str, expected: str):
    # The leaf walk names the value's class to say what it expected instead, so
    # the class answering with a failure costs the name rather than the report.
    with pytest.raises(ConfigError) as info:
        from_dict(ArrayLeafBox, {"x": supplied})
    (issue,) = info.value.issues
    assert issue.path == path
    assert issue.message == expected


class UnnameableScalarMeta(type):
    """A metaclass that raises when its scalar class is named."""

    @override
    def __getattribute__(cls, name: str) -> Any:
        if name == "__name__":
            raise RuntimeError("scalar name boom")
        return super().__getattribute__(name)


class UnnameableScalar(np.generic, metaclass=UnnameableScalarMeta):
    pass


@dataclass
class UnnameableScalarHolder:
    x: np.ndarray[tuple[int, ...], np.dtype[UnnameableScalar]]


def test_an_abstract_scalar_whose_class_declines_to_be_named_is_reported():
    # The annotation parser names the scalar to say numpy builds no dtype for it,
    # and the class is one the author declared, so the name reads through the
    # guarded helper and the schema issue still arrives.
    with pytest.raises(ConfigError) as info:
        validate_schema(UnnameableScalarHolder)
    (issue,) = info.value.issues
    assert issue.path == "x"
    assert issue.message == "unsupported numpy dtype family numpy.a class that could not be named"


ScalarT = TypeVar("ScalarT", bound=np.generic)


@dataclass
class TypeVarScalarHolder:
    x: np.ndarray[tuple[int, ...], np.dtype[ScalarT]]


def test_a_type_variable_scalar_is_recognized_without_reading_its_class_name():
    # A TypeVar scalar argument names no concrete dtype, so the field carries the
    # bare-array rules. It is recognized by what it is rather than by what its
    # class is called, which a hostile metaclass would otherwise answer for.
    validate_schema(TypeVarScalarHolder)
    assert from_dict(TypeVarScalarHolder, {"x": [1.0, 2.0]}).x.tolist() == [1.0, 2.0]


@dataclass
class PostponedScalarHolder[ScalarP: np.generic]:
    x: np.ndarray[tuple[int, ...], np.dtype[ScalarP]]


@dataclass
class PostponedScalarBase[ScalarP: np.generic]:
    x: np.ndarray[tuple[int, ...], np.dtype[ScalarP]]


@dataclass
class PostponedScalarChild(PostponedScalarBase[np.float64]):
    y: int = 2


@pytest.mark.parametrize(
    ("config_cls", "owner", "parameter"),
    [
        (PostponedScalarHolder, "PostponedScalarHolder", "ScalarP"),
        (PostponedScalarChild, "PostponedScalarBase", "ScalarP"),
    ],
    ids=["own-parameter", "inherited-from-a-generic-base"],
)
def test_a_schema_taking_type_parameters_is_reported_with_the_concrete_remedy(
    config_cls: type[Any], owner: str, parameter: str
):
    # A config file carries concrete values, so a schema that takes a parameter
    # names a type a load cannot build. Specializing the base does not settle it
    # either: the annotation lives on the base and names the parameter, so the
    # class that declares it is the one reported.
    with pytest.raises(ConfigError) as info:
        validate_schema(config_cls)
    (issue,) = info.value.issues
    assert issue.message.startswith(f"{owner} takes the type parameter {parameter},")
    assert issue.message.endswith("derive anything that varies in an init=False field")


def test_a_module_level_type_variable_in_an_annotation_stays_supported():
    # The rejected thing is a schema class taking a parameter. A TypeVar reached
    # through an annotation belongs to whoever wrote the annotation, which is how
    # numpy spells its own aliases, so the field keeps the bare-array rules.
    assert TypeVarScalarHolder.__dict__.get("__type_params__", ()) == ()
    built = from_dict(TypeVarScalarHolder, {"x": [1.0, 2.0]})
    assert built.x.dtype == np.float64


@dataclass(frozen=True)
class FrozenArraySection:
    arr: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 1.0]))


@dataclass
class RootWithFrozenArray(ConfigNode):
    fa: FrozenArraySection = field(default_factory=FrozenArraySection)


def test_frozen_section_with_array_is_unhashable_rather_than_raising_on_the_array():
    built = from_dict(RootWithFrozenArray, {})
    assert type(built.fa).__dict__["__hash__"] is None
    with pytest.raises(TypeError, match="unhashable type"):
        hash(built.fa)
    assert config_hash(built.fa) == config_hash(built.fa)
