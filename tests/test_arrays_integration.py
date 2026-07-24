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
)


if TYPE_CHECKING:
    from pathlib import Path

import pytest


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
npt = pytest.importorskip("numpy.typing")

from confingo import (  # noqa: E402
    ConfigError,
    ConfigRoot,
    config_hash,
    from_dict,
    load_json,
    save_json,
    to_dict,
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
class ArrayRoot(ConfigRoot):
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


# --- arrays under Any ---------------------------------------------------------


def test_any_fields_retain_arrays_and_serialize_plain():
    cls = schema_for(Any)
    value = np.array([[1, 2]], dtype=np.int32)
    built = from_dict(cls, {"x": value})
    assert built.x is value
    plain = to_dict(built)
    assert plain == {"x": [[1, 2]]}
    assert from_dict(cls, plain).x == [[1, 2]]


def test_any_fields_reject_invalid_arrays_with_paths():
    cls = schema_for(Any)
    with pytest.raises(ConfigError) as info:
        from_dict(cls, {"x": {"nested": np.array([float("nan")])}})
    assert any(i.path == "x.nested.0" and "finite" in i.message for i in info.value.issues)


def test_any_fields_keep_numpy_scalars_in_memory_but_serialize_plain():
    cls = schema_for(Any)
    built = from_dict(cls, {"x": np.float32(1.5)})
    assert isinstance(built.x, np.floating)
    assert to_dict(built) == {"x": 1.5}


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
class AnyHolder(ConfigRoot):
    value: Any


def test_plain_array_bearing_dataclass_compares_after_injection():
    left = from_dict(PlainWithArray, {"weights": [[1.0, 2.0], [3.0, 4.0]]})
    right = from_dict(PlainWithArray, {"weights": [[1.0, 2.0], [3.0, 4.0]]})
    assert left == right
    assert left != from_dict(PlainWithArray, {"weights": [[1.0, 2.0], [3.0, 9.0]]})


def test_tensor_and_ndarray_under_any_compare_by_value():
    assert AnyHolder(value=np.array([1.0, 2.0])) == AnyHolder(value=torch.tensor([1.0, 2.0]))
    assert AnyHolder(value=torch.tensor([[1, 2]])) == AnyHolder(value=np.array([[1, 2]]))
    assert AnyHolder(value=np.array([1.0, 2.0])) != AnyHolder(value=torch.tensor([1.0, 2.5]))
    assert AnyHolder(value=np.array([1.0, 2.0])) != AnyHolder(value=torch.tensor([[1.0, 2.0]]))


def test_cross_kind_pairs_compare_by_canonical_plain_form():
    huge = 2**63 - 1
    # Precision loss makes these serialize to different numbers -> unequal.
    assert AnyHolder(value=np.array([huge], dtype=np.int64)) != AnyHolder(value=np.array([float(huge)]))
    # int and float serialize to distinct plain forms (3 vs 3.0), exactly how
    # config_hash tokenizes them, so a cross-kind pair is unequal and the
    # equal-configs -> equal-fingerprint invariant holds.
    assert AnyHolder(value=torch.tensor([3], dtype=torch.int64)) != AnyHolder(value=torch.tensor([3.0]))


def test_signed_zero_arrays_compare_by_canonical_plain_form():
    # 0.0 and -0.0 are equal by value but serialize to distinct JSON tokens, so
    # the native float fast path must treat them as unequal to match config_hash.
    assert AnyHolder(value=np.array([0.0])) != AnyHolder(value=np.array([-0.0]))
    assert AnyHolder(value=torch.tensor([0.0])) != AnyHolder(value=torch.tensor([-0.0]))
    assert AnyHolder(value=np.array([0.0, 1.5])) != AnyHolder(value=np.array([-0.0, 1.5]))
    # Cross-backend and same-sign pairs still resolve correctly.
    assert AnyHolder(value=np.array([0.0])) != AnyHolder(value=torch.tensor([-0.0]))
    assert AnyHolder(value=np.array([-0.0])) == AnyHolder(value=torch.tensor([-0.0]))


def test_zero_size_arrays_keep_dimension_collapse_equality():
    assert AnyHolder(value=np.zeros((0, 3))) == AnyHolder(value=np.zeros((0, 4)))
    assert AnyHolder(value=torch.zeros((0, 3))) == AnyHolder(value=torch.zeros((0, 5)))


def test_grad_and_bfloat16_tensors_compare_by_value():
    plain = torch.tensor([1.5, 2.5])
    grad = torch.tensor([1.5, 2.5], requires_grad=True)
    assert AnyHolder(value=plain) == AnyHolder(value=grad)
    assert AnyHolder(value=torch.tensor([1.5], dtype=torch.bfloat16)) == AnyHolder(
        value=np.array([1.5], dtype=np.float32)
    )


def test_nan_bearing_arrays_compare_unequal_outside_the_finite_domain():
    from_dict(PlainWithArray, {})
    left = PlainWithArray(weights=np.array([np.nan], dtype=np.float32))
    right = PlainWithArray(weights=np.array([np.nan], dtype=np.float32))
    assert left != right


def test_scalar_shortcut_stays_exact_for_numpy_scalars():
    assert AnyHolder(value=np.float64(2**53)) != AnyHolder(value=2**53 + 1)
    assert AnyHolder(value=np.float64(1.5)) == AnyHolder(value=1.5)


def test_canonicalizing_dict_keys_compare_by_plain_form():
    from pathlib import Path  # noqa: PLC0415

    assert AnyHolder(value={Path("x"): 1}) == AnyHolder(value={"x": 1})
    assert AnyHolder(value={"x": 1}) != AnyHolder(value={"y": 1})


def test_negative_bit_tensor_views_compare_by_value():
    view = torch._neg_view(torch.tensor([1.0, -2.0]))
    assert AnyHolder(value=view) == AnyHolder(value=np.array([-1.0, 2.0]))
    assert AnyHolder(value=view) == AnyHolder(value=torch.tensor([-1.0, 2.0]))
    assert AnyHolder(value=view) != AnyHolder(value=np.array([1.0, -2.0]))


def test_tensor_subclasses_compare_by_serialized_value():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    assert AnyHolder(value=param) == AnyHolder(value=torch.tensor([1.0, 2.0]))
    assert AnyHolder(value=param) != AnyHolder(value=torch.tensor([1.0, 3.0]))


def test_array_mapping_keys_under_any_are_rejected_at_load():
    cls = schema_for(Any)
    with pytest.raises(ConfigError) as info:
        from_dict(cls, {"x": {torch.tensor([1, 2]): "v"}})
    assert any("as a mapping key" in i.message for i in info.value.issues)


def test_array_mapping_keys_collect_on_marshal():
    cls = schema_for(Any)
    built = cls(x={torch.tensor([1, 2]): "v"})
    with pytest.raises(ConfigError) as info:
        to_dict(built)
    assert any("cannot serialize Tensor as a mapping key" in i.message for i in info.value.issues)


def test_tuple_mapping_keys_collect_on_marshal():
    cls = schema_for(Any)
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
        "from confingo import from_dict, to_dict\n"
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
        "from confingo import config_hash, from_dict\n"
        "cls = make_dataclass('S', [('a', npt.NDArray[np.float32]), ('b', Annotated[torch.Tensor, torch.int32])])\n"
        "built = from_dict(cls, {'a': [1.5, 2.5], 'b': [1, 2]})\n"
        "print(config_hash(built))\n"
    )
    assert run_python(code) == run_python(code)
