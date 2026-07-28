"""Tests for reading a class's own annotations under either evaluation mode.

A schema class carries its annotations one of two ways: as source text, under a
module's postponed evaluation, or as a lazily evaluated declaration. The guards
that read a declaration -- the reserved-name collision on a node, and the
missing-``@dataclass`` remedy -- ask only which names a class declares, so both
storage modes answer them the same way.

The classes here are compiled from source without postponed evaluation, which is
what puts a module on the lazily evaluated route, so the guards are exercised on
both routes whatever the module holding this test declares.
"""

from __future__ import annotations

import itertools
import sys
from types import ModuleType
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterator,
    )

from confingo import ConfigError
from confingo._schema import own_annotations
from confingo.functional import from_dict


type SchemaBuilder = Callable[[str], ModuleType]

_PREAMBLE = "from dataclasses import dataclass, field\n\nfrom confingo import ConfigNode\n\n"

_counter = itertools.count()


@pytest.fixture
def build() -> Iterator[SchemaBuilder]:
    """Provide a builder that turns schema source into a real, importable module.

    Registering the module under its own name in ``sys.modules`` is what lets
    ``get_type_hints`` resolve an annotation against the namespace that declared
    it, matching how a schema written in a file resolves. Compiling with
    ``dont_inherit`` gives the source its own future statements, which is what
    puts the built classes on the interpreter's own annotation route rather than
    the postponed evaluation this test module declares. Each module is removed
    once the test ends.

    Yields:
      SchemaBuilder: A callable taking module source and returning the built module.
    """
    built: list[str] = []

    def _build(body: str) -> ModuleType:
        name = f"_confingo_schema_under_test_{next(_counter)}"
        module = ModuleType(name)
        sys.modules[name] = module
        built.append(name)
        code = compile(_PREAMBLE + body, f"<{name}>", "exec", dont_inherit=True)
        exec(code, module.__dict__)
        return module

    yield _build
    for name in built:
        sys.modules.pop(name, None)


# --- the reader itself ---------------------------------------------------------


def test_own_annotations_reads_the_names_a_class_declares(build: SchemaBuilder):
    module = build("@dataclass\nclass Section:\n    lr: float = 0.1\n    steps: int = 10\n")
    assert set(own_annotations(module.Section)) == {"lr", "steps"}


def test_own_annotations_holds_a_subclass_to_its_own_declarations(build: SchemaBuilder):
    module = build(
        "@dataclass\nclass Parent:\n    lr: float = 0.1\n\n@dataclass\nclass Child(Parent):\n    steps: int = 10\n"
    )
    assert set(own_annotations(module.Child)) == {"steps"}


def test_own_annotations_reads_a_class_whose_annotation_names_a_later_class(build: SchemaBuilder):
    """A declaration is read as written, so a name bound further down resolves later."""
    module = build(
        "@dataclass\n"
        "class Root:\n"
        '    later: "Later" = field(default_factory=lambda: Later())\n'
        "\n"
        "@dataclass\n"
        "class Later:\n"
        "    value: int = 0\n"
    )
    assert set(own_annotations(module.Root)) == {"later"}


# --- the guards that read a declaration ----------------------------------------


@pytest.mark.parametrize(
    ("declaration", "label"),
    [
        ("    cfg: str", "bare annotation"),
        ("    cfg: str = 'a'", "annotation with a default"),
    ],
)
def test_a_node_declaring_the_accessor_is_rejected_at_class_creation(
    build: SchemaBuilder, declaration: str, label: str
):
    """The collision is reported where the class is written, whichever way it stores annotations."""
    with pytest.raises(ConfigError) as error:
        build(f"@dataclass\nclass Shadowing(ConfigNode):\n{declaration}\n")
    assert "Shadowing.cfg is declared as a field" in error.value.issues[0].message


def test_an_undecorated_section_names_the_decorator_as_its_remedy(build: SchemaBuilder):
    module = build("class Section:\n    lr: float = 0.1\n\n@dataclass\nclass Root:\n    section: Section\n")
    with pytest.raises(ConfigError) as error:
        from_dict(module.Root, {"section": {"lr": 0.5}})
    assert "Declare it with @dataclass" in error.value.issues[0].message


def test_a_node_with_a_forward_reference_builds_and_round_trips(build: SchemaBuilder):
    """Reading a declaration leaves it unevaluated, so a pending name stays pending."""
    module = build(
        "@dataclass\n"
        "class Root(ConfigNode):\n"
        '    later: "Later" = field(default_factory=lambda: Later())\n'
        "\n"
        "@dataclass\n"
        "class Later:\n"
        "    value: int = 0\n"
    )
    built = from_dict(module.Root, {"later": {"value": 7}})
    assert built.later.value == 7
    assert built.cfg.to_dict() == {"later": {"value": 7}}
