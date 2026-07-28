"""Implicit instantiation of absent dataclass sections.

Absent dataclass fields build from an empty mapping so required leaves surface
at their nested dotted paths; every other undefaulted field, containers
included, stays required. These schemas live at module level so annotations
resolve.
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)

import pytest

from confingo import (
    ConfigError,
    ConfigValue,
)
from confingo.functional import (
    from_dict,
    to_dict,
)


@dataclass(frozen=True)
class AllDefaulted:
    lr: float = 3e-4
    steps: int = 1000


@dataclass(frozen=True)
class RequiredLeaf:
    name: str
    lr: float = 3e-4


@dataclass(frozen=True)
class MidSection:
    inner: RequiredLeaf
    scale: float = 1.0


@dataclass(frozen=True)
class DeepRoot:
    mid: MidSection
    seed: int = 0


@dataclass(frozen=True)
class DefaultedRoot:
    section: AllDefaulted
    seed: int = 0


@dataclass(frozen=True)
class ContainerRoot:
    stages: list[RequiredLeaf]
    lookup: dict[str, AllDefaulted]
    widths: tuple[int, ...]
    tags: set[str]


@dataclass(frozen=True)
class FactoryContainers:
    stages: list[RequiredLeaf] = field(default_factory=list)
    lookup: dict[str, AllDefaulted] = field(default_factory=dict)


@dataclass(frozen=True)
class UnionLeaf:
    seed: int | None


@dataclass(frozen=True)
class OptionalSection:
    section: AllDefaulted | None


@dataclass(frozen=True)
class AnyLeaf:
    extra: ConfigValue


@dataclass(frozen=True)
class Recursive:
    child: Recursive
    depth: int = 0


@dataclass(frozen=True)
class HookedSection:
    count: int = 3

    def __post_init__(self):
        if self.count < 1:
            raise ValueError("count must be >= 1")

    def __validate__(self):
        if self.count > 100:
            yield "count must be <= 100"


@dataclass(frozen=True)
class HookedRoot:
    section: HookedSection


@dataclass(frozen=True)
class AuthoredFactory:
    section: AllDefaulted = field(default_factory=lambda: AllDefaulted(lr=1.0))


@dataclass(frozen=True)
class AuthoredDirect:
    section: AllDefaulted = AllDefaulted(lr=2.0)


def issue_paths(err: ConfigError) -> list[str]:
    return [issue.path for issue in err.issues]


def test_all_defaulted_section_omitted():
    config = from_dict(DefaultedRoot, {})
    assert config.section == AllDefaulted()


def test_required_leaf_hoists_to_nested_path():
    with pytest.raises(ConfigError) as excinfo:
        from_dict(DeepRoot, {})
    assert issue_paths(excinfo.value) == ["mid.inner.name"]


def test_required_leaf_supplied_through_nested_mapping():
    config = from_dict(DeepRoot, {"mid": {"inner": {"name": "run"}}})
    assert config.mid.inner == RequiredLeaf(name="run")
    assert config.mid.scale == 1.0


def test_absent_containers_stay_required():
    with pytest.raises(ConfigError) as excinfo:
        from_dict(ContainerRoot, {})
    assert issue_paths(excinfo.value) == ["stages", "lookup", "widths", "tags"]


def test_factory_containers_build_empty_when_omitted():
    config = from_dict(FactoryContainers, {})
    assert config.stages == []
    assert config.lookup == {}


def test_supplied_container_elements_enforce_required_leaves():
    with pytest.raises(ConfigError) as excinfo:
        from_dict(FactoryContainers, {"stages": [{"lr": 0.1}]})
    assert issue_paths(excinfo.value) == ["stages.0.name"]


def test_union_leaf_stays_required():
    with pytest.raises(ConfigError) as excinfo:
        from_dict(UnionLeaf, {})
    assert issue_paths(excinfo.value) == ["seed"]


def test_optional_section_stays_required():
    with pytest.raises(ConfigError) as excinfo:
        from_dict(OptionalSection, {})
    assert issue_paths(excinfo.value) == ["section"]


def test_any_leaf_stays_required():
    with pytest.raises(ConfigError) as excinfo:
        from_dict(AnyLeaf, {})
    assert issue_paths(excinfo.value) == ["extra"]


def test_self_referential_schema_terminates():
    with pytest.raises(ConfigError) as excinfo:
        from_dict(Recursive, {})
    assert issue_paths(excinfo.value) == ["child.child"]


def test_hooks_run_on_implicit_section():
    config = from_dict(HookedRoot, {})
    assert config.section.count == 3


def test_authored_factory_takes_precedence():
    config = from_dict(AuthoredFactory, {})
    assert config.section == AllDefaulted(lr=1.0)


def test_authored_direct_default_takes_precedence():
    config = from_dict(AuthoredDirect, {})
    assert config.section == AllDefaulted(lr=2.0)


def test_implicit_build_round_trips():
    config = from_dict(DefaultedRoot, {})
    assert from_dict(DefaultedRoot, to_dict(config)) == config


def test_missing_leaves_collected_alongside_other_issues():
    with pytest.raises(ConfigError) as excinfo:
        from_dict(DeepRoot, {"seed": "x", "sed": 1})
    assert issue_paths(excinfo.value) == ["sed", "mid.inner.name", "seed"]
