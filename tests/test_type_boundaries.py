from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from decimal import Decimal
from typing import (
    Any,
    TypedDict,
)

import pytest

from confingo import (
    ConfigError,
    from_dict,
    to_dict,
)


@dataclass
class HasDecimal:
    amount: Decimal = Decimal(0)


def test_unsupported_leaf_type_rejected_on_load():
    with pytest.raises(ConfigError) as info:
        from_dict(HasDecimal, {"amount": "1.5"})
    assert any("unsupported field type Decimal" in issue.message for issue in info.value.issues)


class Section(TypedDict):
    a: int


@dataclass
class HasTypedDict:
    section: Section = field(default_factory=lambda: {"a": 1})


def test_typeddict_section_rejected_without_crashing():
    with pytest.raises(ConfigError) as info:
        from_dict(HasTypedDict, {"section": {"a": 2}})
    assert any("unsupported field type Section" in issue.message for issue in info.value.issues)


@dataclass
class AnyField:
    x: Any = None


def test_to_dict_rejects_unserializable_value():
    cfg = AnyField(x=Decimal("1.5"))
    with pytest.raises(ConfigError) as info:
        to_dict(cfg)
    assert "Decimal" in str(info.value)


def test_any_field_still_passes_plain_data():
    cfg = AnyField(x=[1, "two", {"three": 3}])
    assert to_dict(cfg) == {"x": [1, "two", {"three": 3}]}
