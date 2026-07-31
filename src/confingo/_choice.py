"""Variant-group registry: the closed set of sections one annotation stands for.

A variant group is a base class standing for a set of config sections, one of
which a config file selects by writing a tag under a key the group names. The
annotation names the group, the file names the variant, and the build constructs
exactly the one class the file selected.

This module holds the registry primitives alone -- the per-group record, the
class-creation bookkeeping, and the generation counter that tells the schema
caches a group's membership moved. ``ConfigChoice``, the base a schema actually
subclasses, lives in ``confingo._node`` beside ``ConfigNode``, which keeps the
import graph acyclic: the schema engine reads this module, and the facade reads
the engine.

Membership is recorded as classes are created, which is when the declaring module
is imported, so a group knows the variants whose modules have been imported.
Declaring a group's variants beside it, in one module, is what keeps that set
complete, and it puts every registration on the thread doing the import.
"""

from __future__ import annotations

import contextlib
from dataclasses import (
    dataclass,
    field,
)
from typing import Any

from confingo._errors import class_label


GROUP_SLOT = "_confingo_choice_group"
"""Attribute a variant-group base carries its own registry record under."""

TAG_SLOT = "_confingo_choice_tag"
"""Attribute a variant carries its own selection string under.

Recorded on the class rather than held only in the registry so it survives class
recreation. ``@dataclass(slots=True)`` builds a replacement class from a copy of
the original's ``__dict__`` and discards the original, which fires
``__init_subclass__`` a second time with none of the class keywords the first
firing saw. The copied recording is what identifies that second firing as a
recreation rather than an untagged subclass.
"""


@dataclass
class ChoiceGroup:
    """The closed set of variants one group base stands for.

    Held on the group base itself, so the record's lifetime is the class's own
    and a group defined in a notebook cell or a schema factory takes its registry
    with it when it is collected.

    Attributes:
      tag_key (str): The mapping key a config file writes the selection under.
      by_tag (dict[str, type]): Selection string mapped to the variant it names.
      tag_of (dict[type, str]): Variant mapped to the selection string it carries.
    """

    tag_key: str
    by_tag: dict[str, type] = field(default_factory=dict)
    tag_of: dict[type, str] = field(default_factory=dict)

    def tags(self) -> str:
        """Render the registered selection strings for a message, in sorted order.

        Returns:
          str: The strings quoted and joined with ``|``, or a phrase naming the
            empty registry when the group has no variants.
        """
        if len(self.by_tag) == 0:
            return "no registered variants"
        return " | ".join(repr(tag) for tag in sorted(self.by_tag))


_generation = 0
"""How many times a variant has been registered, process-wide.

The schema caches store a whole transitive validation result on the entry class,
and that result depends on which variants are registered: a group validated
before a variant's module imported saw a different set than one validated after.
Comparing the generation a cached result was computed at against this counter is
what makes those caches recompute when membership moves, without tracking which
entry classes reach which groups.
"""


def registry_generation() -> int:
    """Read the current registration generation.

    Returns:
      int: A counter incremented on every variant registration.
    """
    return _generation


def group_record(config_cls: Any) -> ChoiceGroup | None:
    """Read the registry record a class carries as a group base of its own.

    The read is scoped to the class's own namespace, so a variant answers with
    None and the group it belongs to answers with the record.

    Args:
      config_cls (Any): The class to inspect.

    Returns:
      ChoiceGroup | None: The record when this class is a group base, else None.
    """
    if not isinstance(config_cls, type):
        return None
    entry = type.__getattribute__(config_cls, "__dict__").get(GROUP_SLOT)
    return entry if isinstance(entry, ChoiceGroup) else None


def owning_group(config_cls: type[Any]) -> type[Any] | None:
    """Find the group base a class descends from, skipping the class itself.

    Args:
      config_cls (type[Any]): The class to inspect.

    Returns:
      type[Any] | None: The nearest base carrying a group record, or None when
        this class descends from no group.
    """
    for base in config_cls.__mro__[1:]:
        if group_record(base) is not None:
            return base
    return None


def is_group(config_cls: Any) -> bool:
    """Report whether a class is a variant-group base.

    Args:
      config_cls (Any): The class to inspect.

    Returns:
      bool: True when the class carries a group record of its own.
    """
    return group_record(config_cls) is not None


def variant_tag(config_cls: Any) -> tuple[type[Any], str] | None:
    """Resolve a class to the group it belongs to and the tag it carries.

    Args:
      config_cls (Any): The class to inspect.

    Returns:
      tuple[type[Any], str] | None: The group base and the selection string, or
        None when the class is not a registered variant.
    """
    if not isinstance(config_cls, type):
        return None
    group = owning_group(config_cls)
    if group is None:
        return None
    record = group_record(group)
    if record is None:
        return None
    tag = record.tag_of.get(config_cls)
    return None if tag is None else (group, tag)


def recorded_tag(config_cls: type[Any]) -> str | None:
    """Read the selection string a class recorded for itself at creation.

    Args:
      config_cls (type[Any]): The class to inspect.

    Returns:
      str | None: The recorded string, or None when the class recorded none.
    """
    entry = type.__getattribute__(config_cls, "__dict__").get(TAG_SLOT)
    return entry if isinstance(entry, str) else None


def inherited_variant(config_cls: type[Any]) -> type[Any] | None:
    """Find a registered variant among a class's bases.

    A variant is a leaf. The whole MRO answers rather than the part ahead of the
    group, since a second base can carry a variant of another group entirely: a
    class reaching one group directly while inheriting a variant of another would
    carry that variant's fields and register under one string while answering to
    two lineages.

    Args:
      config_cls (type[Any]): The class being created.

    Returns:
      type[Any] | None: The variant it inherits, or None when it inherits none.
    """
    for base in config_cls.__mro__[1:]:
        if variant_tag(base) is not None:
            return base
    return None


def groups_in_mro(config_cls: type[Any]) -> list[type[Any]]:
    """Collect every variant-group base a class descends from.

    Args:
      config_cls (type[Any]): The class being created.

    Returns:
      list[type[Any]]: The group bases, nearest first.
    """
    return [base for base in config_cls.__mro__[1:] if group_record(base) is not None]


def superseded_holder(holder: type[Any]) -> bool:
    """Report whether a registered variant gives its selection string up.

    Registration is recorded as the class is created, which is before the
    ``@dataclass`` decorator wrapping the declaration runs, so a decorator that
    raises on the body it was handed leaves the part-made class holding the
    string. Two markers together name that class. It carries fields of its own,
    which the decorator records first, and no ``__init__`` of its own, which the
    decorator installs once the body it read stands; a class between the two is
    one the decorator started and left, and a group builds a section by calling
    it. Its string returns to the group for the corrected declaration.

    Every other holder keeps its string against every later claim, whatever name
    that claim carries. A finished variant is live, and so is a variant declaring
    fields of its own that inherits its group's initializer, so the entry stays
    where marshal finds the string its instances export.

    The pair is read together for that reason: each marker alone answers for a
    live variant, and a claim on a live string belongs to the holder.

    Args:
      holder (type[Any]): The variant already registered under the string.

    Returns:
      bool: True when a new claim takes the entry over.
    """
    namespace = type.__getattribute__(holder, "__dict__")
    return "__dataclass_fields__" in namespace and "__init__" not in namespace


def declare_group(config_cls: type[Any], tag_key: str) -> None:
    """Record a class as a variant-group base.

    Args:
      config_cls (type[Any]): The class being created.
      tag_key (str): The mapping key config files write the selection under.
    """
    setattr(config_cls, GROUP_SLOT, ChoiceGroup(tag_key))


def register_variant(group: type[Any], config_cls: type[Any], tag: str) -> None:
    """Record a class as the variant one selection string names.

    A replacement of the registered holder takes the entry over, and the holder
    it replaced is dropped, so the surviving class is the one dispatch reaches
    and the one marshal finds a tag for.

    Args:
      group (type[Any]): The group base owning the registry.
      config_cls (type[Any]): The variant class being created.
      tag (str): The selection string the variant carries.
    """
    global _generation  # noqa: PLW0603 (one process-wide counter the schema caches read)

    record = group_record(group)
    if record is None:
        return
    holder = record.by_tag.get(tag)
    if holder is not None and holder is not config_cls:
        record.tag_of.pop(holder, None)
    record.by_tag[tag] = config_cls
    record.tag_of[config_cls] = tag
    with contextlib.suppress(TypeError, AttributeError):
        setattr(config_cls, TAG_SLOT, tag)
    _generation += 1


# ---------------------------------------------------------------------------
# Class-creation messages
# ---------------------------------------------------------------------------


def untagged_variant_message(config_cls: type[Any], group: type[Any]) -> str:
    """Build the rejection for a group subclass carrying no selection string.

    Args:
      config_cls (type[Any]): The subclass being created.
      group (type[Any]): The group base it descends from.

    Returns:
      str: The rejection naming the class, the group, and the keyword to write.
    """
    return (
        f"{class_label(config_cls)} subclasses the variant group {class_label(group)} without naming its "
        f'selection string; write class {config_cls.__name__}({group.__name__}, tag="...") so a config file '
        f"can select it, and put fields shared by every variant on {class_label(group)} itself."
    )


def duplicate_tag_message(config_cls: type[Any], group: type[Any], tag: str, holder: type[Any]) -> str:
    """Build the rejection for a selection string another variant already holds.

    Args:
      config_cls (type[Any]): The subclass being created.
      group (type[Any]): The group base owning the registry.
      tag (str): The contested selection string.
      holder (type[Any]): The variant already registered under it.

    Returns:
      str: The rejection naming both classes and the remedy.
    """
    return (
        f"{class_label(config_cls)} claims the selection string {tag!r} in the variant group "
        f"{class_label(group)}, which {class_label(holder)} already carries; give one of them a "
        f"different tag so a config file naming {tag!r} selects one class."
    )


def variant_of_variant_message(config_cls: type[Any], group: type[Any], holder: type[Any]) -> str:
    """Build the rejection for a variant declared under another variant.

    Args:
      config_cls (type[Any]): The subclass being created.
      group (type[Any]): The group base it descends from.
      holder (type[Any]): The registered variant standing in between.

    Returns:
      str: The rejection naming both classes and the group to subclass instead.
    """
    resolved = variant_tag(holder)
    owner = group if resolved is None else resolved[0]
    return (
        f"{class_label(config_cls)} subclasses {class_label(holder)}, which is already a variant of the "
        f"group {class_label(owner)}, and a variant is a leaf; write class {config_cls.__name__}"
        f'({group.__name__}, tag="...") to add it to the group directly, and move fields the two share '
        f"onto {class_label(group)}."
    )


def two_groups_message(config_cls: type[Any], groups: list[type[Any]]) -> str:
    """Build the rejection for a class descending from more than one group.

    Args:
      config_cls (type[Any]): The subclass being created.
      groups (list[type[Any]]): The group bases it descends from.

    Returns:
      str: The rejection naming the groups and the shape to write instead.
    """
    named = ", ".join(class_label(group) for group in groups)
    return (
        f"{class_label(config_cls)} descends from the variant groups {named}, and a config section carries "
        f"one selection under one key; subclass the single group this section belongs to, and hold the "
        f"other as a field of its own."
    )


def nested_group_message(config_cls: type[Any], group: type[Any]) -> str:
    """Build the rejection for a group declared inside another group.

    Args:
      config_cls (type[Any]): The subclass being created.
      group (type[Any]): The group base it descends from.

    Returns:
      str: The rejection naming the class, the enclosing group, and the remedy.
    """
    return (
        f"{class_label(config_cls)} declares tag_key= inside the variant group {class_label(group)}, and a "
        f'variant is a leaf; write class {config_cls.__name__}({group.__name__}, tag="...") to add it to '
        f"{class_label(group)}, or declare a separate group on ConfigChoice directly."
    )


def missing_tag_key_message(config_cls: type[Any]) -> str:
    """Build the rejection for a ``ConfigChoice`` subclass declaring no key.

    Args:
      config_cls (type[Any]): The subclass being created.

    Returns:
      str: The rejection naming the class and the keyword a group declares.
    """
    return (
        f"{class_label(config_cls)} subclasses ConfigChoice without naming the key its config sections "
        f'select a variant under; write class {config_cls.__name__}(ConfigChoice, tag_key="...") to declare '
        f"a variant group, then give each variant class {config_cls.__name__} a tag= of its own."
    )


def keyword_type_message(config_cls: type[Any], keyword: str, value: Any) -> str:
    """Build the rejection for a class keyword that is not a non-empty string.

    Args:
      config_cls (type[Any]): The subclass being created.
      keyword (str): The keyword name, ``tag`` or ``tag_key``.
      value (Any): The value the declaration supplied.

    Returns:
      str: The rejection naming the keyword and the form to write.
    """
    return (
        f"{class_label(config_cls)} declares {keyword}={value!r}, and {keyword} names a key a config file "
        f'writes; write a non-empty str, as {keyword}="...". A file carries the exact class str, which is '
        f"what a load reads back and compares, so the declaration names that class too."
    )


def both_keywords_message(config_cls: type[Any]) -> str:
    """Build the rejection for a class declaring both keywords at once.

    Args:
      config_cls (type[Any]): The subclass being created.

    Returns:
      str: The rejection naming the two roles and how to write each.
    """
    return (
        f"{class_label(config_cls)} declares both tag_key= and tag=, which name the two sides of a variant "
        f"group: tag_key= declares the group and the key its sections select under, and tag= adds one "
        f"variant to a group. Write whichever one this class is."
    )
