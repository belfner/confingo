"""Static conformance probe for the public typing surface.

Every assertion here is checked by the type checker rather than at run time: the
probe bodies sit under ``TYPE_CHECKING`` and are never executed, so the file
paths they name refer to nothing on disk. ``pyrefly check`` covers the whole
repository, so a regression in an annotation, an overload, or a return type
fails the same gate the rest of the suite runs under.

The surface covered is what an installed package exposes: each free function
against a plain dataclass and against a node, each accessor operation from the
class route and the value route, the section route through a field, the issue
and error attributes, and generic helpers written over ``type[ConfigT]``.
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    assert_type,
)

import confingo
from confingo import (
    ConfigError,
    ConfigIssue,
    ConfigNode,
    config_equal,
    config_hash,
    dumps_json,
    dumps_yaml,
    from_dict,
    from_file,
    load_json,
    load_yaml,
    save_json,
    save_yaml,
    to_dict,
    to_file,
    validate,
)


@dataclass
class Optimizer:
    name: str = "adamw"
    lr: float = 1e-3


@dataclass
class Plain:
    optimizer: Optimizer
    tags: list[str] = field(default_factory=list)


@dataclass
class Trainer(ConfigNode):
    lr: float = 1e-3


@dataclass
class Node(ConfigNode):
    optimizer: Optimizer
    trainer: Trainer
    stages: list[Trainer] = field(default_factory=list)
    seed: int = 0


@dataclass
class SubNode(Node):
    extra: int = 1


if TYPE_CHECKING:
    from pathlib import Path

NodeT = TypeVar("NodeT", bound=ConfigNode)
ConfigT = TypeVar("ConfigT")


def build_any(config_cls: type[ConfigT], data: dict[str, Any]) -> ConfigT:
    """Build any dataclass through the free function, keeping the caller's type.

    Args:
      config_cls (type[ConfigT]): The dataclass to build.
      data (dict[str, Any]): Nested mapping of config values.

    Returns:
      ConfigT: The constructed config object.
    """
    return from_dict(config_cls, data)


def build_node(config_cls: type[NodeT]) -> NodeT:
    """Build a node through the class route, keeping the caller's type.

    Args:
      config_cls (type[NodeT]): The node class to build.

    Returns:
      NodeT: The constructed config object.
    """
    return config_cls.cfg.from_dict({})


def fingerprint(node: NodeT) -> str:
    """Fingerprint any node through the value route.

    Args:
      node (NodeT): The config object to fingerprint.

    Returns:
      str: The truncated digest.
    """
    return node.cfg.hash()


def export(node: NodeT) -> Any:
    """Render any node's subtree through the value route.

    Args:
      node (NodeT): The config object to render.

    Returns:
      Any: The plain-data form.
    """
    return node.cfg.to_dict()


def test_the_generic_helpers_keep_the_caller_s_class():
    """The helpers above are type-checked declarations that also run."""
    built = build_node(SubNode)
    assert isinstance(built, SubNode)
    assert build_any(Plain, {}) == from_dict(Plain, {})
    assert fingerprint(built) == config_hash(built)
    assert export(built) == to_dict(built)


def test_the_probed_file_operations_round_trip(tmp_path: Path):
    """The operations the probes annotate carry the same values at run time."""
    built = Node.cfg.from_dict({})
    written = built.cfg.save_json(tmp_path / "c.json")
    assert load_json(Node, written) == built
    assert from_file(Node, written) == built
    assert save_yaml(built, tmp_path / "c.yaml").suffix == ".yaml"
    assert load_yaml(Node, tmp_path / "c.yaml") == built
    assert to_file(built, tmp_path / "out.json") == tmp_path / "out.json"
    assert config_equal(built, Node.cfg.from_dict({})) is True
    assert dumps_json(built).endswith("\n")
    assert dumps_yaml(built).endswith("\n")
    validate(Node)


if TYPE_CHECKING:

    def _free_functions_over_a_plain_dataclass() -> None:
        """A plain dataclass root answers with its own class from every builder."""
        assert_type(from_dict(Plain, {}), Plain)
        assert_type(from_dict(Plain, {}, context="sweep"), Plain)
        assert_type(load_json(Plain, "c.json"), Plain)
        assert_type(load_json(Plain, Path("c.json")), Plain)
        assert_type(load_yaml(Plain, "c.yaml"), Plain)
        assert_type(from_file(Plain, "c.json"), Plain)
        assert_type(validate(Plain), None)
        assert_type(validate(Plain, context="schema"), None)

        config = from_dict(Plain, {})
        assert_type(to_dict(config), Any)
        assert_type(dumps_json(config), str)
        assert_type(dumps_json(config, indent=4), str)
        assert_type(dumps_yaml(config, indent=4, sort_keys=True), str)
        assert_type(save_json(config, "c.json"), Path)
        assert_type(save_yaml(config, Path("c.yaml")), Path)
        assert_type(to_file(config, "c.json"), Path)
        assert_type(config_hash(config), str)
        assert_type(config_hash(config, length=8), str)
        assert_type(config_equal(config, config), bool)

    def _the_class_route() -> None:
        """A builder reached through the class answers with that class."""
        assert_type(Node.cfg.from_dict({}), Node)
        assert_type(Node.cfg.from_dict({}, context="sweep"), Node)
        assert_type(Node.cfg.load_json("c.json"), Node)
        assert_type(Node.cfg.load_yaml(Path("c.yaml")), Node)
        assert_type(Node.cfg.from_file("c.json"), Node)
        assert_type(Node.cfg.validate(), None)
        assert_type(Node.cfg.validate(context="sweep template"), None)
        assert_type(SubNode.cfg.from_dict({}), SubNode)
        assert_type(SubNode.cfg.load_json("c.json"), SubNode)
        assert_type(from_dict(SubNode, {}), SubNode)

    def _the_value_route() -> None:
        """An operation reached through a value answers over that value's class."""
        node = Node.cfg.from_dict({})
        sub = SubNode.cfg.from_dict({})
        assert_type(node.cfg.to_dict(), Any)
        assert_type(node.cfg.dumps_json(indent=4), str)
        assert_type(node.cfg.dumps_yaml(sort_keys=True), str)
        assert_type(node.cfg.save_json("c.json"), Path)
        assert_type(node.cfg.save_yaml("c.yaml", indent=4), Path)
        assert_type(node.cfg.to_file(Path("c.json")), Path)
        assert_type(node.cfg.hash(), str)
        assert_type(node.cfg.hash(length=8), str)
        assert_type(node.cfg.validate(), None)
        assert_type(node.cfg.from_dict({}), Node)
        assert_type(sub.cfg.from_dict({}), SubNode)
        assert_type(sub.cfg.hash(), str)

    def _the_section_route() -> None:
        """A section reached through a field carries its own class and accessor."""
        node = Node.cfg.from_dict({})
        assert_type(node.optimizer, Optimizer)
        assert_type(node.optimizer.lr, float)
        assert_type(config_hash(node.optimizer), str)

        assert_type(node.trainer, Trainer)
        assert_type(node.trainer.cfg.from_dict({}), Trainer)
        assert_type(node.trainer.cfg.to_dict(), Any)
        assert_type(node.trainer.cfg.hash(), str)
        assert_type(node.trainer.cfg.save_json("c.json"), Path)
        assert_type(node.trainer.cfg.validate(), None)

        assert_type(node.stages[0], Trainer)
        assert_type(node.stages[0].cfg.from_dict({}), Trainer)
        assert_type(node.stages[0].cfg.hash(), str)

    def _a_value_operation_belongs_to_the_value_route() -> None:
        """The class route offers the operations a class can answer.

        A value operation names the instance form, which the type checker sees
        as the class facade carrying the builders and ``validate`` alone. The
        same call raises the naming ``TypeError`` at run time, asserted in
        ``test_config_node.py``.

        Each suppression below is load-bearing in both directions: removing one
        reports the missing attribute, and widening the class facade back to the
        value operations reports the suppression as unused.
        """
        # pyrefly: ignore[missing-attribute]
        Node.cfg.to_dict()
        # pyrefly: ignore[missing-attribute]
        Node.cfg.dumps_json()
        # pyrefly: ignore[missing-attribute]
        Node.cfg.dumps_yaml()
        # pyrefly: ignore[missing-attribute]
        Node.cfg.save_json("c.json")
        # pyrefly: ignore[missing-attribute]
        Node.cfg.save_yaml("c.yaml")
        # pyrefly: ignore[missing-attribute]
        Node.cfg.to_file("c.json")
        # pyrefly: ignore[missing-attribute]
        Node.cfg.hash()

    def _the_error_surface() -> None:
        """The raised error carries typed issues."""
        try:
            from_dict(Plain, {})
        except ConfigError as exc:
            assert_type(exc.issues, tuple[ConfigIssue, ...])
            assert_type(exc.context, str)
            assert_type(exc.issues[0].path, str)
            assert_type(exc.issues[0].message, str)
        assert_type(ConfigError.single("boom", context="config"), ConfigError)
        assert_type(confingo.__version__, str)
