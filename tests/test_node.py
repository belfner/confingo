from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import pytest

from confingo import (
    ConfigError,
    config_hash,
    dumps_json,
    dumps_yaml,
    to_dict,
)
from confingo._schema import _SCHEMA_CACHE
from tests.schemas import (
    HOOK_CALLS,
    Device,
    NodeBadTree,
    NodeDerived,
    NodeLeaf,
    NodeLeafTwin,
    NodeMid,
    NodeTree,
    PlainBadTree,
    PlainTree,
    RootConfig,
    Trainer,
    UnhashedSection,
)


if TYPE_CHECKING:
    from pathlib import Path


# The mapping below builds both parallel trees: PlainTree nests ordinary
# dataclasses, NodeTree nests ConfigNode subclasses, and their fields match.
VALID_DATA = {
    "mid": {"leaf": {"name": "adamw", "lr": 0.01}, "tag": "outer"},
    "items": [{"name": "sgd"}, {"name": "adamw", "lr": 0.5}],
    "lookup": {"first": {"name": "sgd", "lr": 0.25}},
    "maybe": {"name": "adamw"},
    "choice": {"kind": "alt", "scale": 2.5},
    "extra": {"free": [1, 2], "nested": {"k": "v"}},
    "frozen_leaf": {"w": 4},
    "slots_leaf": {"s": 5},
    "note": "carried",
    "weight": 0.5,
    "seed": 7,
}

INVALID_DATA = {
    "mid": {"leaf": {"name": "rmsprop", "lr": "fast"}, "tag": 3},
    "items": [{"name": "sgd", "lr": -1.0}],
    "unknown": 1,
}


# ---------------------------------------------------------------------------
# Facade parity
# ---------------------------------------------------------------------------


def test_from_dict_classmethod():
    cfg = RootConfig.from_dict({"device": "cuda", "trainer": {"lr": 1e-4}})
    assert isinstance(cfg, RootConfig)
    assert cfg.device is Device.CUDA
    assert cfg.trainer.lr == 1e-4


def test_from_dict_collects_issues():
    with pytest.raises(ConfigError):
        RootConfig.from_dict({"device": "tpu"})


def test_to_dict_method_matches_free_function():
    cfg = RootConfig(device=Device.CUDA, seed=3)
    assert cfg.to_dict() == to_dict(cfg)


def test_dumps_json_method_matches_free_function():
    cfg = RootConfig(trainer=Trainer(lr=1e-4))
    assert cfg.dumps_json() == dumps_json(cfg)


def test_config_hash_method_matches_free_function():
    cfg = RootConfig(seed=1)
    assert cfg.config_hash() == config_hash(cfg)
    assert len(cfg.config_hash(length=8)) == 8


def test_save_load_json_methods_round_trip(tmp_path: Path):
    cfg = RootConfig(device=Device.CUDA, seed=7)
    path = cfg.save_json(tmp_path / "config.json")
    assert RootConfig.load_json(path) == cfg


# ---------------------------------------------------------------------------
# Node attachment is inert for every engine output
# ---------------------------------------------------------------------------


def test_node_attachment_preserves_plain_output():
    """Nesting nodes rather than plain dataclasses leaves exported data unchanged."""
    assert NodeTree.from_dict(VALID_DATA).to_dict() == PlainTree.from_dict(VALID_DATA).to_dict()


def test_node_attachment_preserves_digest_and_serialized_forms():
    """Nesting nodes leaves the digest and both serialized forms unchanged."""
    node_tree = NodeTree.from_dict(VALID_DATA)
    plain_tree = PlainTree.from_dict(VALID_DATA)
    assert node_tree.config_hash() == plain_tree.config_hash()
    assert node_tree.dumps_json() == plain_tree.dumps_json()
    assert node_tree.dumps_yaml() == plain_tree.dumps_yaml()


def test_node_attachment_preserves_runtime_types():
    """Nesting nodes leaves the rebuilt container and value types unchanged."""
    node_tree = NodeTree.from_dict(VALID_DATA)
    plain_tree = PlainTree.from_dict(VALID_DATA)
    assert type(node_tree.items) is type(plain_tree.items)
    assert type(node_tree.lookup) is type(plain_tree.lookup)
    assert node_tree.mid.leaf.lr == plain_tree.mid.leaf.lr
    assert node_tree.maybe is not None
    assert node_tree.maybe.name == "adamw"


def test_node_attachment_preserves_issue_reports():
    """Nesting nodes leaves every issue path, message, and context unchanged."""
    with pytest.raises(ConfigError) as node_error:
        NodeTree.from_dict(INVALID_DATA, context="same source")
    with pytest.raises(ConfigError) as plain_error:
        PlainTree.from_dict(INVALID_DATA, context="same source")

    def report(error: ConfigError) -> list[tuple[str, str]]:
        return [(issue.path, issue.message) for issue in error.issues]

    assert report(node_error.value) == report(plain_error.value)
    assert node_error.value.context == plain_error.value.context
    assert ("mid.leaf.lr", "expected float, got str") in report(node_error.value)


def test_node_attachment_preserves_hook_counts():
    """Each section's construction hooks run once per kept value, either way."""
    HOOK_CALLS.clear()
    NodeTree.from_dict(VALID_DATA)
    node_calls = list(HOOK_CALLS)

    HOOK_CALLS.clear()
    PlainTree.from_dict(VALID_DATA)
    plain_calls = list(HOOK_CALLS)

    assert len(node_calls) == len(plain_calls)
    # mid.leaf, two items, one lookup value, and maybe: five kept leaves.
    assert node_calls.count("node_leaf_post_init") == 5
    assert node_calls.count("node_leaf_validate") == 5


def test_node_attachment_preserves_schema_preflight_errors():
    """A preflight error reports identically whether the bad section is a node."""
    with pytest.raises(ConfigError) as node_error:
        NodeBadTree.from_dict({})
    with pytest.raises(ConfigError) as plain_error:
        PlainBadTree.from_dict({})
    assert [(i.path, i.message) for i in node_error.value.issues] == [
        (i.path, i.message) for i in plain_error.value.issues
    ]
    assert node_error.value.issues[0].path == "bad.handler"


def test_node_attachment_preserves_union_branch_selection():
    """The union member selected, and its rebuilt type, are unchanged."""
    node_tree = NodeTree.from_dict(VALID_DATA)
    plain_tree = PlainTree.from_dict(VALID_DATA)
    assert type(node_tree.choice).__name__ == "NodeAlt"
    assert type(plain_tree.choice).__name__ == "PlainAlt"
    assert to_dict(node_tree.choice) == to_dict(plain_tree.choice)


def test_node_attachment_preserves_any_and_field_projections():
    """``Any`` data, and the compare/hash/init projections, are unchanged."""
    node_tree = NodeTree.from_dict(VALID_DATA)
    plain_tree = PlainTree.from_dict(VALID_DATA)
    assert node_tree.extra == plain_tree.extra == {"free": [1, 2], "nested": {"k": "v"}}
    assert node_tree.derived == plain_tree.derived == 14

    # compare=False leaves equality untouched; hash=False leaves the digest untouched.
    assert NodeTree.from_dict({**VALID_DATA, "note": "other"}) == node_tree
    assert PlainTree.from_dict({**VALID_DATA, "note": "other"}) == plain_tree
    assert NodeTree.from_dict({**VALID_DATA, "weight": 9.0}).config_hash() == node_tree.config_hash()
    assert PlainTree.from_dict({**VALID_DATA, "weight": 9.0}).config_hash() == plain_tree.config_hash()


def test_node_attachment_preserves_frozen_and_slotted_sections():
    """Frozen and slotted nested sections rebuild the same either way."""
    node_tree = NodeTree.from_dict(VALID_DATA)
    plain_tree = PlainTree.from_dict(VALID_DATA)
    assert node_tree.frozen_leaf.w == plain_tree.frozen_leaf.w == 4
    assert node_tree.slots_leaf.s == plain_tree.slots_leaf.s == 5
    assert to_dict(node_tree.frozen_leaf) == to_dict(plain_tree.frozen_leaf)
    assert to_dict(node_tree.slots_leaf) == to_dict(plain_tree.slots_leaf)


# ---------------------------------------------------------------------------
# Nested nodes as entry points
# ---------------------------------------------------------------------------


def test_nested_node_methods_match_free_functions():
    cfg = NodeTree.from_dict(VALID_DATA)
    assert cfg.mid.leaf.to_dict() == to_dict(cfg.mid.leaf)
    assert cfg.mid.leaf.config_hash() == config_hash(cfg.mid.leaf)
    assert cfg.mid.dumps_json() == dumps_json(cfg.mid)
    assert cfg.mid.dumps_yaml() == dumps_yaml(cfg.mid)


def test_nested_node_round_trips_through_its_own_class():
    cfg = NodeTree.from_dict(VALID_DATA)
    assert NodeLeaf.from_dict(cfg.mid.leaf.to_dict()) == cfg.mid.leaf
    assert NodeMid.from_dict(cfg.mid.to_dict()) == cfg.mid


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_nested_node_file_round_trip(tmp_path: Path, suffix: str):
    cfg = NodeTree.from_dict(VALID_DATA)
    path = cfg.mid.to_file(tmp_path / f"mid{suffix}")
    assert NodeMid.from_file(path) == cfg.mid


def test_nested_snapshot_loaded_through_enclosing_class_reports_its_keys(tmp_path: Path):
    cfg = NodeTree.from_dict(VALID_DATA)
    path = cfg.mid.save_json(tmp_path / "mid.json")
    with pytest.raises(ConfigError) as error:
        NodeTree.load_json(path)
    paths = {issue.path for issue in error.value.issues}
    assert "leaf" in paths
    assert "tag" in paths


def test_issue_paths_are_relative_to_the_entry_node():
    with pytest.raises(ConfigError) as outer:
        NodeTree.from_dict({"mid": {"leaf": {"name": "adamw", "lr": "fast"}}})
    with pytest.raises(ConfigError) as inner:
        NodeLeaf.from_dict({"name": "adamw", "lr": "fast"})

    assert [issue.path for issue in outer.value.issues] == ["mid.leaf.lr"]
    assert [issue.path for issue in inner.value.issues] == ["lr"]


def test_nested_node_entry_keeps_the_callers_context():
    with pytest.raises(ConfigError) as error:
        NodeLeaf.from_dict({"lr": 1.0}, context="optimizer override")
    assert error.value.context == "optimizer override"


# ---------------------------------------------------------------------------
# Hash scope
# ---------------------------------------------------------------------------


def test_enclosing_hash_excludes_an_unhashed_section():
    """An enclosing ``hash=False`` field drops the section from the enclosing digest."""
    left = UnhashedSection(inner=NodeLeaf(name="adamw", lr=0.1))
    right = UnhashedSection(inner=NodeLeaf(name="sgd", lr=0.9))
    assert left.config_hash() == right.config_hash()
    assert left.inner.config_hash() != right.inner.config_hash()


def test_identical_projections_share_a_digest_across_classes():
    """The canonical JSON carries values rather than class identity."""
    leaf = NodeLeaf(name="adamw", lr=0.5)
    twin = NodeLeafTwin(name="adamw", lr=0.5)
    assert type(leaf) is not type(twin)
    assert leaf.config_hash() == twin.config_hash()
    # The same content, compared rather than fingerprinted, stays class-scoped.
    assert leaf != twin


# ---------------------------------------------------------------------------
# Cache warm order and inheritance
# ---------------------------------------------------------------------------


def test_schema_cache_warm_order_does_not_change_reports():
    """Validating inner-first and outer-first yields the same objects and issues."""

    def build() -> tuple[Any, list[tuple[str, str]]]:
        built = NodeTree.from_dict(VALID_DATA).to_dict()
        with pytest.raises(ConfigError) as error:
            NodeTree.from_dict(INVALID_DATA)
        return built, [(issue.path, issue.message) for issue in error.value.issues]

    for entry in (NodeLeaf, NodeMid, NodeTree):
        _SCHEMA_CACHE.pop(entry, None)
    NodeLeaf.from_dict({"name": "adamw"})
    inner_first = build()

    for entry in (NodeLeaf, NodeMid, NodeTree):
        _SCHEMA_CACHE.pop(entry, None)
    outer_first = build()

    assert inner_first == outer_first


def test_node_subclassing_a_node_stays_a_node():
    derived = NodeDerived.from_dict({"a": 5})
    assert derived == NodeDerived(a=5, b=2)
    assert derived.to_dict() == {"a": 5, "b": 2}
    assert derived.config_hash() == config_hash(derived)
