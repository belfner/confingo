"""YAML file IO for config objects, available through the ``yaml`` extra.

The optional counterpart to the JSON loaders (``pip install confingo[yaml]``).
``load_yaml`` reads a YAML document into a dataclass tree, and ``dumps_yaml`` /
``save_yaml`` render a config object back to YAML text. Both sides move through
the same JSON-safe data model as the JSON loaders -- mappings, sequences, and
scalars -- so a config round-trips across the two formats. Documents are read
with the safe loader and written with the safe dumper, which stays within that
shared data model.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)


try:
    import yaml
except ImportError as exc:  # PyYAML is an optional extra.
    raise ImportError("YAML support requires PyYAML; install it with: pip install confingo[yaml]") from exc

from confingo._core import (
    ConfigError,
    to_dict,
)
from confingo._fileio import (
    atomic_write_text,
    build_from_document,
    read_source_text,
)


if TYPE_CHECKING:
    from pathlib import Path

T = TypeVar("T")


def dumps_yaml(config: Any, *, indent: int = 2, sort_keys: bool = False) -> str:
    """Render a config object as a YAML document.

    The config is first marshalled to plain JSON-safe data, so the document holds
    only mappings, sequences, and scalar values.

    Args:
        config: The config object to render.
        indent: Number of spaces per indentation level, by default 2.
        sort_keys: Whether to sort mapping keys, by default False, which keeps
          field-declaration order.

    Returns:
        The YAML document, in field-declaration order, ending with a newline.
    """
    return yaml.safe_dump(
        to_dict(config),
        indent=indent,
        sort_keys=sort_keys,
        default_flow_style=False,
    )


def save_yaml(config: Any, path: str | Path, *, indent: int = 2, sort_keys: bool = False) -> Path:
    """Write the resolved config to a YAML file, replacing the target atomically.

    The serialized content comes from the in-memory object, so the file records
    the value of every field as the caller holds it, including any change made
    programmatically after loading.

    The write goes to a ``.tmp`` sibling and is then renamed onto the target, so
    a reader observes either the previous file or the complete new one.

    Args:
        config: The config object to write.
        path: Destination file path. Parent directories are created as needed.
        indent: Number of spaces per indentation level, by default 2.
        sort_keys: Whether to sort mapping keys, by default False.

    Returns:
        The path written.
    """
    return atomic_write_text(path, dumps_yaml(config, indent=indent, sort_keys=sort_keys))


def load_yaml(config_cls: type[T], path: str | Path) -> T:
    """Load a YAML file into a config object.

    Args:
        config_cls: The root dataclass to build.
        path: Path to the YAML file.

    Returns:
        The constructed config object.

    Raises:
        ConfigError: When the file is unreadable, holds invalid YAML, holds a
          non-mapping document, or fails validation. Validation failures list
          every issue found.
    """
    source, text = read_source_text(path)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError.single(str(exc), context=f"config file {source}") from exc
    return build_from_document(config_cls, data, source)
