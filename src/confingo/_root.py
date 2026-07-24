"""Base class exposing the marshal / unmarshal helpers as methods.

A config dataclass that subclasses ``ConfigRoot`` gains the free-function
helpers as methods: the ``from_*`` / ``load_*`` builders as classmethods and the
``to_*`` / ``dumps_*`` / ``save_*`` / ``config_hash`` operations as instance
methods, each delegating to the matching free function. Only the root config
needs to subclass it; nested sections stay plain dataclasses and are walked by
introspection as before.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from confingo._core import config_hash as _config_hash
from confingo._core import from_dict as _from_dict
from confingo._core import to_dict as _to_dict
from confingo._equality import (
    _CUSTOM_EQ_MARKER,
    _canonical_eq,
)
from confingo._file import from_file as _from_file
from confingo._file import to_file as _to_file
from confingo._json import dumps_json as _dumps_json
from confingo._json import load_json as _load_json
from confingo._json import save_json as _save_json
from confingo._yaml import dumps_yaml as _dumps_yaml
from confingo._yaml import load_yaml as _load_yaml
from confingo._yaml import save_yaml as _save_yaml


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from typing import Self


class ConfigRoot:
    """Mixin adding marshal / unmarshal methods to a config dataclass.

    Subclass this on the root config dataclass, then decorate it with
    ``@dataclass`` as usual. The class carries its own schema, so building and
    loading read as ``Config.load_json(path)`` rather than
    ``load_json(Config, path)``.

    Subclassing also installs canonical equality from class-creation time:
    ``__init_subclass__`` plants the canonical ``__eq__`` and identity
    ``__hash__`` into the subclass ahead of the ``@dataclass`` decorator,
    which then keeps them in place of generating its own. A subclass whose
    body defines ``__eq__`` keeps it, marked so schema processing preserves
    it too.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Install canonical equality on each subclass at class creation.

        Args:
            **kwargs: Keyword arguments forwarded to ``super().__init_subclass__``.
        """
        super().__init_subclass__(**kwargs)
        current = cls.__dict__.get("__eq__")
        if current is None:
            cls.__eq__ = _canonical_eq  # type: ignore[method-assign]
            cls.__hash__ = object.__hash__  # type: ignore[method-assign]
        elif current is not _canonical_eq:
            setattr(cls, _CUSTOM_EQ_MARKER, True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, context: str = "config") -> Self:
        """Build an instance from a nested mapping, reporting every problem at once.

        Args:
            data: Nested mapping of config values, typically parsed from a config file.
            context: Description of the config source used in the error summary, by
              default ``"config"``.

        Returns:
            The constructed config object, typed as the calling subclass.

        Raises:
            ConfigError: When the mapping fails to build; the exception lists every
              issue found.
        """
        return _from_dict(cls, data, context=context)

    @classmethod
    def load_json(cls, path: str | Path) -> Self:
        """Load a JSON file into an instance.

        Args:
            path: Path to the JSON file.

        Returns:
            The constructed config object, typed as the calling subclass.

        Raises:
            ConfigError: When the file is unreadable, holds invalid JSON, holds a
              non-object document, or fails validation.
        """
        return _load_json(cls, path)

    @classmethod
    def load_yaml(cls, path: str | Path) -> Self:
        """Load a YAML file into an instance.

        Args:
            path: Path to the YAML file.

        Returns:
            The constructed config object, typed as the calling subclass.

        Raises:
            ConfigError: When the file is unreadable, holds invalid YAML, holds a
              non-mapping document, or fails validation.
        """
        return _load_yaml(cls, path)

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Load a config file into an instance, choosing the loader by extension.

        A ``.json`` path loads as JSON; a ``.yaml`` or ``.yml`` path loads as YAML,
        which requires the ``yaml`` extra.

        Args:
            path: Path to the config file.

        Returns:
            The constructed config object, typed as the calling subclass.

        Raises:
            ConfigError: When the extension names no supported format, or the file
              is unreadable, malformed, non-mapping, or fails validation.
        """
        return _from_file(cls, path)

    def to_dict(self) -> Any:
        """Convert this config into plain JSON-safe Python data.

        Returns:
            The converted plain-data structure.
        """
        return _to_dict(self)

    def dumps_json(self, *, indent: int = 2) -> str:
        """Render this config as JSON text.

        Args:
            indent: Number of spaces per indentation level, by default 2.

        Returns:
            The JSON document, ending with a newline.
        """
        return _dumps_json(self, indent=indent)

    def save_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write this config to a JSON file, replacing the target atomically.

        Args:
            path: Destination file path. Parent directories are created as needed.
            indent: Number of spaces per indentation level, by default 2.

        Returns:
            The path written.
        """
        return _save_json(self, path, indent=indent)

    def dumps_yaml(self, *, indent: int = 2, sort_keys: bool = False) -> str:
        """Render this config as a YAML document.

        Args:
            indent: Number of spaces per indentation level, by default 2.
            sort_keys: Whether to sort mapping keys, by default False.

        Returns:
            The YAML document, ending with a newline.
        """
        return _dumps_yaml(self, indent=indent, sort_keys=sort_keys)

    def save_yaml(self, path: str | Path, *, indent: int = 2, sort_keys: bool = False) -> Path:
        """Write this config to a YAML file, replacing the target atomically.

        Args:
            path: Destination file path. Parent directories are created as needed.
            indent: Number of spaces per indentation level, by default 2.
            sort_keys: Whether to sort mapping keys, by default False.

        Returns:
            The path written.
        """
        return _save_yaml(self, path, indent=indent, sort_keys=sort_keys)

    def to_file(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write this config to a file, choosing the writer by extension.

        A ``.json`` path writes JSON; a ``.yaml`` or ``.yml`` path writes YAML,
        which requires the ``yaml`` extra.

        Args:
            path: Destination file path. Parent directories are created as needed.
            indent: Number of spaces per indentation level, by default 2.

        Returns:
            The path written.

        Raises:
            ConfigError: When the extension names no supported format.
        """
        return _to_file(self, path, indent=indent)

    def config_hash(self, *, length: int = 12) -> str:
        """Fingerprint this config with a stable digest over its canonical JSON form.

        The digest ranges over the hashing fields (``init=True``, ``compare=True``,
        effective hash enabled), so a ``compare=False`` or ``hash=False`` field is
        carried by ``to_dict`` yet excluded from the digest.

        Args:
            length: Number of leading hex characters to return, by default 12.

        Returns:
            The truncated SHA-256 digest.
        """
        return _config_hash(self, length=length)
