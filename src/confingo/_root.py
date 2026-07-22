"""Base class exposing the marshal / unmarshal helpers as methods.

A config dataclass that subclasses ``ConfigRoot`` gains ``from_dict`` /
``load_json`` as classmethods and ``to_dict`` / ``dumps_json`` / ``save_json`` /
``config_hash`` as instance methods, each delegating to the matching free
function. Only the root config needs to subclass it; nested sections stay plain
dataclasses and are walked by introspection as before.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from confingo._core import config_hash as _config_hash
from confingo._core import from_dict as _from_dict
from confingo._core import to_dict as _to_dict
from confingo._json import dumps_json as _dumps_json
from confingo._json import load_json as _load_json
from confingo._json import save_json as _save_json


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from typing_extensions import Self


class ConfigRoot:
    """Mixin adding marshal / unmarshal methods to a config dataclass.

    Subclass this on the root config dataclass, then decorate it with
    ``@dataclass`` as usual. The class carries its own schema, so building and
    loading read as ``Config.load_json(path)`` rather than
    ``load_json(Config, path)``.
    """

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

        Requires the ``yaml`` extra (``pip install confingo[yaml]``).

        Args:
            path: Path to the YAML file.

        Returns:
            The constructed config object, typed as the calling subclass.

        Raises:
            ConfigError: When the file is unreadable, holds invalid YAML, holds a
              non-mapping document, or fails validation.
        """
        from confingo._yaml import load_yaml as _load_yaml  # noqa: PLC0415

        return _load_yaml(cls, path)

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

        Requires the ``yaml`` extra (``pip install confingo[yaml]``).

        Args:
            indent: Number of spaces per indentation level, by default 2.
            sort_keys: Whether to sort mapping keys, by default False.

        Returns:
            The YAML document, ending with a newline.
        """
        from confingo._yaml import dumps_yaml as _dumps_yaml  # noqa: PLC0415

        return _dumps_yaml(self, indent=indent, sort_keys=sort_keys)

    def save_yaml(self, path: str | Path, *, indent: int = 2, sort_keys: bool = False) -> Path:
        """Write this config to a YAML file, replacing the target atomically.

        Requires the ``yaml`` extra (``pip install confingo[yaml]``).

        Args:
            path: Destination file path. Parent directories are created as needed.
            indent: Number of spaces per indentation level, by default 2.
            sort_keys: Whether to sort mapping keys, by default False.

        Returns:
            The path written.
        """
        from confingo._yaml import save_yaml as _save_yaml  # noqa: PLC0415

        return _save_yaml(self, path, indent=indent, sort_keys=sort_keys)

    def config_hash(self, *, length: int = 12) -> str:
        """Fingerprint this config with a stable digest over its canonical JSON form.

        Args:
            length: Number of leading hex characters to return, by default 12.

        Returns:
            The truncated SHA-256 digest.
        """
        return _config_hash(self, length=length)
