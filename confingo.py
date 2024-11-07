from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any
from typing import IO
from typing import overload
from typing import Union

import yaml


def validate_ident(val: str) -> None:
    """
    Validate whether a given string is a valid Python identifier.

    Parameters
    ----------
    val : str
        The string to validate as an identifier.

    Raises
    ------
    ValueError
        If `val` is not a valid Python identifier.
    """
    if not val.isidentifier():
        raise ValueError(f"'{val}' is not a valid identifier")


@overload
def to_builtin(obj: Config) -> dict[str, Any]:
    ...


@overload
def to_builtin(obj: Any) -> Any:
    ...


def to_builtin(obj: Any) -> Union[dict[str, Any], list[Any], Any]:
    """
    Recursively convert Config objects into native Python dictionaries or lists.

    Parameters
    ----------
    obj : Any
        The object to convert to a built-in Python type.

    Returns
    -------
    Union[dict[str, Any], list[Any], Any]
        Converted object as a dictionary, list, or the original object.
    """
    if isinstance(obj, Config):
        return {k: to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_builtin(i) for i in obj]
    else:
        return obj


class Config(Namespace, dict):
    """
    A configuration object supporting both dictionary-style and attribute-style access.

    Inherits from `argparse.Namespace` and `dict` to provide a flexible configuration
    management solution, allowing nested configurations, attribute access, and
    YAML/JSON-based serialization and deserialization.

    Attributes
    ----------
    Dynamically added through methods.

    Methods
    -------
    __repr__() -> str
        Returns the string representation of the configuration.
    __str__() -> str
        Converts the configuration into a YAML-formatted string.
    __setattr__(key: str, value: Any) -> None
        Sets an attribute with both dictionary and attribute-style access.
    __setitem__(key: str, value: Any) -> None
        Sets an attribute with both dictionary and attribute-style access.
    __delitem__(key: str) -> None
        Deletes an attribute with both dictionary and attribute-style access.
    __delattr__(key: str) -> None
        Deletes an attribute with both dictionary and attribute-style access.
    __contains__(item: Any) -> bool
        Checks if a key is in the configuration.
    to_dict() -> dict[str, Any]
        Converts the Config object to a native Python dictionary.
    to_yaml(**kwargs) -> str
        Serializes the Config object to a YAML-formatted string.
    dump_yaml(stream: IO[Any], **kwargs) -> None
        Writes the Config object as YAML to a file-like stream.
    to_json(**kwargs) -> str
        Serializes the Config object to a JSON-formatted string.
    dump_json(stream: IO[Any], **kwargs) -> None
        Writes the Config object as JSON to a file-like stream.
    """

    def __repr__(self) -> str:
        """
        Return the string representation of the Config object.

        Returns
        -------
        str
            String representation of the Config object, compatible with `Namespace.__repr__`.
        """
        return Namespace.__repr__(self)

    def __str__(self) -> str:
        """
        Convert the configuration to a YAML-formatted string.

        Returns
        -------
        str
            YAML-formatted string representing the configuration.
        """
        return json.dumps(
            self.to_dict(),
            indent=2,
        )

    def _add_item(self, key: str, value: Any) -> None:
        """
        Add an item to the Config object.

        Parameters
        ----------
        key : str
            The attribute name to add.
        value : Any
            The value to assign to the attribute.

        Raises
        ------
        ValueError
            If `key` is not a valid Python identifier.
        """
        validate_ident(key)
        if isinstance(value, dict):
            config = Config()
            _populate_config(value, config=config)
            value = config

        Namespace.__setattr__(self, key, value)
        dict.__setitem__(self, key, value)

    def _delete_item(self, key: str) -> None:
        """
        Delete an item from the Config object.

        Parameters
        ----------
        key : str
            The attribute name to delete.

        Raises
        ------
        KeyError
            If `key` does not exist in the Config object.
        """
        if key not in self:
            raise KeyError(f"'{key}' is not a valid key")
        dict.__delitem__(self, key)
        Namespace.__delattr__(self, key)

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Set an attribute on the Config object, allowing both dictionary
        and attribute-style access.

        Parameters
        ----------
        key : str
            The attribute name.
        value : Any
            The value to assign to the attribute.
        """
        self._add_item(key, value)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set an item in the Config object, allowing both dictionary
        and attribute-style access.

        Parameters
        ----------
        key : str
            The item key.
        value : Any
            The value to assign to the item.
        """
        self._add_item(key, value)

    def __delitem__(self, key: str) -> None:
        """
        Delete an item from the Config object.

        Parameters
        ----------
        key : str
            The item key to delete.
        """
        self._delete_item(key)

    def __delattr__(self, key: str) -> None:
        """
        Delete an item from the Config object.

        Parameters
        ----------
        key : str
            The attribute name to delete.
        """
        self._delete_item(key)

    def __contains__(self, item: Any) -> bool:
        """
        Check if a key exists in the Config object.

        Parameters
        ----------
        item : Any
            The key to check for in the configuration.

        Returns
        -------
        bool
            True if the item exists, False otherwise.
        """
        return dict.__contains__(self, item)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Config object to a native Python dictionary.

        Returns
        -------
        dict[str, Any]
            The configuration represented as a dictionary.
        """
        return to_builtin(self)

    def to_yaml(self, **kwargs) -> str:
        """
        Serialize the Config object to a YAML-formatted string.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments to pass to `yaml.dump`.

        Returns
        -------
        str
            YAML-formatted string representing the configuration.
        """
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            indent=2,
            sort_keys=False,
            **kwargs
        )

    def dump_yaml(self, stream: IO[Any], **kwargs) -> None:
        """
        Write the Config object as YAML to a file-like stream.

        Parameters
        ----------
        stream : IO[Any]
            The file-like object to write the YAML data to.
        **kwargs : Any
            Additional keyword arguments to pass to `yaml.dump`.
        """
        yaml.dump(
            self.to_dict(),
            stream,
            default_flow_style=False,
            indent=2,
            sort_keys=False,
            **kwargs
        )

    def to_json(self, **kwargs) -> str:
        """
        Serialize the Config object to a JSON-formatted string.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments to pass to `json.dumps`.

        Returns
        -------
        str
            JSON-formatted string representing the configuration.
        """
        return json.dumps(self.to_dict(), **kwargs)

    def dump_json(self, stream: IO[Any], **kwargs) -> None:
        """
        Write the Config object as JSON to a file-like stream.

        Parameters
        ----------
        stream : IO[Any]
            The file-like object to write the JSON data to.
        **kwargs : Any
            Additional keyword arguments to pass to `json.dump`.
        """
        json.dump(self.to_dict(), stream, **kwargs)


def _populate_config(
        data_node: Union[dict[str, Any], list[Any]],
        *,
        config: Config | None = None,
        config_list: list[Config | list[Any] | Any] | None = None
) -> None:
    """
    Recursively populate a Config object or a list with configuration data.

    Parameters
    ----------
    data_node : Union[dict[str, Any], list[Any]]
        The data structure to populate from, typically parsed from YAML or JSON.
    config : Config, optional
        The configuration object to populate, required if `data_node` is a dict.
    config_list : list[Config | list[Any] | Any], optional
        A list to populate if `data_node` is a list.

    Raises
    ------
    KeyError
        If a dictionary key is not a valid Python identifier.
    TypeError
        If `data_node` is neither a dictionary nor a list.
    """
    if isinstance(data_node, dict):
        assert config is not None
        for key, value in data_node.items():
            if not str(key).isidentifier():
                raise KeyError(
                    f'"{key}" is not a valid key. All keys must be valid Python identifiers in their string format.'
                )

            if isinstance(value, dict):
                sub_config = Config()
                _populate_config(value, config=sub_config)
                setattr(config, str(key), sub_config)
            elif isinstance(value, list):
                config_sub_list: list[Config | list[Any] | Any] = []
                _populate_config(value, config_list=config_sub_list)
                setattr(config, str(key), config_sub_list)
            else:
                setattr(config, str(key), value)

    elif isinstance(data_node, list):
        assert config_list is not None
        for item in data_node:
            if isinstance(item, dict):
                sub_config = Config()
                _populate_config(item, config=sub_config)
                config_list.append(sub_config)
            elif isinstance(item, list):
                sub_config_list: list[Any] = []
                _populate_config(item, config_list=sub_config_list)
                config_list.append(sub_config_list)
            else:
                config_list.append(item)
    else:
        raise TypeError("data_node must be a dict or list")


def load_config(path: Union[str, Path]) -> Config:
    """
    Load a YAML configuration file and return it as a Config object.

    This function reads a YAML configuration file specified by `path`,
    loads its contents into a Config object.

    Parameters
    ----------
    path : Union[str, Path]
        The path to the YAML configuration file.

    Raises
    ------
    ValueError
        If the YAML file does not have a dictionary structure at the root level.

    Returns
    -------
    Config
        A Config object populated with the loaded configuration data.
    """
    with open(path, "r") as f:
        config = load_config_from_content(f)

    return config


def load_config_from_content(stream: Union[str, bytes, IO[Any]]) -> Config:
    """
    Load a YAML configuration from various input types and return it as a Config object.

    This function can load YAML-formatted configuration data from a variety of sources,
    including in-memory strings, byte strings, or file-like objects. The loaded YAML data
    must have a dictionary structure at the root level to be compatible with the Config
    object format.

    Parameters
    ----------
    stream : Union[str, bytes, IO[Any]]
        The input containing YAML-formatted data, which can be:
          - a string representing YAML content,
          - bytes containing YAML data,
          - a file-like object (e.g., an open file handle).

    Raises
    ------
    ValueError
        If the YAML content does not have a dictionary structure at the root level.

    Returns
    -------
    Config
        A Config object populated with the configuration data.
    """
    data = yaml.safe_load(stream)

    if not isinstance(data, dict):
        raise ValueError("Unable to load config. The config file must use keys at the root level.")

    config = Config()
    _populate_config(data, config=config)

    return config

