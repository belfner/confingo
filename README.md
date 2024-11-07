# Confingo

Confingo is a Python package that simplifies configuration management using YAML and JSON files. It allows you to load, manipulate, and serialize configurations with ease, providing both attribute-style and dictionary-style access to configuration parameters.

## Features

- **Load Configurations**:
  - From YAML files.
  - From JSON files.
  - From strings, bytes, or file-like objects.
- **Access Configuration Parameters**:
  - Using dot notation (attribute-style access).
  - Using key access (dictionary-style access).
- **Programmatically Define and Manipulate Configurations**:
  - Create nested configurations effortlessly.
  - Dynamically add, modify, or delete configuration parameters.
- **Serialization and Deserialization**:
  - Serialize configurations to YAML and JSON formats.
  - Deserialize configurations from YAML and JSON formats.
  - Dump serialized configurations to file-like streams.
- **Delete Configuration Parameters**:
  - Remove configuration parameters using both attribute-style and dictionary-style access.

## Installation

You can install Confingo via pip:

```bash
pip install confingo
```

## Requirements

- Python >= 3.9
- PyYAML >= 5.4

## Usage

### Loading a Configuration File

Confingo supports loading configurations from both YAML and JSON files.

#### Loading from a YAML File

```python
from confingo import load_config

config = load_config('config.yaml')  # Automatically detects YAML based on file extension
print(config.database.host)
print(config['database']['host'])
```

#### Loading from a JSON File

```python
from confingo import load_config_from_json

config = load_config_from_json('config.json')
print(config.server.host)
print(config['server']['host'])
```

### Loading Configuration from Strings or File-Like Objects

You can also load configurations from strings, bytes, or file-like objects.

#### From a YAML String

```python
from confingo import load_config_from_content

yaml_content = """
database:
  host: localhost
  port: 5432
"""
config = load_config_from_content(yaml_content)
print(config.database.host)
```

#### From a JSON String

```python
from confingo import load_config_from_content

json_content = """
{
  "server": {
    "host": "127.0.0.1",
    "port": 8080
  }
}
"""
config = load_config_from_content(json_content)
print(config.server.host)
```

#### From a File-Like Object

```python
from confingo import load_config_from_content
from io import StringIO

yaml_content = """
database:
  host: localhost
  port: 5432
"""
file_like_object = StringIO(yaml_content)
config = load_config_from_content(file_like_object)
print(config.database.port)
```

### Programmatically Defining Configurations

You can create configurations programmatically without using a YAML or JSON file.

```python
from confingo import Config

config = Config()

config.database = Config()
config.database.host = 'localhost'
config.database.port = 5432

config.server = Config()
config.server.host = '127.0.0.1'
config.server.port = 8080

print(config.database.host)  # Outputs: localhost
print(config['server']['port'])  # Outputs: 8080
```

### Accessing Configuration Parameters

Confingo allows you to access configuration parameters using both attribute and key access.

```python
# Attribute access
print(config.database.host)

# Key access
print(config['database']['host'])
```

### Deleting Configuration Parameters

Confingo provides methods to delete configuration parameters using both attribute-style and dictionary-style access.

#### Deleting via Attribute Access

```python
from confingo import Config

config = Config()
config.database = Config()
config.database.host = 'localhost'
config.database.port = 5432

# Delete an attribute
del config.database.port

# Verify deletion
assert not hasattr(config.database, 'port')
assert 'port' not in config.database
```

#### Deleting via Dictionary Access

```python
from confingo import Config

config = Config()
config['server'] = Config()
config['server']['host'] = '127.0.0.1'
config['server']['port'] = 8080

# Delete an item
del config['server']['port']

# Verify deletion
assert 'port' not in config.server
assert not hasattr(config.server, 'port')
```

### Serialization and Deserialization

Confingo provides methods to serialize and deserialize configurations to and from YAML and JSON formats.

#### Serialize to YAML

```python
yaml_str = config.to_yaml()
print(yaml_str)
```

#### Dump YAML to a File

```python
with open('output.yaml', 'w') as f:
    config.dump_yaml(f)
```

#### Serialize to JSON

```python
json_str = config.to_json(indent=2)
print(json_str)
```

#### Dump JSON to a File

```python
with open('output.json', 'w') as f:
    config.dump_json(f, indent=2)
```

### Example

Given a `config.yaml` file:

```yaml
database:
  host: localhost
  port: 5432
  users:
    - name: admin
      role: superuser
    - name: guest
      role: read-only
server:
  host: 127.0.0.1
  port: 8080
```

You can load and access the configuration as follows:

```python
from confingo import load_config

config = load_config('config.yaml')

# Access database host
print(config.database.host)  # Outputs: localhost

# Access server port
print(config.server.port)  # Outputs: 8080

# List all users
for user in config.database.users:
    print(f"{user.name} - {user.role}")
```

#### Output:

```
localhost
8080
admin - superuser
guest - read-only
```

### Serialization Example

```python
from confingo import load_config

config = load_config('config.yaml')

# Serialize to YAML string
yaml_output = config.to_yaml()
print(yaml_output)

# Serialize to JSON string
json_output = config.to_json(indent=2)
print(json_output)

# Dump serialized YAML to a file
with open('serialized_config.yaml', 'w') as f:
    config.dump_yaml(f)

# Dump serialized JSON to a file
with open('serialized_config.json', 'w') as f:
    config.dump_json(f, indent=2)
```

## API Reference

### `Config` Class

A class representing a configuration object that supports both dictionary-style and attribute-style access to its elements.

Inherits from `argparse.Namespace` and `dict` to provide a flexible configuration management solution, allowing nested configurations, attribute access, and YAML/JSON-based serialization and deserialization.

#### Methods

- **`to_dict()`**: Converts the `Config` object to a native Python dictionary.
  
  ```python
  config_dict = config.to_dict()
  ```
  
- **`to_yaml(**kwargs)`**: Serializes the `Config` object to a YAML-formatted string.
  
  ```python
  yaml_str = config.to_yaml()
  ```
  
- **`dump_yaml(stream: IO[Any], **kwargs)`**: Writes the `Config` object as YAML to a file-like stream.
  
  ```python
  with open('output.yaml', 'w') as f:
      config.dump_yaml(f)
  ```
  
- **`to_json(**kwargs)`**: Serializes the `Config` object to a JSON-formatted string.
  
  ```python
  json_str = config.to_json(indent=2)
  ```
  
- **`dump_json(stream: IO[Any], **kwargs)`**: Writes the `Config` object as JSON to a file-like stream.
  
  ```python
  with open('output.json', 'w') as f:
      config.dump_json(f, indent=2)
  ```
  
- **`__delattr__(key: str) -> None`**: Deletes an attribute from the `Config` object, removing it from both attribute-style and dictionary-style access.
  
  ```python
  del config.attribute_name
  ```
  
- **`__delitem__(key: str) -> None`**: Deletes an item from the `Config` object using dictionary-style access, removing it from both attribute-style and dictionary-style access.
  
  ```python
  del config['item_name']
  ```

### Loading Functions

- **`load_config(path: Union[str, Path]) -> Config`**: Loads a YAML or JSON configuration file from a given file path and returns it as a `Config` object.
  
  ```python
  config = load_config('config.yaml')
  ```
  
- **`load_config_from_content(stream: Union[str, bytes, IO[Any]]) -> Config`**: Loads a YAML or JSON configuration from various input types (string, bytes, or file-like object) and returns it as a `Config` object.
  
  ```python
  config = load_config_from_content(yaml_content)
  config = load_config_from_content(json_content)
  ```
  

## Testing

Confingo includes a comprehensive test suite to ensure reliability and correctness. The tests cover configuration loading, access methods, serialization, deletion, and error handling.

### Running the Tests

To run the tests, ensure you have `pytest` installed and execute:

```bash
pytest
```

This command will automatically discover and run all the tests defined in your test suite.

### Test Coverage

The test suite covers the following aspects:

- **Attribute and Dictionary Access**: Ensures both access styles work seamlessly.
- **Loading Configurations**: From YAML files, JSON files, strings, bytes, and file-like objects.
- **Serialization**: To YAML and JSON strings and files, including empty and complex structures.
- **Deserialization**: Ensures loaded configurations match expected structures.
- **Deletion**: Removing attributes and items via both `delattr` and `del` operations.
- **Edge Cases**: Handling empty configurations and deeply nested structures.
- **Error Handling**: Invalid keys and data nodes correctly raise exceptions.
- **Representation**: Ensures `__str__` and `__repr__` provide accurate outputs.

## License

Confingo is licensed under the [MIT License](LICENSE).

## Author

Ben Elfner
