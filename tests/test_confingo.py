import json
from io import BytesIO
from io import StringIO

import pytest
import yaml

from confingo import _populate_config
from confingo import Config
from confingo import load_config
from confingo import load_config_from_content


def test_config_attribute_access():
    """
    Test accessing configuration parameters using attribute-style access.
    """
    config = Config()
    config.database = Config()
    config.database.host = 'localhost'
    config.database.port = 5432

    assert config.database.host == 'localhost'
    assert config.database.port == 5432
    assert config['database']['host'] == 'localhost'
    assert config['database']['port'] == 5432


def test_config_dict_access():
    """
    Test accessing configuration parameters using dictionary-style access.
    """
    config = Config()
    config['database'] = Config()
    config['database']['host'] = 'localhost'
    config['database']['port'] = 5432

    assert config.database.host == 'localhost'
    assert config.database.port == 5432
    assert config['database']['host'] == 'localhost'
    assert config['database']['port'] == 5432


def test_load_config_yaml(tmp_path):
    """
    Test loading a YAML configuration file.
    """
    yaml_content = '''
    database:
      host: localhost
      port: 5432
      users:
        - name: admin
          role: superuser
        - name: guest
          role: read-only
    '''
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml_content)

    config = load_config(config_file)
    assert config.database.host == 'localhost'
    assert config.database.port == 5432
    assert len(config.database.users) == 2
    assert config.database.users[0].name == 'admin'
    assert config.database.users[0].role == 'superuser'


def test_load_config_invalid_file(tmp_path):
    """
    Test loading a YAML configuration file with an invalid root structure.
    """
    yaml_content = '''
    - item1
    - item2
    '''
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml_content)

    with pytest.raises(ValueError) as excinfo:
        load_config(config_file)
    assert 'Unable to load config. The config file must use keys at the root level.' in str(excinfo.value)


def test_load_config_from_string():
    """
    Test loading a configuration from a YAML string.
    """
    yaml_content = '''
    database:
      host: localhost
      port: 5432
    '''
    config = load_config_from_content(yaml_content)
    assert config.database.host == 'localhost'
    assert config.database.port == 5432


def test_load_config_from_bytes():
    """
    Test loading a configuration from YAML bytes.
    """
    yaml_content = b'''
    database:
      host: localhost
      port: 5432
    '''
    config = load_config_from_content(yaml_content)
    assert config.database.host == 'localhost'
    assert config.database.port == 5432


def test_load_config_from_file_like_object():
    """
    Test loading a configuration from a YAML file-like object (StringIO).
    """
    yaml_content = '''
    database:
      host: localhost
      port: 5432
    '''
    file_like_object = StringIO(yaml_content)
    config = load_config_from_content(file_like_object)
    assert config.database.host == 'localhost'
    assert config.database.port == 5432


def test_load_config_from_binary_file_like_object():
    """
    Test loading a configuration from a YAML binary file-like object (BytesIO).
    """
    yaml_content = b'''
    database:
      host: localhost
      port: 5432
    '''
    file_like_object = BytesIO(yaml_content)
    config = load_config_from_content(file_like_object)
    assert config.database.host == 'localhost'
    assert config.database.port == 5432


def test_config_str_representation():
    """
    Test the string representation (__str__) of the Config object (YAML format).
    """
    config = Config()
    config.name = 'TestConfig'
    config.value = 123

    expected_output = json.dumps(
        {'name': 'TestConfig', 'value': 123},
        indent=2,
    )
    assert str(config) == expected_output


def test_config_repr():
    """
    Test the repr of the Config object.
    """
    config = Config()
    config.name = 'Confingo'
    config.value = 123
    expected_repr = "Config(name='Confingo', value=123)"
    assert repr(config) == expected_repr


def test_nested_config():
    """
    Test deeply nested configuration parameters.
    """
    config = Config()
    config.level1 = Config()
    config.level1.level2 = Config()
    config.level1.level2.value = 'deep_value'

    assert config.level1.level2.value == 'deep_value'


def test_list_in_config():
    """
    Test storing and accessing a list within the Config object.
    """
    config = Config()
    config.items = [1, 2, 3]

    assert config.items == [1, 2, 3]


def test_config_contains():
    """
    Test the __contains__ method for checking key existence.
    """
    config = Config()
    config.name = 'Confingo'

    assert 'name' in config
    assert 'missing_key' not in config


def test_config_setattr():
    """
    Test setting an attribute using setattr.
    """
    config = Config()
    config.attr = 'value'

    assert config.attr == 'value'
    assert config['attr'] == 'value'


def test_config_setitem():
    """
    Test setting an item using dictionary-style access.
    """
    config = Config()
    config['item'] = 'value'

    assert config.item == 'value'
    assert config['item'] == 'value'


def test_config_delattr():
    """
    Test deleting an attribute using delattr.
    """
    config = Config()
    config.attr = 'value'

    del config.attr
    assert 'attr' not in config
    with pytest.raises(AttributeError):
        _ = config.attr


def test_config_delitem():
    """
    Test deleting an item using dictionary-style access.
    """
    config = Config()
    config['item'] = 'value'

    del config['item']
    assert 'item' not in config
    with pytest.raises(KeyError):
        _ = config['item']


def test_populate_config_invalid_key():
    """
    Test populating the Config object with an invalid key.
    """
    data = {'invalid-key!': 'value'}
    config = Config()

    with pytest.raises(KeyError) as excinfo:
        _populate_config(data, config=config)
    assert '"invalid-key!" is not a valid key.' in str(excinfo.value)


def test_populate_config_invalid_data_node():
    """
    Test populating the Config object with an invalid data node.
    """
    with pytest.raises(TypeError) as excinfo:
        _populate_config('not a dict or list', config=Config())
    assert 'data_node must be a dict or list' in str(excinfo.value)


def test_empty_config():
    """
    Test behavior of an empty Config object.
    """
    config = Config()
    assert config.to_dict() == {}
    yaml_output = config.to_yaml()
    assert yaml_output == "{}\n"


def test_deeply_nested_config():
    """
    Test the Config object with deeply nested configurations.
    """
    config = Config()
    current = config
    depth = 10
    for i in range(depth):
        current.sub = Config()
        current = current.sub
        current.value = i

    current = config
    for i in range(depth):
        assert hasattr(current, 'sub')
        current = current.sub
        assert current.value == i


def test_dump_empty_yaml(tmp_path):
    """
    Test dumping an empty Config object to a YAML file.
    """
    config = Config()
    yaml_file = tmp_path / 'empty.yaml'
    with open(yaml_file, 'w') as f:
        config.dump_yaml(f)

    with open(yaml_file, 'r') as f:
        loaded_yaml = f.read()

    assert loaded_yaml == "{}\n"


def test_dump_empty_json(tmp_path):
    """
    Test dumping an empty Config object to a JSON file.
    """
    config = Config()
    json_file = tmp_path / 'empty.json'
    with open(json_file, 'w') as f:
        config.dump_json(f, indent=2)

    with open(json_file, 'r') as f:
        loaded_json = json.load(f)

    assert loaded_json == {}


def test_config_to_yaml():
    """
    Test serializing the Config object to a YAML-formatted string.
    """
    config = Config()
    config.app = Config()
    config.app.name = 'MyApp'
    config.app.version = '1.0'
    config.database = Config()
    config.database.host = 'localhost'
    config.database.port = 3306

    yaml_output = config.to_yaml()

    expected_dict = {
        'app': {
            'name': 'MyApp',
            'version': '1.0'
        },
        'database': {
            'host': 'localhost',
            'port': 3306
        }
    }
    expected_yaml = yaml.dump(expected_dict, default_flow_style=False, indent=2, sort_keys=False)
    assert yaml_output == expected_yaml


def test_config_dump_yaml(tmp_path):
    """
    Test dumping the Config object to a YAML file.
    """
    config = Config()
    config.service = Config()
    config.service.enabled = True
    config.service.timeout = 30

    yaml_file = tmp_path / 'output.yaml'
    with open(yaml_file, 'w') as f:
        config.dump_yaml(f)

    with open(yaml_file, 'r') as f:
        loaded_yaml = f.read()

    expected_dict = {
        'service': {
            'enabled': True,
            'timeout': 30
        }
    }
    expected_yaml = yaml.dump(expected_dict, default_flow_style=False, indent=2, sort_keys=False)
    assert loaded_yaml == expected_yaml


def test_config_to_json():
    """
    Test serializing the Config object to a JSON-formatted string.
    """
    config = Config()
    config.server = Config()
    config.server.host = '127.0.0.1'
    config.server.port = 8080
    config.features = ['auth', 'logging', 'metrics']

    json_output = config.to_json(indent=2)

    expected_dict = {
        'server': {
            'host': '127.0.0.1',
            'port': 8080
        },
        'features': ['auth', 'logging', 'metrics']
    }
    expected_json = json.dumps(expected_dict, indent=2)
    assert json_output == expected_json


def test_config_dump_json(tmp_path):
    """
    Test dumping the Config object to a JSON file.
    """
    config = Config()
    config.logging = Config()
    config.logging.level = 'DEBUG'
    config.logging.handlers = ['console', 'file']

    json_file = tmp_path / 'output.json'
    with open(json_file, 'w') as f:
        config.dump_json(f, indent=2)

    with open(json_file, 'r') as f:
        loaded_json = f.read()

    expected_dict = {
        'logging': {
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        }
    }
    expected_json = json.dumps(expected_dict, indent=2)
    assert loaded_json == expected_json


def test_config_to_json_with_complex_structure():
    """
    Test serializing the Config object with a complex structure to JSON.
    """
    config = Config()
    config.application = Config()
    config.application.name = 'ComplexApp'
    config.application.modules = [
        Config(),
        Config(),
        {'name': 'notifications', 'enabled': True}
    ]
    config.application.modules[0].name = 'auth'
    config.application.modules[0].enabled = True
    config.application.modules[1].name = 'payments'
    config.application.modules[1].enabled = False

    json_output = config.to_json(indent=2)

    expected_dict = {
        'application': {
            'name': 'ComplexApp',
            'modules': [
                {'name': 'auth', 'enabled': True},
                {'name': 'payments', 'enabled': False},
                {'name': 'notifications', 'enabled': True}
            ]
        }
    }
    expected_json = json.dumps(expected_dict, indent=2)
    assert json_output == expected_json


def test_config_dump_json_with_complex_structure(tmp_path):
    """
    Test dumping the Config object with a complex structure to a JSON file.
    """
    config = Config()
    config.features = Config()
    config.features.list = ['feature1', 'feature2']
    config.features.settings = Config()
    config.features.settings.option = True

    json_file = tmp_path / 'complex_output.json'
    with open(json_file, 'w') as f:
        config.dump_json(f, indent=4)

    with open(json_file, 'r') as f:
        loaded_json = json.load(f)

    expected_dict = {
        'features': {
            'list': ['feature1', 'feature2'],
            'settings': {
                'option': True
            }
        }
    }
    assert loaded_json == expected_dict
