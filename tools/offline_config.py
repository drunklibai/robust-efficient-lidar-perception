"""Small YAML configuration loader for CPU-only sampling-policy preparation."""
from pathlib import Path
import yaml


class Config(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as err:
            raise AttributeError(key) from err

    def __setattr__(self, key, value):
        self[key] = value


def convert(value):
    if isinstance(value, dict):
        return Config({k: convert(v) for k, v in value.items()})
    if isinstance(value, list):
        return [convert(v) for v in value]
    return value


def merge(target, source):
    if '_BASE_CONFIG_' in source:
        cfg_from_yaml_file(source['_BASE_CONFIG_'], target)
    for key, value in source.items():
        if key == '_BASE_CONFIG_':
            continue
        if isinstance(value, dict):
            if not isinstance(target.get(key), dict):
                target[key] = Config()
            merge(target[key], value)
        else:
            target[key] = convert(value)
    return target


def cfg_from_yaml_file(path, config):
    with open(path, encoding='utf-8') as f:
        return merge(config, yaml.safe_load(f))


cfg = Config(ROOT_DIR=Path(__file__).resolve().parents[1])
