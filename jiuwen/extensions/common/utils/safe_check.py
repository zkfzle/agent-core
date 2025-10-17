import os

import yaml

SENSITIVE_CONFIG_FILE = "sensitive_config.yaml"


def load_sensitive_keywords() -> list:
    sensitive_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), SENSITIVE_CONFIG_FILE)
    config = {}
    with open(sensitive_config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('sensitive_paths', [])


def is_sensitive_path(path):
    for sensitive in load_sensitive_keywords():
        if path.startswith(os.path.abspath(sensitive)):
            return True
    return False
