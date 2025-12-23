#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import os
import yaml
from typing import Any, Dict, Optional

from openjiuwen.core.common.logging import logger


def load_yaml_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Loading YAML files.
    
    Args:
        file_path: YAML file path

    Returns:
        Parsed dictionary data, if file is empty return `None`
    
    Raises:
        FileNotFoundError: the file does not exist.
        YAMLError: YAML parsing error.
        Exception: other reading error.

    Example:
        config = load_yaml_file("config.yaml")
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise
