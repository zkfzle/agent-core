# jiuwen/extensions/common/configs/config_manager.py
"""
配置管理器

提供配置访问接口，支持从代码配置和 YAML 文件加载
"""

import os
import yaml
from typing import Dict, Any, Union

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

name_to_level = {
    'CRITICAL': CRITICAL,
    'FATAL': FATAL,
    'ERROR': ERROR,
    'WARNING': WARNING,
    'WARN': WARN,
    'INFO': INFO,
    'DEBUG': DEBUG,
    'NOTSET': NOTSET,
}


class ConfigManager:
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # 默认从tests/unit_tests/common/log/app_config.yaml读取
            config_path = os.path.join(
                os.path.dirname(__file__), 
                '..', '..', '..', '..', 'tests', 'unit_tests', 'common', 'log', 'app_config.yaml'
            )
        
        self._config = None
        self._load_config(config_path)
    
    def _load_config(self, config_path: str):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)

            if 'logging' in config_dict:
                level_str = config_dict['logging'].get('level', 'WARNING').upper()
                config_dict['logging']['level'] = name_to_level.get(level_str, WARNING)
            
            self._config = config_dict
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到配置文件: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"YAML配置文件格式错误: {e}")
        except Exception as e:
            raise Exception(f"加载配置文件失败: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


class ConfigDict(dict):
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager._config)
        self._config_manager = config_manager
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._config_manager.get(key, default)
    
    def __call__(self):
        return self


config_manager = ConfigManager()
config = ConfigDict(config_manager) 