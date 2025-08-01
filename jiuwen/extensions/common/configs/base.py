# jiuwen/extensions/common/configs/base.py
import os
import yaml

# --- 日志级别映射 ---
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


class SingletonConfig:
    _instance: yaml.constructor = None

    def __init__(self):
        if SingletonConfig._instance is not None:
            raise Exception("This class is a singleton!")
        # 修复配置文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_config_path = os.path.join(current_dir, "..", "..", "config.yaml")
        with open(log_config_path, encoding="utf-8") as file:
            # 加载配置文件
            raw_instance = yaml.safe_load(file)
            if 'logging' in raw_instance:
                level_str = raw_instance['logging'].get('level', 'WARNING').upper()
                raw_instance['logging']['level'] = name_to_level.get(level_str, WARNING)
            # 将转换后的配置存入 _instance
            SingletonConfig._instance = raw_instance

    @staticmethod
    def get_config():
        if SingletonConfig._instance is None:
            SingletonConfig()
        return SingletonConfig._instance


class ConfigDict(dict):
    def __call__(self):
        return self


config = ConfigDict(SingletonConfig.get_config())
