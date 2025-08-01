# jiuwen/extensions/common/log/log_manager.py
import threading
import os
import logging
from typing import Dict, Tuple, Optional

from .logger_protocol import LoggerProtocol
from .logger_impl import DefaultLogger
from jiuwen.extensions.common.configs.base import config


class LogManager:
    """日志管理器，支持多种日志类型和自定义实现"""
    _loggers: Dict[str, LoggerProtocol] = {}
    _lock = threading.RLock()
    _initialized = False

    @classmethod
    def initialize(cls) -> None:
        """初始化日志系统"""
        with cls._lock:
            if cls._initialized:
                return

            log_config = config.get("logging", {})

            jiuwen_log_path = os.environ.get('JIUWEN_LOG_PATH', log_config.get('log_path', '/var/log/jiuwen'))

            common_config = {
                'log_file': os.path.join(jiuwen_log_path, log_config.get('log_file', 'common.log')),
                'output': log_config.get('output', ['console', 'file']),
                'level': log_config.get('level', 'WARNING'),
                'backup_count': log_config.get('backup_count', 20),
                'max_bytes': log_config.get('max_bytes', 20 * 1024 * 1024),
                'format': log_config.get('format',
                                         '%(asctime)s | %(log_type)s | %(filename)s | %(lineno)d | %(funcName)s | %(trace_id)s | %(levelname)s | %(message)s')
            }

            interface_config = {
                'log_file': os.path.join(jiuwen_log_path, log_config.get('interface_log_file', 'interface.log')),
                'output': log_config.get('interface_output', ['console']),
                'level': log_config.get('level', 'WARNING'),
                'backup_count': log_config.get('backup_count', 20),
                'max_bytes': log_config.get('max_bytes', 20 * 1024 * 1024),
                'format': log_config.get('format',
                                         '%(asctime)s | %(log_type)s | %(trace_id)s | %(levelname)s | %(message)s')
            }

            prompt_builder_config = {
                'log_file': os.path.join(jiuwen_log_path,
                                         log_config.get('prompt_builder_interface_log_file', 'prompt_builder.log')),
                'output': log_config.get('interface_output', ['console']),
                'level': log_config.get('level', 'WARNING'),
                'backup_count': log_config.get('backup_count', 20),
                'max_bytes': log_config.get('max_bytes', 20 * 1024 * 1024),
                'format': log_config.get('format',
                                         '%(asctime)s | %(log_type)s | %(trace_id)s | %(levelname)s | %(message)s')
            }

            performance_config = {
                'log_file': os.path.join(jiuwen_log_path, log_config.get('performance_log_file', 'performance.log')),
                'output': log_config.get('performance_output', ['console', 'file']),
                'level': log_config.get('level', 'WARNING'),
                'backup_count': log_config.get('backup_count', 20),
                'max_bytes': log_config.get('max_bytes', 20 * 1024 * 1024),
                'format': log_config.get('format',
                                         '%(asctime)s | %(log_type)s | %(trace_id)s | %(levelname)s | %(message)s')
            }

            cls.register_logger('common', DefaultLogger('common', common_config))
            cls.register_logger('interface', DefaultLogger('interface', interface_config))
            cls.register_logger('prompt_builder', DefaultLogger('prompt_builder', prompt_builder_config))
            cls.register_logger('performance', DefaultLogger('performance', performance_config))

            cls._initialized = True

    @classmethod
    def register_logger(cls, log_type: str, logger: LoggerProtocol) -> None:
        """注册自定义日志记录器"""
        if not isinstance(logger, LoggerProtocol):
            raise TypeError(f"Logger must implement LoggerProtocol, got {type(logger)}")

        with cls._lock:
            cls._loggers[log_type] = logger

    @classmethod
    def get_logger(cls, log_type: str) -> LoggerProtocol:
        """获取指定类型的日志记录器"""
        if not cls._initialized:
            cls.initialize()

        with cls._lock:
            if log_type not in cls._loggers:
                cls._loggers[log_type] = DefaultLogger(log_type, {})

            return cls._loggers[log_type]

    @classmethod
    def get_all_loggers(cls) -> Dict[str, LoggerProtocol]:
        """获取所有已注册的日志记录器"""
        if not cls._initialized:
            cls.initialize()
        return cls._loggers.copy()

    @classmethod
    def reset(cls):
        """重置日志管理器状态（用于测试）"""
        with cls._lock:
            cls._loggers = {}
            cls._initialized = False
