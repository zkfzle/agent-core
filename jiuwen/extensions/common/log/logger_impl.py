# jiuwen/extensions/common/log/logger_impl.py
import os
import sys
import ast
import logging
from typing import Dict, Any, Optional

from .log_handlers import SafeRotatingFileHandler, ThreadContextFilter
from .log_utils import get_log_max_bytes, set_thread_session, get_thread_session
from .logger_protocol import LoggerProtocol


class DefaultLogger(LoggerProtocol):
    """默认日志实现"""
    def __init__(self, log_type: str, config: Dict[str, Any]):
        self.log_type = log_type
        self.config = config
        self._logger = logging.getLogger(log_type)
        self._setup_logger()

    def _setup_logger(self):
        """配置日志记录器"""
        level_config = self.config.get('level', 'WARNING')

        if isinstance(level_config, str):
            level = getattr(logging, level_config.upper(), logging.WARNING)
        elif isinstance(level_config, int):
            level = level_config
        else:
            level = logging.WARNING
            
        self._logger.setLevel(level)

        output = self.config.get('output', ['console'])
        log_file = self.config.get('log_file', f'{self.log_type}.log')

        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)

        if 'console' in output:
            stream_handler = logging.StreamHandler(stream=sys.stdout)  # 明确指定 sys.stdout
            stream_handler.addFilter(ThreadContextFilter(self.log_type))
            stream_handler.setFormatter(self._get_formatter())
            self._logger.addHandler(stream_handler)

        if 'file' in output:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, mode=0o750, exist_ok=True)

            backup_count = self.config.get('backup_count', 20)
            max_bytes = get_log_max_bytes(self.config.get('max_bytes', 20 * 1024 * 1024))

            file_handler = SafeRotatingFileHandler(
                filename=log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.addFilter(ThreadContextFilter(self.log_type))
            file_handler.setFormatter(self._get_formatter())
            self._logger.addHandler(file_handler)

    def _get_formatter(self) -> logging.Formatter:
        """获取日志格式化器"""
        log_format = self.config.get(
            'format') or '%(asctime)s.%(msecs)03d | %(log_type)s | %(trace_id)s | %(levelname)s | %(message)s'
        return logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        self._logger.exception(msg, *args, **kwargs)

    def log(self, level: int, msg: str, *args, **kwargs) -> None:
        self._logger.log(level, msg, *args, **kwargs)

    def setLevel(self, level: int) -> None:
        self._logger.setLevel(level)

    def addHandler(self, handler: logging.Handler) -> None:
        self._logger.addHandler(handler)

    def removeHandler(self, handler: logging.Handler) -> None:
        self._logger.removeHandler(handler)

    def addFilter(self, filter) -> None:
        self._logger.addFilter(filter)

    def removeFilter(self, filter) -> None:
        self._logger.removeFilter(filter)

    def get_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.config.copy()

    def reconfigure(self, config: Dict[str, Any]) -> None:
        """重新配置日志记录器"""
        self.config = config
        self._setup_logger()
