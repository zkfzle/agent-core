# jiuwen/extensions/common/log/__init__.py
"""logger for common and interface."""
__all__ = ("common_logger", "interface_logger", "prompt_builder_interface_logger",
           "performance_logger", "get_thread_session", "set_thread_session",
           "LoggerProtocol", "LogManager")

from .log_manager import LogManager
from .log_utils import set_thread_session, get_thread_session
from .logger_protocol import LoggerProtocol


common_logger = LogManager.get_logger('common') # 普通系统日志
interface_logger = LogManager.get_logger('interface') # 接口调用系统日志
prompt_builder_interface_logger = LogManager.get_logger('prompt_builder') # 特定模块日志
performance_logger = LogManager.get_logger('performance') # 性能监控日志
