# jiuwen/extensions/common/log/__init__.py
"""
扩展日志模块

提供自定义日志实现和配置管理
"""

from .default_impl import DefaultLogger, SafeRotatingFileHandler, ThreadContextFilter, CallerAwareFormatter
from jiuwen.core.common.logging.utils import set_thread_session, get_thread_session

__all__ = (
    "DefaultLogger", 
    "SafeRotatingFileHandler", 
    "ThreadContextFilter", 
    "CallerAwareFormatter",
    "set_thread_session", 
    "get_thread_session"
) 