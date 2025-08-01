# jiuwen/extensions/common/log/log_utils.py
import os
import threading
import ast
from jiuwen.extensions.common.exception.base import JiuWenBaseException

_thread_log_instance = threading.local()  # 线程局部存储，用于保存trace_id


def set_thread_session(trace_id: str) -> None:
    """设置当前线程的trace_id"""
    _thread_log_instance.trace_id = trace_id


def get_thread_session() -> str:
    """获取当前线程的trace_id"""
    return getattr(_thread_log_instance, 'trace_id', '')


def get_log_max_bytes(max_bytes_config) -> int:
    """验证并获取有效的日志文件大小限制"""
    try:
        max_bytes = int(max_bytes_config)
    except ValueError as e:
        raise JiuWenBaseException(
            error_code=-1, message="-1"
        ) from e

    DEFAULT_LOG_MAX_BYTES = 100 * 1024 * 1024
    if max_bytes <= 0 or max_bytes > DEFAULT_LOG_MAX_BYTES:
        max_bytes = DEFAULT_LOG_MAX_BYTES

    return max_bytes
