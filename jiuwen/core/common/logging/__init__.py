

from .protocol import LoggerProtocol
from .manager import LogManager
from .utils import set_thread_session, get_thread_session


_initialized = False

def _ensure_initialized():
    global _initialized
    if not _initialized:
        try:
            from jiuwen.extensions.common.log.default_impl import DefaultLogger
            LogManager.set_default_logger_class(DefaultLogger)
        except ImportError:
            pass
        LogManager.initialize()
        _initialized = True


class LazyLogger:
    def __init__(self, getter_func):
        self._getter_func = getter_func
        self._logger = None
    
    def __getattr__(self, name):
        if self._logger is None:
            _ensure_initialized()  # 确保初始化
            self._logger = self._getter_func()
        return getattr(self._logger, name)


logger = LazyLogger(lambda: LogManager.get_logger('common'))
interface_logger = LazyLogger(lambda: LogManager.get_logger('interface'))
performance_logger = LazyLogger(lambda: LogManager.get_logger('performance'))
prompt_builder_logger = LazyLogger(lambda: LogManager.get_logger('prompt_builder'))

__all__ = [
    "LoggerProtocol",
    "LogManager", 
    "set_thread_session",
    "get_thread_session",
    "logger",
    "interface_logger",
    "performance_logger",
    "prompt_builder_logger"
]
