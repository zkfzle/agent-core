# jiuwen/extensions/common/log/log_handlers.py
import os
import logging
from logging.handlers import RotatingFileHandler
from .log_utils import get_thread_session

class SafeRotatingFileHandler(RotatingFileHandler):
    """安全轮转文件处理器，添加进程ID和文件权限控制"""
    def __init__(self, filename, *args, **kwargs):
        pid = os.getpid()
        filename = f"{filename}-{pid}"
        super().__init__(filename, *args, **kwargs)
        os.chmod(self.baseFilename, 0o640)

    def doRollover(self):
        """执行日志轮转并设置文件权限"""
        super().doRollover()
        for i in range(self.backupCount, 0, -1):
            sfn = f"{self.baseFilename}.{i}"
            if os.path.exists(sfn):
                os.chmod(sfn, 0o440)
        os.chmod(self.baseFilename, 0o640)

class ThreadContextFilter(logging.Filter):
    """线程上下文过滤器，添加trace_id"""
    def __init__(self, log_type: str):
        super().__init__()
        self.log_type = log_type

    def filter(self, record):
        record.trace_id = get_thread_session()
        record.log_type = "perf" if self.log_type == 'performance' else self.log_type
        return True
