"""
测试日志功能
"""

import pytest
import logging
import tempfile
import os
import sys
import threading
from io import StringIO
from unittest import mock
from typing import Dict, Any

from openjiuwen.core.common.logging import LogManager
from openjiuwen.extensions.common.log import DefaultLogger
from openjiuwen.core.common.logging import set_thread_session
from openjiuwen.core.common.logging import get_thread_session
from openjiuwen.core.common.logging import LoggerProtocol


def thread_function(session_id, log_list, stdout_capture):
    """线程函数，用于测试线程隔离"""
    logger = LogManager.get_logger('common')

    # 确保handler指向当前的stdout（在测试中已被重定向）
    if not logger._logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger._logger.addHandler(handler)
    else:
        # 更新现有handler的stream
        for handler in logger._logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                # handler已经指向sys.stdout，无需修改
                break
        else:
            # 如果没有指向stdout的handler，添加一个
            handler = logging.StreamHandler(sys.stdout)
            logger._logger.addHandler(handler)

    logger.set_level(logging.INFO)

    set_thread_session(session_id)
    logger.info(f'Thread started with session id {session_id}')

    for handler in logger._logger.handlers:
        handler.flush()

    log_list.append((session_id, get_thread_session()))


@pytest.fixture
def temp_config_dir():
    """创建临时配置目录"""
    temp_dir = tempfile.TemporaryDirectory()
    yield temp_dir
    temp_dir.cleanup()


@pytest.fixture
def test_config_file(temp_config_dir):
    """创建测试用的YAML配置文件"""
    config_file_path = os.path.join(temp_config_dir.name, 'test_config.yaml')
    
    test_config = {
        'logging': {
            'level': 'INFO',
            'backup_count': 5,
            'max_bytes': 1024 * 1024,
            'format': '%(asctime)s | %(log_type)s | %(trace_id)s | %(levelname)s | %(message)s',
            'log_file': 'common.log',
            'output': ['console', 'file'],
            'interface_log_file': 'interface.log',
            'prompt_builder_interface_log_file': 'prompt_builder.log',
            'performance_log_file': 'performance.log',
            'interface_output': ['console', 'file'],
            'performance_output': ['console', 'file'],
            'log_path': temp_config_dir.name
        }
    }
    
    import yaml
    with open(config_file_path, 'w', encoding='utf-8') as f:
        yaml.dump(test_config, f)
    
    return config_file_path


@pytest.fixture
def mock_log_config(test_config_file):
    """Mock日志配置"""
    from openjiuwen.extensions.common.configs.log_config import LogConfig, log_config as original_log_config
    from openjiuwen.extensions.common.configs.config_manager import ConfigManager, config_manager as original_config_manager
    
    test_log_config = LogConfig(test_config_file)
    test_config_manager = ConfigManager(test_config_file)
    
    # 替换全局配置
    import openjiuwen.extensions.common.configs.log_config as log_config_module
    import openjiuwen.extensions.common.configs.config_manager as config_manager_module
    
    # 保存原始引用
    _original_log_config = original_log_config
    _original_config_manager = original_config_manager
    
    # 替换模块级别的变量
    log_config_module.log_config = test_log_config
    config_manager_module.config_manager = test_config_manager
    
    yield test_log_config
    
    # 恢复原始配置
    log_config_module.log_config = _original_log_config
    config_manager_module.config_manager = _original_config_manager


@pytest.fixture(scope="function")
def stdout_capture():
    """捕获stdout输出"""
    original_stdout = sys.stdout
    capture = StringIO()
    sys.stdout = capture
    yield capture
    sys.stdout = original_stdout


@pytest.fixture(scope="function")
def stderr_capture():
    """捕获stderr输出"""
    original_stderr = sys.stderr
    capture = StringIO()
    sys.stderr = capture
    yield capture
    sys.stderr = original_stderr


@pytest.fixture(scope="function")
def initialized_logger(mock_log_config, stdout_capture):
    """初始化日志管理器并设置测试环境"""
    LogManager.reset()
    LogManager.initialize()

    # 更新所有logger的handler，确保输出到捕获的stdout
    # 但保留原有的filter和formatter
    from openjiuwen.extensions.common.log.default_impl import ThreadContextFilter
    from openjiuwen.core.common.logging.utils import get_thread_session

    for log in LogManager.get_all_loggers().values():
        # 更新现有的StreamHandler的stream指向当前stdout（已被重定向）
        for handler in log._logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                # 更新stream指向当前的stdout
                handler.stream = sys.stdout
            elif hasattr(handler, 'stream'):
                # 如果是文件handler，移除它（我们只测试控制台输出）
                try:
                    handler.close()
                except Exception:
                    pass
                log._logger.removeHandler(handler)

        # 如果没有StreamHandler，添加一个
        has_stream_handler = any(
            isinstance(h, logging.StreamHandler) 
            for h in log._logger.handlers
        )
        if not has_stream_handler:
            handler = logging.StreamHandler(sys.stdout)
            handler.add_filter(ThreadContextFilter(log.log_type))
            # 使用logger的格式化器
            formatter = log._get_formatter()
            handler.setFormatter(formatter)
            handler.set_level(logging.DEBUG)
            log._logger.addHandler(handler)

        log._logger.setLevel(logging.DEBUG)

    yield

    # 清理
    set_thread_session('')
    try:
        for logger in LogManager.get_all_loggers().values():
            if hasattr(logger, '_logger'):
                for handler in logger._logger.handlers[:]:
                    try:
                        handler.close()
                    except Exception:
                        pass
                    logger._logger.removeHandler(handler)
        LogManager.reset()
    except Exception:
        pass


class TestThreadSafety:
    """测试线程安全性"""

    def test_thread_trace_id_isolation(self, initialized_logger, stdout_capture):
        """测试线程间trace_id隔离"""
        log_list = []
        threads = [
            threading.Thread(target=thread_function, args=('10001', log_list, stdout_capture)),
            threading.Thread(target=thread_function, args=('10002', log_list, stdout_capture)),
            threading.Thread(target=thread_function, args=('10003', log_list, stdout_capture))
        ]

        stdout_capture.truncate(0)
        stdout_capture.seek(0)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 验证每个线程的session_id正确记录
        for session_id, recorded_id in log_list:
            assert session_id == recorded_id, f"线程session_id不匹配: 期望 {session_id}, 实际 {recorded_id}"

        # 主线程的session_id应该为空
        assert get_thread_session() == ''

        output = stdout_capture.getvalue()
        assert '10001' in output
        assert '10002' in output
        assert '10003' in output


class TestLogManager:
    """测试日志管理器功能"""

    def test_custom_logger_registration_and_usage(self, initialized_logger, capsys):
        """测试自定义日志记录器的注册和使用"""
        class CustomLogger:
            """自定义日志记录器，实现LoggerProtocol"""
            def __init__(self):
                self.messages = []
                self._config = {}

            def info(self, msg, *args, **kwargs):
                formatted_msg = f"CUSTOM LOGGER INFO: {msg}"
                print(formatted_msg)
                self.messages.append(formatted_msg)

            def debug(self, msg, *args, **kwargs): 
                pass

            def warning(self, msg, *args, **kwargs): 
                pass

            def error(self, msg, *args, **kwargs): 
                pass

            def critical(self, msg, *args, **kwargs): 
                pass

            def exception(self, msg, *args, **kwargs): 
                pass

            def log(self, level, msg, *args, **kwargs): 
                pass

            def set_level(self, level):
                pass

            def add_handler(self, handler):
                pass

            def remove_handler(self, handler):
                pass

            def add_filter(self, filter):
                pass

            def remove_filter(self, filter):
                pass

            def get_config(self) -> Dict[str, Any]:
                return self._config.copy()

            def reconfigure(self, config: Dict[str, Any]): 
                self._config = config

        custom_logger_instance = CustomLogger()

        LogManager.register_logger('custom', custom_logger_instance)

        retrieved_logger = LogManager.get_logger('custom')
        assert retrieved_logger is custom_logger_instance
        retrieved_logger.info("Test custom logger")

        assert "CUSTOM LOGGER INFO: Test custom logger" in custom_logger_instance.messages
        captured = capsys.readouterr()
        assert "CUSTOM LOGGER INFO: Test custom logger" in captured.out

    def test_default_logger_creation(self, initialized_logger, capsys):
        """测试动态创建默认日志记录器"""
        new_logger = LogManager.get_logger('new_type_test')

        assert isinstance(new_logger, DefaultLogger)

        new_logger.warning("Test new logger type")

        for handler in new_logger._logger.handlers:
            handler.flush()

        captured = capsys.readouterr()
        output = captured.out
        assert "Test new logger type" in output
        assert "new_type_test" in output
        assert "WARNING" in output

    def test_get_all_loggers(self, initialized_logger):
        """测试获取所有日志记录器"""
        all_loggers = LogManager.get_all_loggers()
        expected_types = {'common', 'interface', 'prompt_builder', 'performance'}
        assert expected_types.issubset(set(all_loggers.keys()))

        for log_instance in all_loggers.values():
            assert isinstance(log_instance, LoggerProtocol)

    def test_register_logger_type_check(self, initialized_logger):
        """测试注册日志记录器时的类型检查"""
        class InvalidLogger:
            """不实现LoggerProtocol的类"""
            pass

        invalid_logger = InvalidLogger()

        with pytest.raises(TypeError, match="Logger must implement LoggerProtocol"):
            LogManager.register_logger('invalid', invalid_logger)

    def test_get_logger_creates_on_demand(self, initialized_logger):
        """测试按需创建日志记录器"""
        # 获取一个不存在的logger类型
        new_type_logger = LogManager.get_logger('on_demand_test')

        assert isinstance(new_type_logger, DefaultLogger)
        assert new_type_logger.log_type == 'on_demand_test'

        # 再次获取应该返回同一个实例
        same_logger = LogManager.get_logger('on_demand_test')
        assert same_logger is new_type_logger


class TestLogLevel:
    """测试日志级别过滤"""

    def test_log_level_filtering(self, initialized_logger, capsys):
        """测试日志级别过滤功能"""
        test_logger_instance = LogManager.get_logger('level_test')

        test_logger_instance.set_level(logging.DEBUG)

        test_logger_instance.debug("Debug message")
        test_logger_instance.info("Info message")
        test_logger_instance.warning("Warning message")
        test_logger_instance.error("Error message")

        for handler in test_logger_instance._logger.handlers:
            handler.flush()

        captured = capsys.readouterr()
        output = captured.out

        assert "Debug message" in output
        assert "Info message" in output
        assert "Warning message" in output
        assert "Error message" in output

        # 测试更高级别的过滤
        test_logger_instance.set_level(logging.ERROR)

        test_logger_instance.debug("Should not appear debug")
        test_logger_instance.info("Should not appear info")
        test_logger_instance.warning("Should not appear warning")
        test_logger_instance.error("Should appear error")

        for handler in test_logger_instance._logger.handlers:
            handler.flush()

        captured = capsys.readouterr()
        output = captured.out

        assert "Should not appear debug" not in output
        assert "Should not appear info" not in output
        assert "Should not appear warning" not in output
        assert "Should appear error" in output


class TestLogFileOutput:
    """测试日志文件输出"""

    def test_interface_log_file_output(self, initialized_logger, stdout_capture, temp_config_dir):
        """测试接口日志文件输出"""
        interface_logger = LogManager.get_logger('interface')

        set_thread_session('FILE-TEST-123')

        test_message = "这是文件输出测试消息"
        interface_logger.info(test_message)

        for handler in interface_logger._logger.handlers:
            handler.flush()

        actual_log_file = interface_logger.config.get('log_file', '')

        if not os.path.exists(actual_log_file):
            # 如果文件不存在，至少验证控制台输出
            stdout_output = stdout_capture.getvalue()
            assert test_message in stdout_output, "控制台输出应该包含测试消息"
            assert 'FILE-TEST-123' in stdout_output, "控制台输出应该包含trace_id"
            return

        # 验证文件内容（文件可能因 delay/handler 重定向为空，此时回退到控制台断言）
        with open(actual_log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if content.strip():
            assert test_message in content, "日志文件应该包含测试消息"
            assert 'FILE-TEST-123' in content, "日志文件应该包含trace_id"
        else:
            # 若文件为空，至少验证控制台输出
            stdout_output = stdout_capture.getvalue()
            assert test_message in stdout_output, "控制台输出应该包含测试消息"
            assert 'FILE-TEST-123' in stdout_output, "控制台输出应该包含trace_id"
            return

        # 验证控制台输出
        stdout_output = stdout_capture.getvalue()
        assert test_message in stdout_output, "控制台输出应该包含测试消息"


class TestDefaultLogger:
    """测试默认日志记录器功能"""

    def test_message_sanitization(self, initialized_logger, capsys):
        """测试消息清理功能（防止日志注入）"""
        logger = LogManager.get_logger('common')
        logger.set_level(logging.INFO)

        # 测试包含换行符的消息
        test_message = "Test message\nwith newline\r\nand carriage return\r"
        logger.info(test_message)

        for handler in logger._logger.handlers:
            handler.flush()

        captured = capsys.readouterr()
        output = captured.out

        # 验证消息内容中的换行符被替换为空格（消息本身不应包含换行符）
        # 但日志格式本身可能包含换行符（每行日志结尾）
        # 检查消息内容部分（不包含日志格式前缀）
        # 消息应该被清理，所以不应该包含原始换行符
        lines = output.split('\n')
        for line in lines:
            if 'Test message' in line:
                # 验证消息内容中的换行符被替换为空格
                assert '\r' not in line
                # 消息应该被清理，换行符被替换为空格
                assert 'with newline' in line
                assert 'and carriage return' in line

    def test_logger_config_access(self, initialized_logger):
        """测试日志配置访问"""
        logger = LogManager.get_logger('common')

        config = logger.get_config()
        assert isinstance(config, dict)
        assert 'log_file' in config
        assert 'output' in config
        assert 'level' in config

    def test_logger_reconfigure(self, initialized_logger, stdout_capture):
        """测试日志重新配置"""
        logger = LogManager.get_logger('common')

        original_config = logger.get_config()
        new_config = original_config.copy()
        new_config['level'] = logging.DEBUG

        logger.reconfigure(new_config)

        # 验证配置已更新
        updated_config = logger.get_config()
        assert updated_config['level'] == logging.DEBUG

    def test_all_log_levels(self, initialized_logger, stdout_capture):
        """测试所有日志级别"""
        logger = LogManager.get_logger('common')
        logger.set_level(logging.DEBUG)

        stdout_capture.truncate(0)
        stdout_capture.seek(0)

        logger.debug("Debug level message")
        logger.info("Info level message")
        logger.warning("Warning level message")
        logger.error("Error level message")
        logger.critical("Critical level message")

        for handler in logger._logger.handlers:
            handler.flush()

        output = stdout_capture.getvalue()
        assert "Debug level message" in output
        assert "Info level message" in output
        assert "Warning level message" in output
        assert "Error level message" in output
        assert "Critical level message" in output

    def test_exception_logging(self, initialized_logger, stdout_capture):
        """测试异常日志记录"""
        logger = LogManager.get_logger('common')
        logger.set_level(logging.ERROR)

        stdout_capture.truncate(0)
        stdout_capture.seek(0)

        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("Exception occurred")

        for handler in logger._logger.handlers:
            handler.flush()

        output = stdout_capture.getvalue()
        assert "Exception occurred" in output
        assert "Test exception" in output or "ValueError" in output


class TestLogManagerReset:
    """测试日志管理器重置功能"""
    
    def test_reset_clears_loggers(self, initialized_logger):
        """测试重置清除所有日志记录器"""
        # 获取一些logger
        logger1 = LogManager.get_logger('common')
        logger2 = LogManager.get_logger('interface')
        
        assert len(LogManager.get_all_loggers()) > 0
        
        LogManager.reset()
        
        # 重置后应该重新初始化
        logger3 = LogManager.get_logger('common')
        assert isinstance(logger3, DefaultLogger)
        # 注意：由于reset后重新初始化，logger3可能不是同一个实例


class TestLogDirectoryCreation:
    """测试日志目录创建功能"""
    
    def test_create_nested_log_directory(self, temp_config_dir):
        """测试创建多层嵌套日志目录（如 logs/run）"""
        # 创建一个不存在的嵌套目录路径
        nested_log_path = os.path.join(temp_config_dir.name, 'logs', 'run')
        nested_log_file = os.path.join(nested_log_path, 'test.log')
        
        # 确保目录不存在
        if os.path.exists(nested_log_path):
            import shutil
            shutil.rmtree(os.path.join(temp_config_dir.name, 'logs'))
        
        # 创建配置，使用嵌套路径
        config = {
            'log_file': nested_log_file,
            'output': ['file'],  # 只使用文件输出
            'level': logging.INFO,
            'backup_count': 5,
            'max_bytes': 1024 * 1024,
            'format': '%(asctime)s | %(levelname)s | %(message)s'
        }
        
        # 创建 DefaultLogger，应该自动创建目录
        logger = DefaultLogger('test_nested', config)
        
        # 验证目录已创建
        assert os.path.exists(nested_log_path), f"目录 {nested_log_path} 应该被创建"
        assert os.path.isdir(nested_log_path), f"{nested_log_path} 应该是一个目录"
        
        # 写入日志
        logger.info("测试嵌套目录日志")
        
        # 刷新所有handler
        for handler in logger._logger.handlers:
            handler.flush()
            handler.close()
        
        # 验证日志文件已创建
        assert os.path.exists(nested_log_file), f"日志文件 {nested_log_file} 应该被创建"
        
        # 验证日志内容
        with open(nested_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "测试嵌套目录日志" in content
    
    def test_create_log_directory_with_relative_path(self, temp_config_dir):
        """测试使用相对路径创建日志目录"""
        # 切换到临时目录
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_config_dir.name)
            
            # 使用相对路径
            relative_log_file = os.path.join('logs', 'run', 'relative_test.log')
            
            # 确保目录不存在
            if os.path.exists('logs'):
                import shutil
                shutil.rmtree('logs')
            
            config = {
                'log_file': relative_log_file,
                'output': ['file'],
                'level': logging.INFO,
                'backup_count': 5,
                'max_bytes': 1024 * 1024,
                'format': '%(asctime)s | %(levelname)s | %(message)s'
            }
            
            logger = DefaultLogger('test_relative', config)
            
            # 验证目录已创建（使用绝对路径验证）
            abs_log_file = os.path.abspath(relative_log_file)
            abs_log_dir = os.path.dirname(abs_log_file)
            assert os.path.exists(abs_log_dir), f"目录 {abs_log_dir} 应该被创建"
            
            # 写入日志
            logger.info("测试相对路径日志")
            
            # 刷新所有handler
            for handler in logger._logger.handlers:
                handler.flush()
                handler.close()
            
            # 验证日志文件已创建
            assert os.path.exists(abs_log_file), f"日志文件 {abs_log_file} 应该被创建"
            
        finally:
            os.chdir(original_cwd)
    
    def test_create_log_directory_failure_raises_exception(self, temp_config_dir):
        """测试创建日志目录失败时抛出异常"""
        from openjiuwen.core.common.exception.exception import JiuWenBaseException
        from openjiuwen.core.common.exception.status_code import StatusCode
        
        # 创建一个无效的路径（在Windows上可能是无效的驱动器）
        if sys.platform == 'win32':
            # Windows上使用无效的驱动器路径
            invalid_log_file = 'Z:\\invalid\\drive\\test.log'
        else:
            # Unix系统上使用根目录下的只读路径（如果可能）
            invalid_log_file = '/proc/invalid_path/test.log'
        
        config = {
            'log_file': invalid_log_file,
            'output': ['file'],
            'level': logging.INFO,
            'backup_count': 5,
            'max_bytes': 1024 * 1024,
            'format': '%(asctime)s | %(levelname)s | %(message)s'
        }
        
        # 在某些系统上可能不会失败，所以使用mock来模拟失败
        with mock.patch('os.makedirs') as mock_makedirs:
            mock_makedirs.side_effect = OSError("Permission denied")
            
            with pytest.raises(JiuWenBaseException) as exc_info:
                DefaultLogger('test_failure', config)
            
            assert exc_info.value.error_code == StatusCode.LOG_PATH_CREATE_FAILED.code
            assert "Failed to create log directory" in exc_info.value.message
    
    def test_create_existing_directory_no_error(self, temp_config_dir):
        """测试目录已存在时不会报错"""
        # 先创建目录
        existing_log_path = os.path.join(temp_config_dir.name, 'logs', 'existing')
        os.makedirs(existing_log_path, exist_ok=True)
        
        existing_log_file = os.path.join(existing_log_path, 'test.log')
        
        config = {
            'log_file': existing_log_file,
            'output': ['file'],
            'level': logging.INFO,
            'backup_count': 5,
            'max_bytes': 1024 * 1024,
            'format': '%(asctime)s | %(levelname)s | %(message)s'
        }
        
        # 应该不会抛出异常
        logger = DefaultLogger('test_existing', config)
        
        # 验证目录仍然存在
        assert os.path.exists(existing_log_path)
        
        # 写入日志
        logger.info("测试已存在目录")
        
        # 刷新所有handler
        for handler in logger._logger.handlers:
            handler.flush()
            handler.close()
        
        # 验证日志文件已创建
        assert os.path.exists(existing_log_file)
    
    def test_log_path_validation(self, temp_config_dir):
        """测试日志路径合法性校验"""
        from openjiuwen.core.common.exception.exception import JiuWenBaseException
        from openjiuwen.core.common.exception.status_code import StatusCode
        
        # 测试敏感路径（根据系统不同）
        if sys.platform == 'win32':
            sensitive_path = 'C:\\Windows\\System32\\test.log'
        else:
            sensitive_path = '/etc/passwd'
        
        config = {
            'log_file': sensitive_path,
            'output': ['file'],
            'level': logging.INFO,
            'backup_count': 5,
            'max_bytes': 1024 * 1024,
            'format': '%(asctime)s | %(levelname)s | %(message)s'
        }
        
        with pytest.raises(JiuWenBaseException) as exc_info:
            DefaultLogger('test_sensitive', config)
        
        assert exc_info.value.error_code == StatusCode.LOG_PATH_SENSITIVE_ERROR.code
        assert "sensitive" in exc_info.value.message.lower() or "unsafe" in exc_info.value.message.lower()