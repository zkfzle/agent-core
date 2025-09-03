import sys
import types
import os
from unittest.mock import Mock, patch

import pytest

from jiuwen.core.context_engine.base import EngineInput, ContextType
from jiuwen.core.context_engine.processor.preprocess.llm_compressor import (
    LLMCompressorConfig,
    LLMCompressor,
)
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.utils.llm.messages import AIMessage, HumanMessage

# Mock modules to avoid import issues
fake_base = types.ModuleType("base")
fake_base.logger = Mock()

fake_exception_module = types.ModuleType("base")
fake_exception_module.JiuWenBaseException = Mock()

sys.modules["jiuwen.core.common.logging.base"] = fake_base
sys.modules["jiuwen.core.common.exception.base"] = fake_exception_module


@pytest.fixture
def mock_llm_client():
    """Mock LLM client that returns predictable responses"""
    mock_client = Mock(spec=BaseChatModel)
    mock_client.invoke = Mock(
        return_value=AIMessage(content="""
        ```json\n{
            "user_input": "你是一个航班查询助手",
            "chat_history": "用户请求查询飞往天津的航班，航班号为CA9876，明天上午9:40起飞，请提前半小时登机"
        }
        ```
        """)
    )
    return mock_client


@pytest.fixture
def basic_config():
    """Basic LLMCompressorConfig for testing"""
    return LLMCompressorConfig(
        compress_targets=[ContextType.USER_INPUT, ContextType.CHAT_HISTORY],
    )


@pytest.fixture
def compressor_with_mock_llm(basic_config, mock_llm_client):
    """LLMCompressor with mocked LLM client"""
    with patch("jiuwen.core.utils.llm.model_utils.model_factory") as mock_factory:
        mock_factory_instance = Mock()
        mock_factory_instance.get_model.return_value = mock_llm_client
        mock_factory.return_value = mock_factory_instance

        compressor = LLMCompressor(basic_config)
        # Replace the LLM client with our mock
        compressor.bind_llm(mock_llm_client)
        return compressor

os.environ["API_BASE"] = "https://api.siliconflow.cn/v1/chat/completions"
os.environ["API_KEY"] = "sk-hbxmtgozxrqlmtjvkksoqlfmdwofuzovulewueptdjfyxqfz"
os.environ["MODEL_NAME"] = "Qwen/Qwen2.5-32B-Instruct"
os.environ["MODEL_PROVIDER"] = "siliconflow"

API_BASE = os.getenv("API_BASE", "")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "")

class TestLLMCompressorTest:
    """Integration tests with actual ModelFactory"""
    def test_compression_with_user_input_and_chat_history(self, compressor_with_mock_llm):
        compressor = compressor_with_mock_llm
        input = EngineInput(user_input="你是一个航班查询助手，请按用户的问题查询航班信息，注意信息的准确以及不要返回用户隐私数据",
                            chat_history=[AIMessage(content="你好，请问有什么可以帮助你？"),
                                          HumanMessage(content="请帮我查询一下飞往天津的航班"),
                                          AIMessage(content="好的，查询到行帮为CA9876"),
                                          HumanMessage(content="这班飞机啥时候起飞"),
                                          AIMessage(content="明天上午9:40起飞，请提前半小时登机"),])
        output = compressor.run(input)
        assert output.user_input == "你是一个航班查询助手"
        assert output.chat_history == "用户请求查询飞往天津的航班，航班号为CA9876，明天上午9:40起飞，请提前半小时登机"

