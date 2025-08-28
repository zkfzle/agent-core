import sys
import types
from typing import Any

import pytest
from unittest.mock import Mock

from jiuwen.core.component.common.configs.model_config import ModelConfig
from jiuwen.core.stream.emitter import StreamEmitter
from jiuwen.core.stream.manager import StreamWriterManager
from jiuwen.core.utils.llm.messages import AIMessage

fake_base = types.ModuleType("base")
fake_base.logger = Mock()

fake_exception_module = types.ModuleType("base")
fake_exception_module.JiuWenBaseException = Mock()

sys.modules["jiuwen.core.common.logging.base"] = fake_base
sys.modules["jiuwen.core.common.exception.base"] = fake_exception_module

from tests.unit_tests.workflow.test_mock_node import MockStartNode, MockEndNode
from tests.unit_tests.workflow.test_workflow import create_flow, create_context

from unittest.mock import patch, AsyncMock

from jiuwen.core.common.constants.constant import USER_FIELDS
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.component.llm_comp import LLMCompConfig, LLMExecutable, LLMComponent
from jiuwen.core.context.context import NodeContext
from jiuwen.core.utils.llm.base import BaseModelInfo


@pytest.fixture
def fake_node_ctx():
    from unittest.mock import MagicMock
    ctx = MagicMock(spec=NodeContext)
    ctx.store = MagicMock()
    ctx.store.read.return_value = []
    ctx.executable_id.return_value = "test"
    emitter = StreamEmitter()
    ctx.stream_writer_manager.return_value = StreamWriterManager(emitter)
    return ctx


@pytest.fixture
def fake_input():
    return lambda **kw: {USER_FIELDS: kw}


@pytest.fixture
def fake_model_config() -> ModelConfig:
    """构造一个最小可用的 ModelConfig"""
    return ModelConfig(
        model_provider="openai",
        model_info=BaseModelInfo(
            api_key="sk-fake",
            api_base="https://api.openai.com/v1",
            model_name="gpt-3.5-turbo",
            temperature=0.8,
            top_p=0.9,
            streaming=False,
            timeout=30.0,
        ),
    )


@patch(
    "jiuwen.core.utils.llm.model_utils.model_factory.ModelFactory.get_model",
    autospec=True,
)
class TestLLMExecutableInvoke:

    @pytest.mark.asyncio
    async def test_invoke_success(
            self,
            mock_get_model,  # 这就是补丁
            fake_node_ctx,
            fake_input,
            fake_model_config,
    ):
        config = LLMCompConfig(
            model=fake_model_config,
            template_content=[{"role": "user", "content": "Hello {query}"}],
            response_format={"type": "text"},
            output_config={"result": {
                "type": "string",
                "required": True,
            }},
        )
        exe = LLMExecutable(config)

        fake_llm = AsyncMock()
        fake_llm.ainvoke = AsyncMock(return_value=AIMessage(content="mocked response"))
        mock_get_model.return_value = fake_llm

        output = await exe.invoke(fake_input(userFields=dict(query="pytest")), fake_node_ctx)

        assert output[USER_FIELDS] == {'result': 'mocked response'}
        fake_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_success(
            self,
            mock_get_model,  # 这就是补丁
            fake_node_ctx,
            fake_input,
            fake_model_config,
    ):
        config = LLMCompConfig(
            model=fake_model_config,
            template_content=[{"role": "user", "content": "Hello {query}"}],
            response_format={"type": "text"},
            output_config={"result": {
                "type": "string",
                "required": True,
            }},
        )
        exe = LLMExecutable(config)

        fake_llm = AsyncMock()

        # 模拟异步生成器，返回多个 AIMessage chunk
        async def mock_stream_response(input: Any):
            for chunk in ["mocked ", "response"]:
                yield AIMessage(content=chunk)

        fake_llm.astream = mock_stream_response
        mock_get_model.return_value = fake_llm

        # 调用 stream 方法，异步迭代所有 chunk
        chunks = []
        async for chunk in exe.stream(fake_input(userFields=dict(query="pytest")), fake_node_ctx):
            chunks.append(chunk)

        # 假设 LLMExecutable.stream 会把每个 AIMessage.content 直接 yield 出来
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_invoke_llm_exception(
            self,
            mock_get_model,
            fake_node_ctx,
            fake_input,
            fake_model_config,
    ):
        config = LLMCompConfig(model=fake_model_config, template_content=[{"role": "user", "content": "Hello {name}"}])
        exe = LLMExecutable(config)

        fake_llm = Mock()
        fake_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        mock_get_model.return_value = fake_llm

        with pytest.raises(JiuWenBaseException) as exc_info:
            await exe.invoke(fake_input(userFields=dict(query="pytest")), fake_node_ctx)

        assert "LLM down" in str(exc_info.value.message)

    @pytest.mark.asyncio  # 新增
    async def test_llm_in_workflow(
            self,
            mock_get_model,
            fake_model_config,
    ):
        """LLM 节点在完整工作流中的异步测试"""
        context = create_context()

        # 1. 打桩 LLM
        fake_llm = AsyncMock()
        fake_llm.ainvoke = AsyncMock(return_value=AIMessage(
            role="assistant", content="mocked response", tool_calls=[], usage_metadata=None))
        mock_get_model.return_value = fake_llm

        # 2. 构造工作流
        flow = create_flow()
        flow.set_start_comp("start", MockStartNode("start"),
                            inputs_schema={
                                "a": "${user.inputs.a}",
                                "b": "${user.inputs.b}"})

        flow.set_end_comp(
            "end",
            MockEndNode("end"),
            inputs_schema={"a": "${a.result}", "b": "${b.result}"},
        )

        config = LLMCompConfig(
            model=fake_model_config,
            template_content=[{"role": "user", "content": "Hello {name}"}],
            response_format={"type": "text"},
            output_config={"result": {"type": "string", "required": True}},
        )
        llm_comp = LLMComponent(config)
        flow.add_workflow_comp("llm", llm_comp,
                               inputs_schema={"a": "${start.a}",
                                              "userFields": {"query": "${start.query}"}})

        flow.add_connection("start", "llm")
        flow.add_connection("llm", "end")

        # 3. 直接异步调用
        result = await flow.invoke(inputs={"a": 2, "userFields": dict(query="pytest")}, context=context)
        assert result is not None