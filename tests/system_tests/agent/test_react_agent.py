import os
import unittest
from datetime import datetime

import pytest

from jiuwen.agent.common.schema import PluginSchema
from jiuwen.agent.react_agent import create_react_agent_config, create_react_agent, ReActAgent
from jiuwen.core.agent.controller.react_controller import ReActControllerInput, ReActControllerOutput
from jiuwen.core.component.common.configs.model_config import ModelConfig
from jiuwen.core.utils.llm.base import BaseModelInfo
from jiuwen.core.utils.llm.messages_chunk import BaseMessageChunk
from jiuwen.core.utils.tool.param import Param
from jiuwen.core.utils.tool.service_api.restful_api import RestfulApi


API_BASE = os.getenv("API_BASE", "")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "")

def build_current_date():
    current_datetime = datetime.now()
    return current_datetime.strftime("%Y-%m-%d")


class ReActAgentTest(unittest.IsolatedAsyncioTestCase):  # ① 关键改动
    @staticmethod
    def _create_model():
        return ModelConfig(model_provider=MODEL_PROVIDER,
                           model_info=BaseModelInfo(
                               model=MODEL_NAME,
                               api_base=API_BASE,
                               api_key=API_KEY,
                               temperature=0.7,
                               top_p=0.9,
                               timeout=30  # 添加超时设置
                           ))

    @staticmethod
    def _create_tool():
        weather_plugin = RestfulApi(
            name="WeatherReporter",
            description="天气查询插件",
            params=[
                Param(name="location", description="天气查询的地点，必须为英文", type="string", required=True),
                Param(name="date", description="天气查询的时间，格式为YYYY-MM-DD", type="string", required=True),
            ],
            path="http://127.0.0.1:9000/weather",
            headers={},
            method="GET",
            response=[],
        )
        return weather_plugin

    @staticmethod
    def _create_tool_schema():
        tool_info = PluginSchema(
            name='WeatherReporter',
            description='天气查询插件',
            inputs={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "天气查询的地点。\n注意：地点名称必须为英文",
                        "required": True
                    },
                    "date": {
                        "type": "string",
                        "description": "天气查询的时间，格式为YYYY-MM-DD",
                        "required": True
                    }
                }
            }
        )
        return tool_info

    @staticmethod
    def _create_prompt_template():
        system_prompt = "你是一个AI助手，在适当的时候调用合适的工具，帮助我完成任务！今天的日期为：{}\n注意：1. 如果用户请求中未指定具体时间，则默认为今天。"
        return [
            dict(role="system", content=system_prompt.format(build_current_date()))
        ]

    @unittest.skip("skip system test")
    async def test_react_agent_invoke_with_real_plugin(self):
        tools_schema = [self._create_tool_schema()]
        model_config = self._create_model()
        prompt_template = self._create_prompt_template()

        react_agent_config = create_react_agent_config(
            agent_id="react_agent_123",
            agent_version="0.0.1",
            description="AI助手",
            plugins=tools_schema,
            workflows=[],
            model=model_config,
            prompt_template=prompt_template
        )

        react_agent: ReActAgent = create_react_agent(
            agent_config=react_agent_config,
            workflows=[],
            tools=[self._create_tool()]
        )

        result = await react_agent.invoke({"query": "查询杭州的天气"})
        print(f"ReActAgent 最终输出结果：{result}")

    @unittest.skip("skip system test")
    async def test_react_agent_stream_with_real_plugin(self):
        tools_schema = [self._create_tool_schema()]
        model_config = self._create_model()
        prompt_template = self._create_prompt_template()

        react_agent_config = create_react_agent_config(
            agent_id="react_agent_123",
            agent_version="0.0.1",
            description="AI助手",
            plugins=tools_schema,
            workflows=[],
            model=model_config,
            prompt_template=prompt_template
        )

        react_agent: ReActAgent = create_react_agent(
            agent_config=react_agent_config,
            workflows=[],
            tools=[self._create_tool()]
        )

        res = react_agent.stream({"query": "查询杭州的天气"})
        async for i in res:
            print(i)

    @unittest.skip("skip system test")
    @pytest.mark.asyncio
    async def test_react_controller_stream(self):
        tools_schema = [self._create_tool_schema()]
        model_config = self._create_model()
        prompt_template = self._create_prompt_template()

        react_agent_config = create_react_agent_config(
            agent_id="react_agent_123",
            agent_version="0.0.1",
            description="AI助手",
            plugins=tools_schema,
            workflows=[],
            model=model_config,
            prompt_template=prompt_template
        )

        react_agent: ReActAgent = create_react_agent(
            agent_config=react_agent_config,
            workflows=[],
            tools=[self._create_tool()]
        )

        controller = react_agent._controller
        task = react_agent._task_manager.create_task("conversation_id")
        context = task.context
        inputs = dict(query="请调用工具WeatherReporter，帮我查询上海今天的天气")
        async for chunk in controller.stream(ReActControllerInput(**inputs), context):
            if isinstance(chunk, BaseMessageChunk):
                print(f"stream output 中间帧 >>> {chunk}")
            elif isinstance(chunk, ReActControllerOutput):
                print(f"stream output 最后一帧是controller的批结果 >>> {chunk}")
