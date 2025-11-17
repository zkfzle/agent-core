#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import os
import pytest
from openjiuwen.agent.llm_agent.llm_agent import create_react_agent_config, create_react_agent, ReActAgent
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.runner.drunner.remote_client.remote_agent import RemoteAgent
from openjiuwen.core.runner.runner_config import RunnerConfig, DistributedConfig, MessageQueueConfig, PulsarConfig
from openjiuwen.core.utils.llm.base import BaseModelInfo
from openjiuwen.core.runner.runner import Runner

API_BASE = os.getenv("API_BASE")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME",)
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
os.environ.setdefault("LLM_SSL_VERIFY", "false")


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires real uv sync --extra pulsar and llm")
class TestAdapterTest:
    def setup_method(self):
        os.environ.setdefault("LLM_SSL_VERIFY", "false")
        os.environ.setdefault("RESTFUL_SSL_VERIFY", "false")

        pulsar_mq = RunnerConfig(
            distributed_mode=True,
            distributed_config=DistributedConfig(
                request_timeout=10.0,
                message_queue_config=MessageQueueConfig(
                    type="pulsar",
                    pulsar_config=PulsarConfig(
                        max_workers=8,
                        url="pulsar://localhost:6650",
                    ),
                )
            )
        )
        Runner.set_config(pulsar_mq)

    def teardown_method(self):
        """清理测试环境"""
        # 重置Runner配置
        from openjiuwen.core.runner.runner_config import DEFAULT_RUNNER_CONFIG
        Runner.set_config(DEFAULT_RUNNER_CONFIG)

    async def _create_and_register_agent(self, agent_id, agent_version="0.0.1"):
        """Create and register a React agent with common configuration"""
        # Create agent configuration
        react_agent_config = create_react_agent_config(
            agent_id=agent_id,
            agent_version=agent_version,
            description="AI助手",
            plugins=[],
            workflows=[],
            model=self._create_model(),
            prompt_template=[],
            tools=[]
        )

        # Create agent instance
        react_agent: ReActAgent = create_react_agent(
            agent_config=react_agent_config,
            workflows=[],
            tools=[]
        )

        # Register agent with runner
        Runner.add_agent(agent_id, react_agent)
        return react_agent

    @staticmethod
    def _create_model():
        return ModelConfig(
            model_provider=MODEL_PROVIDER,
            model_info=BaseModelInfo(
                model=MODEL_NAME,
                api_base=API_BASE,
                api_key=API_KEY,
                temperature=0.7,
                top_p=0.9,
                timeout=30
            )
        )

    async def test_adapter_invoke(self):
        """测试adater使用真实agent输出正常"""
        await Runner.start()

        try:
            await self._create_and_register_agent("weather-agent")

            client = RemoteAgent(agent_id="weather-agent")
            result = await client.invoke({"query": "你好"})
            print("[Remote Client] invoke result: ", result)

            assert result["output"] is not None
        finally:
            await Runner.stop()

    async def test_adapter_stream(self):
        """测试adater使用真实agent 流式输出正常"""
        await Runner.start()

        try:
            react_agent = await self._create_and_register_agent("weather-agent-stream")

            print("=== Testing stream response ===")
            client = RemoteAgent(agent_id="weather-agent-stream")
            chunks = []
            async for chunk in client.stream({"query": "你好"}):
                print(f"Stream chunk received: {chunk}")
                chunks.append(chunk)
            assert len(chunks) > 1

        finally:
            await Runner.stop()
