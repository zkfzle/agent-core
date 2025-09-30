import os
from typing import Any, Dict, AsyncIterator
import unittest

from jiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from jiuwen.core.utils.llm.model_utils.model_factory import ModelFactory
from jiuwen.core.common.logging import logger
from jiuwen.core.multi_agent.runner.standalone_runner import StandaloneRunner
from jiuwen.core.multi_agent.stream.stream_handler import StreamHandler
from jiuwen.core.utils.llm.messages import SystemMessage, HumanMessage
from jiuwen.core.stream.base import StreamData, StreamCode

from jiuwen.core.agent.agent import Agent
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.runtime.interaction.base import AgentInterrupt

API_BASE = os.getenv("API_BASE", "")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "")

class PlannerAgent(Agent):
    def __init__(self, config=None):
        super().__init__(config)
        self.model = ModelFactory().get_model(model_provider=MODEL_PROVIDER,
                                              api_key=API_KEY,
                                              api_base=API_BASE)
        self.system_message = SystemMessage(
            content=(
                "You are a research assistant. Given a research question, you should answer the question briefly."
            )
        )

    async def invoke(self, inputs: Dict) -> Dict:
        """同步调用接口"""
        pass

    async def stream(self, inputs: Dict) -> AsyncIterator[Any]:
        """流式调用接口"""
        # 1. 初始化 Runtime
        session_id = inputs.pop("conversation_id", "default_session")
        runtime = await self._runtime.pre_run(session_id=session_id)
        messages = [self.system_message, HumanMessage(content=f"Research question: \n{inputs}\n")]
        # here just mock the model input
        # result = self.model.invoke(MODEL_NAME, messages).content
        mock_result = "Identify key areas and techniques in large model acceleration, including model parallelism, quantization, pruning and hardware optimization."

        yield StreamData(
            code=StreamCode.PARTIAL_CONTENT,
            msg="success",
            data=mock_result,
            execution_id="test"
        )
        runtime.update_state({"planner_agent": mock_result})
        await runtime.post_run()


class UserAgent(Agent):
    def __init__(self, config=None):
        super().__init__(config)

    async def invoke(self, inputs: Dict) -> Dict:
        """同步调用接口"""
        pass

    async def stream(self, inputs: Dict) -> AsyncIterator[Any]:
        """流式调用接口"""
        # 1. 初始化Runtime
        session_id = inputs.pop("conversation_id", "default_session")
        runtime = await self._runtime.pre_run(session_id=session_id)
        formatted_input = inputs.get("query", "")
        output = "do you like the plan, {}".format(formatted_input)
        try:
            await runtime.interact({"output": output, "result_type": "question"})
        except AgentInterrupt as e:
            logger.info("User agent interrupted")
            yield StreamData(
                code=StreamCode.CONTROLLER_AGENT_INTERRUPT_MESSAGE,
                msg="success",
                data={"result": output, "interrupted_agent": "searcher_agent"},
                execution_id="test"
            )

            runtime.update_state({"user_agent": inputs})
            await runtime.post_run()


class SearcherAgent(Agent):
    def __init__(self, config=None):
        super().__init__(config)
        self.model = ModelFactory().get_model(model_provider=MODEL_PROVIDER,
                                              api_key=API_KEY,
                                              api_base=API_BASE)

    async def invoke(self, inputs: Dict) -> Dict:
        """同步调用接口"""
        pass

    async def stream(self, inputs: Dict) -> AsyncIterator[Any]:
        """流式调用接口"""
        # 1. 初始化 Runtime
        session_id = inputs.pop("conversation_id", "default_session")
        runtime = await self._runtime.pre_run(session_id=session_id)
        research_topic = runtime.get_state("planner_agent")
        logger.info(f"resume research topic: {research_topic}")
        # here just mock the model input
        # result = self.model.invoke(MODEL_NAME, research_topic).content
        mock_result = "Found 3 related papers. paper 1:<>, paper 2:<>, paper 3:<>"

        yield StreamData(
            code=StreamCode.PARTIAL_CONTENT,
            msg="success",
            data=mock_result,
            execution_id="test"
        )
        runtime.update_state({"searcher_agent": mock_result})
        await runtime.post_run()


class Research:
    """Research流程控制器"""

    def __init__(self):
        self.runner = StandaloneRunner()
        self._setup_agents()

    def _create_agent_config(self, id: str) -> AgentConfig:
        return AgentConfig(id=id)

    def _setup_agents(self):
        """设置代理，注册成员"""
        planner_agent_config = self._create_agent_config("planner_agent")
        user_agent_config = self._create_agent_config("user_agent")
        searcher_agent_config = self._create_agent_config("searcher_agent")

        self.runner \
            .register_member("planner_agent", PlannerAgent, planner_agent_config) \
            .register_member("user_agent", UserAgent, user_agent_config) \
            .register_member("searcher_agent", SearcherAgent, searcher_agent_config)

    async def run(self, query: str, conversation_id: str, interrupted_agent: str):
        """执行Research流程"""
        if not interrupted_agent:
            await self.runner.start()
        try:
            # 1. 任务规划
            if not interrupted_agent or interrupted_agent == "planner_agent":
                stream_handler = StreamHandler()
                plan_messages = await self.runner.send_message(
                    {"inputs": {"conversation_id": conversation_id, "query": query}},
                    recipient="planner_agent",
                    sender="user",
                    stream_handler=stream_handler
                )
                print(f"get plan_messages: {plan_messages[-1].data.data}")
                async for chunk in stream_handler.stream_output():
                    yield chunk
                if stream_handler.is_running():
                    await stream_handler.stop()
            else:
                logger.info("No need to execute planner_agent")

            # 2. 询问user意见
            if not interrupted_agent or interrupted_agent == "user_agent":
                stream_handler = StreamHandler()
                await self.runner.send_message(
                    {"inputs": {"conversation_id": conversation_id, "query": plan_messages[-1].data.data}},
                    recipient="user_agent",
                    sender="planner_agent",
                    stream_handler=stream_handler
                )
                async for chunk in stream_handler.stream_output():
                    yield chunk
                if stream_handler.is_running():
                    await stream_handler.stop()
            else:
                logger.info("No need to execute user_agent")

            # 3. 查询相关研究
            if not interrupted_agent or interrupted_agent == "searcher_agent":
                stream_handler = StreamHandler()
                await self.runner.send_message(
                    {"inputs": {"conversation_id": conversation_id, "query": "test"}},
                    recipient="searcher_agent",
                    sender="user_agent",
                    stream_handler=stream_handler
                )
                async for chunk in stream_handler.stream_output():
                    yield chunk
                if stream_handler.is_running():
                    await stream_handler.stop()
            else:
                logger.info("No need to execute searcher_agent")


        except Exception as e:
            logger.error(f"运行出错: {e}")


    async def stop(self):
        """关闭并清理Research对象"""
        try:
            await self.runner.close()
            self.runner = None
            logger.info("Research runner stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop runner: {e}")

class MultiAgentTest(unittest.IsolatedAsyncioTestCase):

    async def test_multi_agent_send_messages(self):
        """多Agent点对点通信 Agent状态自保存 不需要在runner内部保存"""
        conversation_id = "test_conversation"
        query = "Find papers related to large model acceleration"
        logger.info(f"当前用户query：{query}")

        try:
            interaction_output = []
            interrupted_agent = ""
            runner = Research()
            # 运行Research流程
            async for result in runner.run(query, conversation_id, interrupted_agent):
                print("\n" + "=" * 60 + "\n第一次 收到流式输出\n", result.data, "\n" + "=" * 60)
                if hasattr(result.data, 'code') and result.data.code == StreamCode.CONTROLLER_AGENT_INTERRUPT_MESSAGE:
                    logger.info("接收到中断消息，退出循环")
                    interaction_output.append(result.data)
                    interrupted_agent = result.data.data.get("interrupted_agent", "")
                    break

            if interaction_output:
                interactive_input = InteractiveInput()
                interactive_input.update("user_feedback", "yes")
                async for result in runner.run(interactive_input.user_inputs.get("user_feedback"),
                                               conversation_id, interrupted_agent):
                    print("\n" + "=" * 60 + "\n第二次 收到流式输出\n", result.data, "\n" + "=" * 60)
                    if hasattr(result.data, 'code') and result.data.code == StreamCode.CONTROLLER_AGENT_INTERRUPT_MESSAGE:
                        logger.info("接收到中断消息，退出循环")
                        interaction_output.append(result.data)
                        interrupted_agent = result.data.data.get("interrupted_agent", "")
                        break

            await runner.stop()

        except Exception as e:
            logger.error(f"运行出错: {e}")
