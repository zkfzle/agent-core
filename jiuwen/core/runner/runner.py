import asyncio
from typing import Union, Any, List, Optional

from jiuwen.agent.chat_agent import ChatAgent
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.agent import Agent, AgentRuntime
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.common.logging import logger
from jiuwen.core.runtime.agent_group_manager import AgentGroupProvider, AgentGroupMgr
from jiuwen.core.runtime.agent_manager import AgentProvider, AgentMgr
from jiuwen.core.runtime.resource_manager import ResourceMgr
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.runtime.workflow import WorkflowRuntime
from jiuwen.core.runtime.workflow_manager import generate_workflow_key
from jiuwen.core.runtime.wrapper import TaskRuntime
from jiuwen.core.utils.config.user_config import UserConfig
from jiuwen.core.utils.tool.base import Tool
from jiuwen.core.utils.tool.mcp.base import McpToolInfo
from jiuwen.core.workflow.base import Workflow
from jiuwen.core.runner.agent_group import AgentGroup


# mock
class LocalMessageQueue:
    async def start(self):
        pass

    async def stop(self):
        pass


DEFAULT_RUNNER_ID = "global"


class Runner:
    """
    Runner接口
    """

    _DEFAULT_AGENT_SESSION_ID = "default_session"

    _AGENT_CONVERSATION_ID = "conversation_id"

    def __init__(self, resource_manager: ResourceMgr, runner_id: str = ""):
        self._runner_id = runner_id
        self._resource_manager = resource_manager
        self._message_queue = LocalMessageQueue()
        self._agent_group_mgr: AgentGroupMgr = AgentGroupMgr()
        self._agent_mgr: AgentMgr = AgentMgr(resource_manager)

    async def start(self) -> bool:
        return await self._message_queue.start()

    async def stop(self):
        return await self._message_queue.stop()

    def message_queue(self):
        return self._message_queue

    async def add_agent_group(self, agent_group_id: str, agent_group: Union[AgentGroup, AgentGroupProvider]):
        self._agent_group_mgr.add_agent_group(agent_group_id, agent_group)
        if isinstance(agent_group, AgentGroup):
            topic = agent_group.get_topic()
            subscription = await self._message_queue.subscribe(topic)
            agent_group.set_subscription(subscription)
        else:
            agent_group_instance = self._agent_group_mgr.get_agent_group(agent_group_id)
            topic = agent_group_instance.get_topic()
            subscription = await self._message_queue.subscribe(topic)
            agent_group_instance.set_subscription(subscription)

    async def remove_agent_group(self, agent_group_id: str) -> Union[AgentGroup, AgentGroupProvider]:
        agent_group = self._agent_group_mgr.remove_agent_group(agent_group_id)
        if agent_group:
            topic = agent_group.get_topic()
            await self._message_queue.unsubscribe(topic, agent_group._subscription)
        return agent_group

    def add_agent(self, agent_id, agent: Union[Agent, AgentProvider]):
        self._agent_mgr.add_agent(agent_id, agent)

    def remove_agent(self, agent_id) -> Union[Agent, AgentProvider]:
        return self._agent_mgr.remove_agent(agent_id)

    async def run_workflow(self, workflow: Union[str, Workflow], inputs: Any,
                           *, runtime: Union[Runtime, WorkflowRuntime] = None):
        workflow_instance, workflow_runtime = self._prepare_workflow(workflow, runtime)
        return await workflow_instance.invoke(inputs, runtime=workflow_runtime)

    async def run_workflow_streaming(self, workflow: Union[str, Workflow], inputs: Any,
                                     *, runtime: Union[Runtime, WorkflowRuntime] = None):
        workflow_instance, workflow_runtime = self._prepare_workflow(workflow, runtime)
        return workflow_instance.stream(inputs, runtime=workflow_runtime)

    async def run_agent(self, agent: Union[str, Agent], inputs: Any):
        agent_instance, agent_runtime = await self._prepare_agent(agent, inputs)
        res = await agent_instance.invoke(inputs, agent_runtime)
        await agent_runtime.post_run()
        return res

    async def run_agent_streaming(self, agent: Union[str, Agent], inputs: Any):
        agent_instance, agent_runtime = await self._prepare_agent(agent, inputs)
        if isinstance(agent_instance, ChatAgent):
            try:
                async for chunk in agent_instance.stream(inputs, agent_runtime):
                    yield chunk
            finally:
                await agent_runtime.post_run()
        else:
            async def stream_process():
                try:
                    await agent_instance.runner_controller_stream(inputs, agent_runtime)
                finally:
                    await agent_runtime.post_run()

            task = asyncio.create_task(stream_process())
            async for chunk in agent_runtime.stream_iterator():
                yield chunk

            try:
                await task
            except Exception as e:
                logger.error(f"{self.__class__.__name__} stream error.")
                if UserConfig.is_sensitive():
                    raise JiuWenBaseException(StatusCode.AGENT_SUB_TASK_TYPE_ERROR.code,
                                              f"{self.__class__.__name__} stream error.")
                else:
                    raise JiuWenBaseException(StatusCode.AGENT_SUB_TASK_TYPE_ERROR.code,
                                              f"{self.__class__.__name__} stream error.") from e


    async def run_agent_group(self, agent_group: Union[str, AgentGroup], inputs: Any):
        agent_group_instance = self._prepare_agent_group(agent_group)
        return await agent_group_instance.invoke(inputs)

    async def run_agent_group_streaming(self, agent_group: Union[str, AgentGroup], inputs: Any):
        agent_group_instance = self._prepare_agent_group(agent_group)
        return agent_group_instance.stream(inputs)

    async def run_tool(self, tool: Union[str, Tool], inputs, *, runtime: Runtime = None):
        tool_instance = self._prepare_tool(tool, runtime)
        return await tool_instance.ainvoke(inputs, runtime=runtime)

    async def run_tool_streaming(self, tool: Union[str, Tool], inputs, *, runtime: Runtime = None):
        tool_instance = self._prepare_tool(tool, runtime)
        return tool_instance.astream(inputs, runtime=runtime)

    async def list_tools(self, tool_server_name: Union[str, List[str]]) -> Union[
        Optional[List[McpToolInfo]], List[Optional[List[McpToolInfo]]]]:
        return

    def _check_is_agent_tool(self, runtime, tool) -> bool:
        if not self._is_called_by_agent(runtime):
            return True
        agent_config: AgentConfig = runtime.get_agent_config()

        if isinstance(tool, str):
            tool_name = tool
        else:
            tool_name = tool.name

        for agent_tool in agent_config.tools:
            if agent_tool == tool_name:
                return True
        return False

    def _check_is_agent_workflow(self, runtime, workflow_key) -> bool:
        if not self._is_called_by_agent(runtime):
            return True
        agent_config: AgentConfig = runtime.get_agent_config()

        for workflow_schema in agent_config.workflows:
            if generate_workflow_key(workflow_schema.id, workflow_schema.version) == workflow_key:
                return True
        return False

    def _is_called_by_agent(self, runtime: Runtime) -> bool:
        return runtime and isinstance(runtime, TaskRuntime)

    def _create_workflow_runtime(self, runtime):
        # workflow的runtime的转换
        if not runtime:
            workflow_runtime = WorkflowRuntime()
        elif isinstance(runtime, TaskRuntime):
            workflow_runtime = runtime.create_workflow_runtime()
        else:
            workflow_runtime = runtime
        return workflow_runtime

    async def _prepare_agent(self, agent: Union[str, Agent], inputs: Any):
        session_id = inputs.pop(self._AGENT_CONVERSATION_ID, self._DEFAULT_AGENT_SESSION_ID)
        if isinstance(agent, str):
            agent_with_runtime = self._agent_mgr.get_agent(agent)
            runtime = await agent_with_runtime.runtime.pre_run(session_id=session_id)
            return agent_with_runtime.agent, runtime

        agent_runtime = AgentRuntime(agent.config(), self._resource_manager)
        agent_runtime = await agent_runtime.pre_run(session_id=session_id)
        return agent, agent_runtime

    def _prepare_workflow(self, workflow: Union[str, Workflow],
                          runtime: Union[Runtime, WorkflowRuntime]) -> tuple[Workflow, WorkflowRuntime]:
        if isinstance(workflow, str):
            workflow_key = workflow
        else:
            workflow_key = generate_workflow_key(workflow.config().metadata.id, workflow.config().metadata.version)

        if not self._check_is_agent_workflow(runtime, workflow_key):
            raise JiuWenBaseException(StatusCode.WORKFLOW_NOT_BOUND_TO_AGENT.code,
                                      StatusCode.WORKFLOW_NOT_BOUND_TO_AGENT.errmsg)

        workflow_runtime = self._create_workflow_runtime(runtime)
        if isinstance(workflow, str):
            workflow_instance = self._resource_manager.workflow().get_workflow(workflow_key, workflow_runtime)
        else:
            workflow_instance = workflow
        return workflow_instance, workflow_runtime

    def _prepare_agent_group(self, agent_group: Union[str, AgentGroup]):
        if isinstance(agent_group, str):
            return self._agent_group_mgr.get_agent_group(agent_group)
        return agent_group

    def _prepare_tool(self, tool: Union[str, Tool], runtime: Runtime = None):
        if not self._check_is_agent_tool(runtime, tool):
            raise JiuWenBaseException(StatusCode.TOOL_NOT_BOUND_TO_AGENT.code,
                                      StatusCode.TOOL_NOT_BOUND_TO_AGENT.errmsg)
        if not isinstance(tool, str):
            return tool
        return self._resource_manager.tool().get_tool(tool, runtime)


resource_mgr = ResourceMgr()
Runner = Runner(resource_mgr, runner_id=DEFAULT_RUNNER_ID)
