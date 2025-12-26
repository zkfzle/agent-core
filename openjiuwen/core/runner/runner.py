#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import asyncio
from typing import Union, Any, List, Optional

from openjiuwen.core.single_agent.config import AgentConfig
from openjiuwen.core.single_agent.agent import BaseAgent
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.logging import logger
from openjiuwen.core.common.security.user_config import UserConfig
from openjiuwen.core.context_engine.base import Context
from openjiuwen.core.runner.agent_group import AgentGroup
from openjiuwen.core.runner.drunner.dmessage_queue.dsubscription.reply_topic_subscription import ReplyTopicSubscription
from openjiuwen.core.runner.drunner.dmessage_queue.message_queue_factory import MessageQueueFactory
from openjiuwen.core.runner.drunner.remote_client.remote_agent import RemoteAgent
from openjiuwen.core.runner.drunner.server_adapter.agent_adapter import AgentAdapter
from openjiuwen.core.runner.runner_config import RunnerConfig, DEFAULT_RUNNER_CONFIG, set_runner_config, \
    get_runner_config
from openjiuwen.core.session import StaticAgentRuntime
from openjiuwen.core.session import get_default_inmemory_checkpointer
from openjiuwen.core.runner.resources_manager.agent_group_manager import AgentGroupProvider, AgentGroupMgr
from openjiuwen.core.runner.resources_manager.agent_manager import AgentProvider, AgentMgr
from openjiuwen.core.runner.resources_manager.resource_manager import ResourceMgr
from openjiuwen.core.session import Runtime
from openjiuwen.core.session import WorkflowRuntime
from openjiuwen.core.session import TaskRuntime
from openjiuwen.core.session.stream import BaseStreamMode
from openjiuwen.core.foundation.tool import Tool
from openjiuwen.core.foundation.tool import McpToolInfo
from openjiuwen.core.workflow import generate_workflow_key
from openjiuwen.core.workflow.base import Workflow

AGENT_ADAPTER = "agent_adapter_"


# mock
class LocalMessageQueue:
    async def start(self):
        pass

    async def stop(self):
        pass


DEFAULT_RUNNER_ID = "global"


class Runner:
    """
    Runner
    """

    _DEFAULT_AGENT_SESSION_ID = "default_session"

    _AGENT_CONVERSATION_ID = "conversation_id"

    def __init__(self, resource_manager: ResourceMgr, runner_id: str = "", config: RunnerConfig = None):
        self._runner_id = runner_id
        self._resource_manager = resource_manager
        self._message_queue = LocalMessageQueue()
        self._agent_group_mgr: AgentGroupMgr = AgentGroupMgr()
        self._agent_mgr: AgentMgr = AgentMgr(resource_manager)
        if config is not None:
            set_runner_config(config)
        else:
            set_runner_config(DEFAULT_RUNNER_CONFIG)
        # Distributed system related components
        self.system_reply_sub: ReplyTopicSubscription | None = None
        self._distribute_message_queue = None

    def set_config(self, config: RunnerConfig):
        set_runner_config(config)

    def get_config(self):
        return get_runner_config()

    async def start(self) -> bool:
        if get_runner_config().distributed_mode:
            # start dmq
            self._distribute_message_queue = MessageQueueFactory.create(
                get_runner_config().distributed_config.message_queue_config)
            self._distribute_message_queue.start()
            # start reply topic sub
            self.system_reply_sub = ReplyTopicSubscription(self._distribute_message_queue)
            self.system_reply_sub.activate()
        return await self._message_queue.start()

    async def stop(self):
        logger.info("[Runner] Stopping...")
        if get_runner_config().distributed_mode:
            # 2. Stop ReplyTopicSubscription, clean up collector
            if self.system_reply_sub:
                await self.system_reply_sub.deactivate()
                self.system_reply_sub = None
            # 3. Stop MQ
            if self._distribute_message_queue:
                await self._distribute_message_queue.stop()
                self._distribute_message_queue = None
        await resource_mgr.tool().stop()
        result = await self._message_queue.stop()
        logger.info("[Runner] Stopped...")
        return result

    def message_queue(self):
        return self._message_queue

    def distribute_message_queue(self):
        return self._distribute_message_queue

    async def add_agent_group(self, agent_group_id: str, agent_group: Union[AgentGroup, AgentGroupProvider]):
        self._agent_group_mgr.add_agent_group(agent_group_id, agent_group)
        # Only subscribe to message queue for AgentGroup with get_topic method
        # BaseGroup (HierarchicalGroup etc.) uses different message mechanism
        if hasattr(agent_group, 'get_topic') and callable(agent_group.get_topic):
            topic = agent_group.get_topic()
            if topic is not None:
                subscription = await self._message_queue.subscribe(topic)
                agent_group.set_subscription(subscription)

    async def remove_agent_group(self, agent_group_id: str) -> Union[AgentGroup, AgentGroupProvider]:
        agent_group = self._agent_group_mgr.remove_agent_group(agent_group_id)
        # Only unsubscribe for AgentGroup with get_topic method and subscription
        if agent_group and hasattr(agent_group, 'get_topic') and callable(agent_group.get_topic):
            topic = agent_group.get_topic()
            if topic is not None and hasattr(agent_group, '_subscription'):
                await self._message_queue.unsubscribe(topic, agent_group.get_subscription)
        return agent_group

    def add_agent(self, agent_id, agent: Union[BaseAgent, AgentProvider, RemoteAgent]):
        if get_runner_config().distributed_mode:
            if not isinstance(agent, RemoteAgent):
                mqAgentAdapter = AgentAdapter(agent_id)
                mqAgentAdapter.start()
                self._agent_mgr.add_agent(AGENT_ADAPTER + agent_id, mqAgentAdapter)
        self._agent_mgr.add_agent(agent_id, agent)

    def remove_agent(self, agent_id) -> Union[BaseAgent, AgentProvider]:
        if get_runner_config().distributed_mode:
            adapter = self._agent_mgr.remove_agent(AGENT_ADAPTER + agent_id)
            if adapter is not None:
                adapter.stop()
        return self._agent_mgr.remove_agent(agent_id)

    async def run_workflow(self, workflow: Union[str, Workflow], inputs: Any,
                           *, runtime: Union[Runtime, WorkflowRuntime] = None, context: Context = None):
        workflow_instance, workflow_runtime = await self._prepare_workflow(workflow, runtime)
        return await workflow_instance.invoke(inputs, runtime=workflow_runtime, context=context)

    async def run_workflow_streaming(self, workflow: Union[str, Workflow], inputs: Any,
                                     *, runtime: Union[Runtime, WorkflowRuntime] = None,
                                     stream_modes: list[BaseStreamMode] = None, context: Context = None):
        workflow_instance, workflow_runtime = await self._prepare_workflow(workflow, runtime)
        async for chunk in workflow_instance.stream(inputs, runtime=workflow_runtime,
                                                    stream_modes=stream_modes, context=context):
            yield chunk

    async def run_agent(self, agent: Union[str, BaseAgent], inputs: Any):
        agent_instance, agent_runtime = await self._prepare_agent(agent, inputs)
        if isinstance(agent_instance, RemoteAgent):
            res = await agent_instance.invoke(inputs)
        elif isinstance(agent_instance, BaseAgent):
            # ControllerAgent handles its own runtime lifecycle
            res = await agent_instance.invoke(inputs, runtime=None)
        else:
            res = await agent_instance.invoke(inputs, agent_runtime)
            await agent_runtime.post_run()
        return res

    async def run_agent_streaming(self, agent: Union[str, BaseAgent], inputs: Any):
        agent_instance, agent_runtime = await self._prepare_agent(agent, inputs)
        if isinstance(agent_instance, RemoteAgent):
            async for chunk in agent_instance.stream(inputs):
                yield chunk
        elif isinstance(agent_instance, BaseAgent):
            # ControllerAgent handles its own runtime lifecycle
            async for chunk in agent_instance.stream(inputs, runtime=None):
                yield chunk

    async def run_agent_group(self, agent_group: Union[str, AgentGroup], inputs: Any):
        agent_group_instance = self._prepare_agent_group(agent_group)
        return await agent_group_instance.invoke(inputs)

    async def run_agent_group_streaming(self, agent_group: Union[str, AgentGroup], inputs: Any):
        agent_group_instance = self._prepare_agent_group(agent_group)
        async for chunk in agent_group_instance.stream(inputs):
            yield chunk

    async def run_tool(self, tool: Union[str, Tool], inputs, *, runtime: Runtime = None):
        tool_instance = self._prepare_tool(tool, runtime)
        if tool_instance is None:
            logger.error(f"{self.__class__.__name__} tool not found.")
            if UserConfig.is_sensitive():
                raise JiuWenBaseException(StatusCode.TOOL_NOT_FOUND.code,
                                          f"{self.__class__.__name__} tool not found.")
            else:
                tool_name = tool if isinstance(tool, str) else getattr(tool, 'name', 'unknown')
                raise JiuWenBaseException(StatusCode.TOOL_NOT_FOUND.code,
                                          f"{self.__class__.__name__} tool not found: {tool_name}.")
        return await tool_instance.invoke(inputs, runtime=runtime)

    async def run_tool_streaming(self, tool: Union[str, Tool], inputs, *, runtime: Runtime = None):
        tool_instance = self._prepare_tool(tool, runtime)
        if tool_instance is None:
            logger.error(f"{self.__class__.__name__} tool not found.")
            if UserConfig.is_sensitive():
                raise JiuWenBaseException(StatusCode.TOOL_NOT_FOUND.code,
                                          f"{self.__class__.__name__} tool not found.")
            else:
                tool_name = tool if isinstance(tool, str) else getattr(tool, 'name', 'unknown')
                raise JiuWenBaseException(StatusCode.TOOL_NOT_FOUND.code,
                                          f"{self.__class__.__name__} tool not found: {tool_name}.")
        async for chunk in tool_instance.astream(inputs, runtime=runtime):
            yield chunk

    async def list_tools(self, tool_server_name: Union[str, List[str]], *, name_delimiter: str = None) -> Union[
        Optional[List[McpToolInfo]], List[Optional[List[McpToolInfo]]]]:
        if not tool_server_name:
            return None
        tool_mgr = self._resource_manager.tool()
        single = isinstance(tool_server_name, str)
        names = [tool_server_name] if single else tool_server_name
        results = [tool_mgr.get_tool_infos(tool_server_name=n, name_delimiter=name_delimiter) for n in names]
        return results[0] if single else results

    async def release(self, session_id: str):
        await get_default_inmemory_checkpointer().release(session_id)

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

    @classmethod
    def _is_called_by_agent(cls, runtime: Runtime) -> bool:
        return runtime and isinstance(runtime, TaskRuntime)

    @classmethod
    def _create_workflow_runtime(cls, runtime):
        # Convert workflow runtime
        if not runtime:
            workflow_runtime = WorkflowRuntime()
        elif isinstance(runtime, TaskRuntime):
            workflow_runtime = runtime.create_workflow_runtime()
        else:
            workflow_runtime = runtime
        return workflow_runtime

    async def _prepare_agent(self, agent: Union[str, BaseAgent], inputs: Any):
        session_id = inputs.get(self._AGENT_CONVERSATION_ID, self._DEFAULT_AGENT_SESSION_ID)
        if isinstance(agent, str):
            agent_with_runtime = self._agent_mgr.get_agent(agent)
            if agent_with_runtime is None:
                raise JiuWenBaseException(StatusCode.AGENT_NOT_FOUND.code,
                                          StatusCode.AGENT_NOT_FOUND.errmsg.format(agent))
            if isinstance(agent_with_runtime, RemoteAgent):
                # Remote single_agent does not add runtime, keep sessionId in input
                if self._AGENT_CONVERSATION_ID not in inputs:
                    inputs[self._AGENT_CONVERSATION_ID] = session_id
                return agent_with_runtime, None
            task_runtime = TaskRuntime(inner=await agent_with_runtime.runtime.create_agent_runtime(session_id, inputs))
            return agent_with_runtime.agent, task_runtime
        agent_runtime = StaticAgentRuntime(agent.config(), resource_mgr=self._resource_manager)
        task_runtime = TaskRuntime(inner=await agent_runtime.create_agent_runtime(session_id, inputs))
        return agent, task_runtime

    async def _prepare_workflow(self, workflow: Union[str, Workflow],
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
            workflow_instance = await self._resource_manager.workflow().get_workflow(workflow_key, workflow_runtime)
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
Runner = Runner(resource_mgr, runner_id=DEFAULT_RUNNER_ID, config=DEFAULT_RUNNER_CONFIG)
