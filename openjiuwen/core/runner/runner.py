#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import Union, Any, List, Optional

from openjiuwen.core.multi_agent import BaseGroup
from openjiuwen.core.runner.message_queue_base import LocalMessageQueue
from openjiuwen.core.single_agent import AgentConfig, BaseAgent
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.logging import logger
from openjiuwen.core.common.security.user_config import UserConfig
from openjiuwen.core.context_engine import Context
from openjiuwen.core.runner.drunner.dmessage_queue.dsubscription.reply_topic_subscription import ReplyTopicSubscription
from openjiuwen.core.runner.drunner.dmessage_queue.message_queue_factory import MessageQueueFactory
from openjiuwen.core.runner.drunner.remote_client.remote_agent import RemoteAgent
from openjiuwen.core.runner.runner_config import RunnerConfig, DEFAULT_RUNNER_CONFIG, set_runner_config, \
    get_runner_config
from openjiuwen.core.session import StaticAgentSession
from openjiuwen.core.session import get_default_inmemory_checkpointer
from openjiuwen.core.runner.resources_manager.resource_manager import ResourceMgr
from openjiuwen.core.session import Session
from openjiuwen.core.session import WorkflowSession
from openjiuwen.core.session import TaskSession
from openjiuwen.core.session.stream import BaseStreamMode
from openjiuwen.core.foundation.tool import Tool
from openjiuwen.core.foundation.tool import McpToolCard
from openjiuwen.core.workflow import generate_workflow_key
from openjiuwen.core.workflow import Workflow


class Runner:
    """
    Runner
    """
    DEFAULT_RUNNER_ID = "global"

    _DEFAULT_AGENT_SESSION_ID = "default_session"

    _AGENT_CONVERSATION_ID = "conversation_id"

    def __init__(self, runner_id: str = DEFAULT_RUNNER_ID, config: RunnerConfig = None):
        self._runner_id = runner_id
        self._resource_manager = ResourceMgr()
        self._message_queue = LocalMessageQueue()
        if config is not None:
            set_runner_config(config)
        else:
            set_runner_config(DEFAULT_RUNNER_CONFIG)
        # Distributed system related components
        self.system_reply_sub: ReplyTopicSubscription | None = None
        self._distribute_message_queue = None

    @property
    def resource_mgr(self) -> ResourceMgr:
        return self._resource_manager

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

        result = await self._message_queue.stop()
        await self._resource_manager.release()
        logger.info("[Runner] Stopped...")
        return result

    async def run_workflow(self, workflow: Union[str, Workflow], inputs: Any,
                           *, session: Union[Session, WorkflowSession] = None, context: Context = None):
        workflow_instance, workflow_session = await self._prepare_workflow(workflow, session)
        return await workflow_instance.invoke(inputs, session=workflow_session, context=context)

    async def run_workflow_streaming(self, workflow: Union[str, Workflow], inputs: Any,
                                     *, session: Union[Session, WorkflowSession] = None,
                                     stream_modes: list[BaseStreamMode] = None, context: Context = None):
        workflow_instance, workflow_session = await self._prepare_workflow(workflow, session)
        async for chunk in workflow_instance.stream(inputs, session=workflow_session,
                                                    stream_modes=stream_modes, context=context):
            yield chunk

    async def run_agent(self, agent: Union[str, BaseAgent], inputs: Any):
        agent_instance, agent_session = await self._prepare_agent(agent, inputs)
        if isinstance(agent_instance, RemoteAgent):
            res = await agent_instance.invoke(inputs)
        elif isinstance(agent_instance, BaseAgent):
            # ControllerAgent handles its own session lifecycle
            res = await agent_instance.invoke(inputs, session=None)
        else:
            res = await agent_instance.invoke(inputs, agent_session)
            await agent_session.post_run()
        return res

    async def run_agent_streaming(self, agent: Union[str, BaseAgent], inputs: Any):
        agent_instance, agent_session = await self._prepare_agent(agent, inputs)
        if isinstance(agent_instance, RemoteAgent):
            async for chunk in agent_instance.stream(inputs):
                yield chunk
        elif isinstance(agent_instance, BaseAgent):
            # ControllerAgent handles its own session lifecycle
            async for chunk in agent_instance.stream(inputs, session=None):
                yield chunk

    async def run_agent_group(self, agent_group: Union[str, BaseGroup], inputs: Any):
        agent_group_instance = self._prepare_agent_group(agent_group)
        return await agent_group_instance.invoke(inputs)

    async def run_agent_group_streaming(self, agent_group: Union[str, BaseGroup], inputs: Any):
        agent_group_instance = self._prepare_agent_group(agent_group)
        async for chunk in agent_group_instance.stream(inputs):
            yield chunk

    async def run_tool(self, tool: Union[str, Tool], inputs, *, session: Session = None):
        tool_instance = self._prepare_tool(tool, session)
        if tool_instance is None:
            logger.error(f"{self.__class__.__name__} tool not found.")
            if UserConfig.is_sensitive():
                raise JiuWenBaseException(StatusCode.TOOL_NOT_FOUND.code,
                                          f"{self.__class__.__name__} tool not found.")
            else:
                tool_name = tool if isinstance(tool, str) else getattr(tool, 'name', 'unknown')
                raise JiuWenBaseException(StatusCode.TOOL_NOT_FOUND.code,
                                          f"{self.__class__.__name__} tool not found: {tool_name}.")
        return await tool_instance.invoke(inputs, session=session)

    async def run_tool_streaming(self, tool: Union[str, Tool], inputs, *, session: Session = None):
        tool_instance = self._prepare_tool(tool, session)
        if tool_instance is None:
            logger.error(f"{self.__class__.__name__} tool not found.")
            if UserConfig.is_sensitive():
                raise JiuWenBaseException(StatusCode.TOOL_NOT_FOUND.code,
                                          f"{self.__class__.__name__} tool not found.")
            else:
                tool_name = tool if isinstance(tool, str) else getattr(tool, 'name', 'unknown')
                raise JiuWenBaseException(StatusCode.TOOL_NOT_FOUND.code,
                                          f"{self.__class__.__name__} tool not found: {tool_name}.")
        async for chunk in tool_instance.astream(inputs, session=session):
            yield chunk

    async def list_tools(
            self, tool_server_name: Union[str, List[str]], *, name_delimiter: str = None
    ) -> Union[Optional[List[McpToolCard]], List[Optional[List[McpToolCard]]]]:
        if not tool_server_name:
            return None
        tool_mgr = self._resource_manager.tool()
        single = isinstance(tool_server_name, str)
        names = [tool_server_name] if single else tool_server_name
        results = [tool_mgr.get_tool_infos(tool_server_name=n, name_delimiter=name_delimiter) for n in names]
        return results[0] if single else results

    async def release(self, session_id: str):
        await get_default_inmemory_checkpointer().release(session_id)

    def _check_is_agent_tool(self, session, tool) -> bool:
        if not self._is_called_by_agent(session):
            return True
        agent_config: AgentConfig = session.get_agent_config()

        if isinstance(tool, str):
            tool_name = tool
        else:
            tool_name = tool.name

        for agent_tool in agent_config.tools:
            if agent_tool == tool_name:
                return True
        return False

    def _check_is_agent_workflow(self, session, workflow_key) -> bool:
        if not self._is_called_by_agent(session):
            return True
        agent_config: AgentConfig = session.get_agent_config()

        for workflow_schema in agent_config.workflows:
            if generate_workflow_key(workflow_schema.id, workflow_schema.version) == workflow_key:
                return True
        return False

    @classmethod
    def _is_called_by_agent(cls, session: Session) -> bool:
        return session and isinstance(session, TaskSession)

    @classmethod
    def _create_workflow_session(cls, session):
        # Convert workflow session
        if not session:
            workflow_session = WorkflowSession()
        elif isinstance(session, TaskSession):
            workflow_session = session.create_workflow_session()
        else:
            workflow_session = session
        return workflow_session

    async def _prepare_agent(self, agent: Union[str, BaseAgent], inputs: Any):
        session_id = inputs.get(self._AGENT_CONVERSATION_ID, self._DEFAULT_AGENT_SESSION_ID)
        if isinstance(agent, str):
            agent_with_session = await self._resource_manager.get_agent(id=agent)
            if agent_with_session is None:
                raise JiuWenBaseException(StatusCode.AGENT_NOT_FOUND.code,
                                          StatusCode.AGENT_NOT_FOUND.errmsg.format(agent))
            if isinstance(agent_with_session, RemoteAgent):
                # Remote single_agent does not add session, keep sessionId in input
                if self._AGENT_CONVERSATION_ID not in inputs:
                    inputs[self._AGENT_CONVERSATION_ID] = session_id
                return agent_with_session, None
            task_session = TaskSession(inner=(await agent_with_session.session.create_agent_session(session_id, inputs)))
            return agent_with_session.agent, task_session
        agent_session = StaticAgentSession(agent.config(), resource_mgr=self._resource_manager)
        task_session = TaskSession(inner=await agent_session.create_agent_session(session_id, inputs))
        return agent, task_session

    async def _prepare_workflow(self, workflow: Union[str, Workflow],
                                session: Union[Session, WorkflowSession]) -> tuple[Workflow, WorkflowSession]:
        if isinstance(workflow, str):
            workflow_key = workflow
        else:
            workflow_key = generate_workflow_key(workflow.config().metadata.id, workflow.config().metadata.version)

        if not self._check_is_agent_workflow(session, workflow_key):
            raise JiuWenBaseException(StatusCode.WORKFLOW_NOT_BOUND_TO_AGENT.code,
                                      StatusCode.WORKFLOW_NOT_BOUND_TO_AGENT.errmsg)

        workflow_session = self._create_workflow_session(session)
        if isinstance(workflow, str):
            workflow_instance = await self._resource_manager.get_workflow(id=workflow_key, session=workflow_session)
        else:
            workflow_instance = workflow
        return workflow_instance, workflow_session

    def _prepare_agent_group(self, agent_group: Union[str, BaseGroup]):
        if isinstance(agent_group, str):
            return self._resource_manager.get_agent_group(id=agent_group)
        return agent_group

    def _prepare_tool(self, tool: Union[str, Tool], session: Session = None):
        if not self._check_is_agent_tool(session, tool):
            raise JiuWenBaseException(StatusCode.TOOL_NOT_BOUND_TO_AGENT.code,
                                      StatusCode.TOOL_NOT_BOUND_TO_AGENT.errmsg)
        if not isinstance(tool, str):
            return tool
        return self._resource_manager.get_tool(id=tool, session=session)

    def _pubsub(self):
        return self._message_queue

    def _dist_pubsub(self):
        return self._distribute_message_queue


Runner = Runner(config=DEFAULT_RUNNER_CONFIG)
