#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from dataclasses import dataclass
from typing import Optional, Union, Callable

from openjiuwen.core.single_agent import BaseAgent
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.runner.drunner.remote_client.remote_agent import RemoteAgent
from openjiuwen.core.runner.drunner.server_adapter.agent_adapter import AgentAdapter
from openjiuwen.core.runner.runner_config import get_runner_config
from openjiuwen.core.session import StaticAgentRuntime
from openjiuwen.core.runner.resources_manager.abstract_manager import AbstractManager
from openjiuwen.core.runner.resources_manager.resource_manager import ResourceMgr


@dataclass
class AgentWithRuntime:
    runtime: StaticAgentRuntime
    agent: BaseAgent


AgentProvider = Callable[[], BaseAgent]


class AgentMgr(AbstractManager[AgentWithRuntime]):
    def __init__(self, resource_manager: ResourceMgr):
        super().__init__()
        self._resource_manager: ResourceMgr = resource_manager

    from openjiuwen.core.runner.drunner.server_adapter.agent_adapter import AgentAdapter
    def add_agent(self, agent_id: str, agent: Union[BaseAgent, AgentProvider, RemoteAgent, AgentAdapter]) -> None:
        self._validate_id(agent_id, StatusCode.RUNTIME_AGENT_ADD_FAILED, "single_agent")
        self._validate_resource(agent, StatusCode.RUNTIME_AGENT_ADD_FAILED, "single_agent cannot be None")

        # Define validation function for non-callable agents
        def validate_agent(agent_obj):
            if isinstance(agent, (RemoteAgent, AgentAdapter)):
                if get_runner_config().distributed_mode:
                    return agent
                raise JiuWenBaseException(
                    StatusCode.RUNTIME_AGENT_ADD_FAILED.code,
                    StatusCode.RUNTIME_AGENT_ADD_FAILED.errmsg.format(reason="RemoteAgent must be in distributed mode")
                )
            if not hasattr(agent_obj, "config"):
                raise JiuWenBaseException(
                    StatusCode.RUNTIME_AGENT_ADD_FAILED.code,
                    StatusCode.RUNTIME_AGENT_ADD_FAILED.errmsg.format(reason="Agent must have config method")
                )
            return AgentWithRuntime(
                runtime=StaticAgentRuntime(config=agent_obj.config(), resource_mgr=self._resource_manager),
                agent=agent_obj
            )

        self._add_resource(agent_id, agent, StatusCode.RUNTIME_AGENT_ADD_FAILED, validate_agent)

    def remove_agent(self, agent_id: str) -> Optional[BaseAgent | RemoteAgent | AgentAdapter]:
        self._validate_id(agent_id, StatusCode.RUNTIME_AGENT_REMOVE_FAILED, "single_agent")

        agent_with_runtime = self._remove_resource(agent_id, StatusCode.RUNTIME_AGENT_REMOVE_FAILED)
        if isinstance(agent_with_runtime, (RemoteAgent, AgentAdapter)):
            return agent_with_runtime
        return agent_with_runtime.agent if agent_with_runtime else None

    def get_agent(self, agent_id: str) -> Optional[AgentWithRuntime | RemoteAgent]:

        self._validate_id(agent_id, StatusCode.RUNTIME_AGENT_GET_FAILED, "single_agent")

        # Define function to create single_agent from provider
        def create_agent_from_provider(provider):
            agent = provider()
            if not hasattr(agent, "config"):
                raise JiuWenBaseException(
                    StatusCode.RUNTIME_AGENT_GET_FAILED.code,
                    StatusCode.RUNTIME_AGENT_GET_FAILED.errmsg.format(
                        reason="Agent returned by provider must have config method"
                    )
                )
            return AgentWithRuntime(
                runtime=StaticAgentRuntime(config=agent.config(), resource_mgr=self._resource_manager),
                agent=agent
            )

        return self._get_resource(agent_id, StatusCode.RUNTIME_AGENT_GET_FAILED, create_agent_from_provider)
