#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from abc import ABC
from typing import TypedDict, Any, Optional

from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.context.controller_context.workflow_manager import generate_workflow_key
from jiuwen.core.workflow.workflow_config import WorkflowConfig


class MetadataLike(TypedDict):
    name: str
    event: str




class Config(ABC):
    """
    Config is the class defines the basic infos of workflow
    """

    def __init__(self):
        """
        initialize the config
        """
        self._callback_metadata: dict[str, MetadataLike] = {}
        self._env: dict = {}
        self._workflow_configs: dict[str, WorkflowConfig] = {}
        self._agent_config: AgentConfig = None

    def set_envs(self, envs: dict[str, str]) -> None:
        """
        set environment variables
        :param envs: envs
        """
        self._env.update(envs)

    def get_env(self, key: str) -> Any:
        """
        get environment variable by given key
        :param key: environment variable key
        :return: environment variable value
        """
        if key in self._env:
            return self._env[key]
        else:
            return None

    def __load_envs__(self) -> None:
        pass

    def get_workflow_config(self, workflow_id):
        return self._workflow_configs.get(workflow_id)

    def get_agent_config(self):
        return self._agent_config

    def set_agent_config(self, agent_config):
        self._agent_config = agent_config

    def add_workflow_config(self, workflow_id, workflow_config):
        self._workflow_configs[workflow_id] = workflow_config


class WrappedWorkflowConfig(Config):
    """
    Config is the class defines the basic infos of workflow
    """

    def __init__(self, workflow_config: WorkflowConfig, base: Config):
        """
        initialize the config
        """
        super().__init__()
        self._workflow_config = workflow_config
        self._base = base
        if self._workflow_config.metadata:
            workflow_id = generate_workflow_key(self._workflow_config.metadata.id,
                                                self._workflow_config.metadata.version)
            self._base.add_workflow_config(workflow_id, self)

    def get_env(self, key: str) -> Any:
        return self._base.get_env(key)

    def set_envs(self, envs: dict[str, str]) -> None:
        self._base.set_envs(envs)

    def get_agent_config(self):
        return self._base.get_agent_config()

    def get_workflow_config(self, workflow_id=None):
        if not workflow_id:
            return self._workflow_config
        else:
            return self._base.get_workflow_config(workflow_id)
