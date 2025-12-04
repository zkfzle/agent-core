#!/usr/bin/env python
# coding: utf-8
# Copyright c) Huawei Technologies Co. Ltd. 2025-2025. All rights reserved.
import os
from abc import ABC
from typing import TypedDict, Any, Optional

from openjiuwen.agent.config.base import AgentConfig
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.runtime import workflow_runtime_vars
from openjiuwen.core.runtime.constants import COMP_STREAM_CALL_TIMEOUT_KEY, STREAM_INPUT_GEN_TIMEOUT_KEY, \
    END_COMP_TEMPLATE_BATCH_READER_TIMEOUT_KEY, END_COMP_TEMPLATE_RENDER_POSITION_TIMEOUT_KEY, \
    WORKFLOW_EXECUTE_TIMEOUT, WORKFLOW_STREAM_FRAME_TIMEOUT, WORKFLOW_EXECUTE_TIMEOUT_ENV_KEY, \
    WORKFLOW_STREAM_FRAME_TIMEOUT_ENV_KEY, COMP_STREAM_CALL_TIMEOUT_ENV_KEY, STREAM_INPUT_GEN_TIMEOUT_ENV_KEY
from openjiuwen.core.workflow.workflow_config import WorkflowConfig


class MetadataLike(TypedDict):
    name: str
    event: str


def _load_env_configs() -> dict:
    env_configs = {}
    workflow_execute_timeout = os.environ.get(WORKFLOW_EXECUTE_TIMEOUT_ENV_KEY)
    if workflow_execute_timeout is not None and workflow_execute_timeout.isdigit():
        env_configs[WORKFLOW_EXECUTE_TIMEOUT] = int(workflow_execute_timeout)

    workflow_execute_timeout = workflow_runtime_vars.get(WORKFLOW_EXECUTE_TIMEOUT_ENV_KEY)
    if isinstance(workflow_execute_timeout, int):
        env_configs[WORKFLOW_EXECUTE_TIMEOUT] = workflow_execute_timeout

    workflow_stream_frame_timeout = os.environ.get(WORKFLOW_STREAM_FRAME_TIMEOUT_ENV_KEY)
    if workflow_stream_frame_timeout is not None and workflow_stream_frame_timeout.isdigit():
        env_configs[WORKFLOW_STREAM_FRAME_TIMEOUT] = int(workflow_stream_frame_timeout)

    workflow_stream_frame_timeout = workflow_runtime_vars.get(WORKFLOW_STREAM_FRAME_TIMEOUT_ENV_KEY)
    if isinstance(workflow_stream_frame_timeout, int):
        env_configs[WORKFLOW_STREAM_FRAME_TIMEOUT] = workflow_stream_frame_timeout

    comp_stream_call_timeout = os.environ.get(COMP_STREAM_CALL_TIMEOUT_ENV_KEY)
    if comp_stream_call_timeout is not None and comp_stream_call_timeout.isdigit():
        env_configs[COMP_STREAM_CALL_TIMEOUT_KEY] = int(comp_stream_call_timeout)

    comp_stream_call_timeout = workflow_runtime_vars.get(COMP_STREAM_CALL_TIMEOUT_ENV_KEY)
    if isinstance(comp_stream_call_timeout, int):
        env_configs[COMP_STREAM_CALL_TIMEOUT_KEY] = comp_stream_call_timeout

    stream_input_gen_timeout = os.environ.get(STREAM_INPUT_GEN_TIMEOUT_ENV_KEY)
    if stream_input_gen_timeout is not None and stream_input_gen_timeout.isdigit():
        env_configs[STREAM_INPUT_GEN_TIMEOUT_KEY] = int(stream_input_gen_timeout)

    stream_input_gen_timeout = workflow_runtime_vars.get(STREAM_INPUT_GEN_TIMEOUT_ENV_KEY)
    if isinstance(stream_input_gen_timeout, int):
        env_configs[STREAM_INPUT_GEN_TIMEOUT_KEY] = stream_input_gen_timeout

    return env_configs


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
        self._load_envs_()

    def set_envs(self, envs: dict[str, Any]) -> None:
        """
        set environment variables
        :param envs: envs
        """
        if not isinstance(envs, dict):
            return
        self._env.update(envs)

    def get_env(self, key: str, default: Any = None) -> Optional[Any]:
        """
        get environment variable by given key
        :param key: environment variable key
        :default key: environment variable default key
        :return: environment variable value
        """
        if key in self._env:
            return self._env[key]
        else:
            return default

    def _load_envs_(self) -> None:
        self._load_builtin_configs_()

    def _load_builtin_configs_(self):
        builtin_configs = {
            COMP_STREAM_CALL_TIMEOUT_KEY: -1,
            STREAM_INPUT_GEN_TIMEOUT_KEY: -1,
            END_COMP_TEMPLATE_BATCH_READER_TIMEOUT_KEY: 5,
            END_COMP_TEMPLATE_RENDER_POSITION_TIMEOUT_KEY: 5,
            WORKFLOW_EXECUTE_TIMEOUT: 60,
            WORKFLOW_STREAM_FRAME_TIMEOUT: -1
        }

        builtin_configs.update(_load_env_configs())

        self.set_envs(builtin_configs)

    def get_workflow_config(self, workflow_id):
        if workflow_id is None:
            raise JiuWenBaseException(StatusCode.RUNTIME_WORKFLOW_CONFIG_GET_FAILED.code,
                                      message=StatusCode.RUNTIME_WORKFLOW_CONFIG_GET_FAILED.errmsg.format(
                                          reason="workflow_id is invalid, cannot be None"))
        return self._workflow_configs.get(workflow_id)

    def get_agent_config(self):
        return self._agent_config

    def set_agent_config(self, agent_config):
        self._agent_config = agent_config

    def add_workflow_config(self, workflow_id, workflow_config):
        if workflow_id is None:
            raise JiuWenBaseException(StatusCode.RUNTIME_WORKFLOW_CONFIG_ADD_FAILED.code,
                                      message=StatusCode.RUNTIME_WORKFLOW_CONFIG_ADD_FAILED.errmsg.format(
                                          reason="workflow_id is invalid, cannot be None"))
        if workflow_config is None:
            raise JiuWenBaseException(StatusCode.RUNTIME_WORKFLOW_CONFIG_ADD_FAILED.code,
                                      message=StatusCode.RUNTIME_WORKFLOW_CONFIG_ADD_FAILED.errmsg.format(
                                          reason="workflow config is invalid, cannot be None"))
        self._workflow_configs[workflow_id] = workflow_config
