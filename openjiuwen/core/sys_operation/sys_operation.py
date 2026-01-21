#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
from typing import Optional, List

from pydantic import Field, field_validator

from openjiuwen.core.common.schema import BaseCard
from openjiuwen.core.sys_operation.base import OperationMode
from openjiuwen.core.sys_operation.local.config import LocalWorkConfig
from openjiuwen.core.sys_operation.registry import OperationRegistry
from openjiuwen.core.sys_operation.sandbox.config import SandboxGatewayConfig


class SysOperationCard(BaseCard):
    mode: OperationMode = Field(
        default=OperationMode.LOCAL,
        description="Running mode, available values: local / sandbox"
    )
    work_config: Optional[LocalWorkConfig] = Field(
        default=None,
        description="Local work config (required when mode is local)"
    )
    gateway_config: Optional[SandboxGatewayConfig] = Field(
        default=None,
        description="Sandbox gateway config (required when mode is sandbox)"
    )

    @classmethod
    @field_validator("mode")
    def mode_must_be_valid_enum(cls, v):
        """Validate that mode is a valid value in OperationMode enum"""
        if not isinstance(v, OperationMode):
            try:
                return OperationMode(v.lower())
            except ValueError:
                raise ValueError(
                    f"mode must be one of {[e.value for e in OperationMode]}, current value: {v}"
                )
        return v

    @classmethod
    @field_validator("work_config")
    def work_config_required_when_local(cls, v, values):
        """work_config cannot be None when mode is set to local"""
        mode = values.get("mode")
        if mode == OperationMode.LOCAL and v is None:
            raise ValueError("work_config is required when mode is local")
        return v

    @classmethod
    @field_validator("gateway_config")
    def gateway_config_required_when_sandbox(cls, v, values):
        """gateway_config cannot be None when mode is set to sandbox"""
        mode = values.get("mode")
        if mode == OperationMode.SANDBOX and v is None:
            raise ValueError("gateway_config is required when mode is sandbox")
        return v

    @classmethod
    @field_validator("gateway_config")
    def cannot_have_both_configs(cls, v, values):
        work_config = values.get("work_config")
        if work_config is not None and v is not None:
            raise ValueError("work_config and gateway_config cannot be configured simultaneously")
        return v


# TODO: 与Runner结合
# # step1. Agent初始化
# card = SysOperationCard(...)
# Runner.resource_mgr.add_sys_operation(card)
#     tool_card_list = sys_operation.fs().list_tools
#     # tool_id示例：card.id + "." + "fs().read_file"
#     for tool_card in tool_card_list:
#         tool_id = card.id + "." + tool_card.name
#         # tool_card.name与函数之间映射，desc中决定调用invoke/stream
#         tool = LocalFunction(tool_card, sys_operation.fs().read_file)
#         self.add_tool(tool_id, tool)
# # step2. 使用sys_operation调用方法
# sys_operation = Runner.resource_mgr.get_sys_operation(card.id)
# sys_operation.fs().read_file(...)
# sys_operation.code().execute_code(...)
# sys_operation.shell().execute_cmd(...)
#
# # step3. 使用工具调用
# sys_operation_tool = Runner.resource_mgr.get_tool(tool_id)
# sys_operation_tool.invoke(...)

class SysOperation:
    """SysOperation"""

    def __init__(self, card: SysOperationCard):
        self.mode = card.mode
        self._run_config = card.work_config if self.mode == OperationMode.LOCAL else card.gateway_config
        self._instances = {}

    def __getattr__(self, name):
        return self._get_operation(name)

    def fs(self):
        return self.fs

    def code(self):
        return self.code

    def shell(self):
        return self.shell

    def _get_operation(self, name):
        """get operation"""
        if name in self._instances:
            return self._instances[name]
        operation_info = OperationRegistry.get_operation_info(name, self.mode)
        if operation_info is None:
            return None
        operation_cls = operation_info["cls"]
        operation_desc = operation_info["description"]
        instance = operation_cls(name, self.mode, operation_desc, self._run_config)
        self._instances[name] = instance
        return instance
