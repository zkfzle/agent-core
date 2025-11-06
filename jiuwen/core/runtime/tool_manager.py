#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import List, Optional, Tuple, Union

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.runtime.abstract_manager import AbstractManager
from jiuwen.core.tracer.decorator import decrate_tool_with_trace
from jiuwen.core.utils.llm.messages import ToolInfo, Function
from jiuwen.core.utils.tool.base import Tool
from jiuwen.core.runtime.thread_safe_dict import ThreadSafeDict
from jiuwen.core.utils.tool.mcp.base import ToolServerConfig, McpToolInfo

ToolProvider = lambda: Tool

class ToolMgr(AbstractManager[Tool]):
    def __init__(self) -> None:
        super().__init__()
        self._tool_infos: dict[str, ToolInfo] = {}
        self._server_tool_infos : dict[str, List[McpToolInfo]] = {}
        self._server_configs : dict[str, ToolServerConfig] = {}

    def add_tool(self, tool_id: str, tool: Union[Tool, ToolProvider]) -> None:
        self._validate_id(tool_id, StatusCode.RUNTIME_TOOL_GET_FAILED, "tool")
        self._validate_resource(tool, StatusCode.RUNTIME_TOOL_GET_FAILED, "tool is invalid, can not be None")

        # Define validation function for non-callable tools
        def validate_tool(tool_obj):
            # Store tool info
            if hasattr(tool_obj, "get_tool_info"):
                self._tool_infos[tool_id] = tool_obj.get_tool_info()
            else:
                self._tool_infos[tool_id] = ToolInfo(function=Function())
            return tool_obj

        self._add_resource(tool_id, tool, StatusCode.RUNTIME_TOOL_GET_FAILED, validate_tool)

    def add_tools(self, tools: List[Tuple[str, Union[Tool, ToolProvider]]]):
        if not tools:
            return
        for id, tool in tools:
            self.add_tool(id, tool)

    def find_tool_by_name(self, name: str) -> Optional[Tool]:
        self._validate_id(name, StatusCode.RUNTIME_TOOL_GET_FAILED, "name")

        # Define function to create tool from provider
        def create_tool_from_provider(provider):
            tool = provider()
            # Store tool info
            if hasattr(tool, "get_tool_info"):
                self._tool_infos[name] = tool.get_tool_info()
            else:
                self._tool_infos[name] = ToolInfo(function=Function())
            return tool

        return self._get_resource(name, StatusCode.RUNTIME_TOOL_GET_FAILED, create_tool_from_provider)

    def get_tool(self, tool_id: str, runtime=None) -> Optional[Tool]:
        # Validate ID using base class method
        self._validate_id(tool_id, StatusCode.RUNTIME_TOOL_GET_FAILED, "tool")

        try:
            tool = self.find_tool_by_name(tool_id)
            return decrate_tool_with_trace(tool, runtime)
        except JiuWenBaseException:
            raise
        except Exception as e:
            self._handle_exception(e, StatusCode.RUNTIME_TOOL_GET_FAILED, "get")

    def remove_tool(self, tool_id: str) -> Optional[Tool]:
        if tool_id is None:
            return None

        try:
            tool = self._remove_resource(tool_id, StatusCode.RUNTIME_TOOL_GET_FAILED)
            self._tool_infos.pop(tool_id, None)
            return tool
        except Exception as e:
            self._handle_exception(e, StatusCode.RUNTIME_TOOL_GET_FAILED, "remove")

    def get_tool_infos(self, tool_ids: List[str] = None, *, tool_server_name: str) -> Optional[List[Union[ToolInfo, McpToolInfo]]]:
        try:
            if not tool_ids:
                return [info for info in self._tool_infos.values()]

            infos = []
            for tool_id in tool_ids:
                self._validate_id(tool_id, StatusCode.RUNTIME_TOOL_TOOL_INFO_GET_FAILED, "tool")
                infos.append(self._tool_infos.get(tool_id))
            return infos
        except JiuWenBaseException:
            raise
        except Exception as e:
            self._handle_exception(e, StatusCode.RUNTIME_TOOL_TOOL_INFO_GET_FAILED, "get_tool_info")

    async def add_tool_servers(self, server_config: Union[ToolServerConfig, List[ToolServerConfig]], *,
                               wait_for_connect: bool = True):
        pass

    def remove_tool_server(self, tool_server_name: str):
        pass
