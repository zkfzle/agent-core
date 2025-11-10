#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import List, Optional, Tuple, Union

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.tracer.decorator import decrate_tool_with_trace
from jiuwen.core.utils.llm.messages import ToolInfo, Function
from jiuwen.core.utils.tool.base import Tool
from jiuwen.core.runtime.thread_safe_dict import ThreadSafeDict
from jiuwen.core.utils.tool.mcp.base import ToolServerConfig, McpToolInfo

ToolProvider = lambda: Tool

class ToolMgr:
    def __init__(self) -> None:
        self._tools: ThreadSafeDict[str, Tool] = ThreadSafeDict()
        self._tool_providers: ThreadSafeDict[str, ToolProvider] = ThreadSafeDict()
        self._tool_infos: ThreadSafeDict[str, ToolInfo] = ThreadSafeDict()
        self._server_tool_infos : ThreadSafeDict[str, List[McpToolInfo]] = ThreadSafeDict()
        self._server_configs : ThreadSafeDict[str, ToolServerConfig] = ThreadSafeDict()

    def add_tool(self, tool_id: str, tool: Union[Tool, ToolProvider]) -> None:
        if tool_id is None or tool_id.strip() == "":
            raise JiuWenBaseException(StatusCode.RUNTIME_TOOL_GET_FAILED.code,
                                      StatusCode.RUNTIME_TOOL_GET_FAILED.errmsg.format(
                                          reason="tool_id is invalid, can not be None or empty"))
        if tool is None:
            raise JiuWenBaseException(StatusCode.RUNTIME_TOOL_GET_FAILED.code,
                                      StatusCode.RUNTIME_TOOL_GET_FAILED.errmsg.format(
                                          reason="tool is invalid, can not be None"))

        try:
            if callable(tool):
                self._tool_providers[tool_id] = tool
            else:
                self._tools[tool_id] = tool
                if hasattr(tool, "get_tool_info"):
                    self._tool_infos[tool_id] = tool.get_tool_info()
                else:
                    self._tool_infos[tool_id] = ToolInfo(function=Function())
        except Exception as e:
            raise JiuWenBaseException(StatusCode.RUNTIME_TOOL_GET_FAILED.code,
                                      StatusCode.RUNTIME_TOOL_GET_FAILED.errmsg.format(
                                          reason=f"Failed to add tool: {str(e)}"))

    def add_tools(self, tools: List[Tuple[str, Union[Tool, ToolProvider]]]):
        if not tools:
            return
        for id, tool in tools:
            self.add_tool(id, tool)

    def find_tool_by_name(self, name: str) -> Optional[Tool]:
        if name is None or name.strip() == "":
            raise JiuWenBaseException(StatusCode.RUNTIME_TOOL_GET_FAILED.code,
                                      StatusCode.RUNTIME_TOOL_GET_FAILED.errmsg.format(
                                          reason="name is invalid, can not be None or empty"))

        try:
            tool = self._tools.get(name)
            if tool:
                return tool

            provider = self._tool_providers.get(name)
            if provider:
                tool = provider()
                self._tools[name] = tool
                if hasattr(tool, "get_tool_info"):
                    self._tool_infos[name] = tool.get_tool_info()
                else:
                    self._tool_infos[name] = ToolInfo(function=Function())
                return tool
            return None
        except Exception as e:
            raise JiuWenBaseException(StatusCode.RUNTIME_TOOL_GET_FAILED.code,
                                      StatusCode.RUNTIME_TOOL_GET_FAILED.errmsg.format(
                                          reason=f"Failed to find tool: {str(e)}"))

    def get_tool(self, tool_id: str, runtime=None) -> Optional[Tool]:
        if tool_id is None or tool_id.strip() == "":
            raise JiuWenBaseException(StatusCode.RUNTIME_TOOL_GET_FAILED.code,
                                      StatusCode.RUNTIME_TOOL_GET_FAILED.errmsg.format(
                                          reason="tool_id is invalid, can not be None or empty"))

        try:
            tool = self.find_tool_by_name(tool_id)
            return decrate_tool_with_trace(tool, runtime)
        except JiuWenBaseException:
            raise
        except Exception as e:
            raise JiuWenBaseException(StatusCode.RUNTIME_TOOL_GET_FAILED.code,
                                      StatusCode.RUNTIME_TOOL_GET_FAILED.errmsg.format(
                                          reason=f"Failed to get tool: {str(e)}"))

    def remove_tool(self, tool_id: str) -> Optional[Tool]:
        if tool_id is None:
            return None

        try:
            tool = self._tools.pop(tool_id, None)
            self._tool_providers.pop(tool_id, None)
            self._tool_infos.pop(tool_id, None)
            return tool
        except Exception as e:
            raise JiuWenBaseException(StatusCode.RUNTIME_TOOL_GET_FAILED.code,
                                      StatusCode.RUNTIME_TOOL_GET_FAILED.errmsg.format(
                                          reason=f"Failed to remove tool: {str(e)}"))

    def get_tool_infos(self, tool_id: List[str] = None, *, tool_server_name: str = None) -> Optional[List[Union[ToolInfo, McpToolInfo]]]:
        try:
            if not tool_id:
                return [info for info in self._tool_infos.values()]
            infos = []
            for id in tool_id:
                if id is None or id.strip() == "":
                    raise JiuWenBaseException(StatusCode.RUNTIME_TOOL_TOOL_INFO_GET_FAILED.code,
                                              StatusCode.RUNTIME_TOOL_TOOL_INFO_GET_FAILED.errmsg.format(
                                                  reason="tool_id is invalid, can not be None or empty"))
                infos.append(self._tool_infos.get(id))
            return infos
        except JiuWenBaseException:
            raise
        except Exception as e:
            raise JiuWenBaseException(StatusCode.RUNTIME_TOOL_TOOL_INFO_GET_FAILED.code,
                                      StatusCode.RUNTIME_TOOL_TOOL_INFO_GET_FAILED.errmsg.format(
                                          reason=f"Failed to get tool infos: {str(e)}"))

    async def add_tool_servers(self, server_config: Union[ToolServerConfig, List[ToolServerConfig]], *,
                               wait_for_connect: bool = True):
        pass

    def remove_tool_server(self, tool_server_name: str):
        pass
