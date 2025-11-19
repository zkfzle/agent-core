#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import List, Optional, Tuple, Union, Callable

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.runtime.resources_manager.abstract_manager import AbstractManager
from openjiuwen.core.tracer.decorator import decorate_tool_with_trace
from openjiuwen.core.utils.tool.schema import ToolInfo
from openjiuwen.core.utils.tool.base import Tool
from openjiuwen.core.utils.tool.mcp.base import (
    ToolServerConfig,
    McpToolInfo,
    McpToolClient,
    SseClient,
    StdioClient,
    PlaywrightClient,
    MCPTool
)
from openjiuwen.core.common.logging import logger

ToolProvider = Callable[[], Tool]

class ToolMgr(AbstractManager[Tool]):
    def __init__(self) -> None:
        super().__init__()
        self._tool_infos: dict[str, ToolInfo] = {}
        self._server_tool_infos : dict[str, List[McpToolInfo]] = {}
        self._server_configs : dict[str, ToolServerConfig] = {}
        self._mcp_clients: dict[str, McpToolClient] = {}

    def add_tool(self, tool_id: str, tool: Union[Tool, ToolProvider]) -> None:
        self._validate_id(tool_id, StatusCode.RUNTIME_TOOL_GET_FAILED, "tool")
        self._validate_resource(tool, StatusCode.RUNTIME_TOOL_GET_FAILED, "tool is invalid, can not be None")

        # Define validation function for non-callable tools
        def validate_tool(tool_obj):
            # Store tool info
            if hasattr(tool_obj, "get_tool_info"):
                self._tool_infos[tool_id] = tool_obj.get_tool_info()
            else:
                self._tool_infos[tool_id] = ToolInfo()
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
                self._tool_infos[name] = ToolInfo()
            return tool

        return self._get_resource(name, StatusCode.RUNTIME_TOOL_GET_FAILED, create_tool_from_provider)

    def get_tool(self, tool_id: str, runtime=None) -> Optional[Tool]:
        # Validate ID using base class method
        self._validate_id(tool_id, StatusCode.RUNTIME_TOOL_GET_FAILED, "tool")

        try:
            tool = self.find_tool_by_name(tool_id)
            return decorate_tool_with_trace(tool, runtime)
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

    def get_tool_infos(self, tool_ids: List[str] = None, *, tool_server_name: str = None) -> Optional[List[Union[ToolInfo, McpToolInfo]]]:
        try:
            if tool_server_name:
                server_tools = self._server_tool_infos.get(tool_server_name)
                return list(server_tools) if server_tools is not None else None

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

    async def add_tool_servers(self, server_config: Union[ToolServerConfig, List[ToolServerConfig]]) -> List[bool]:
        """
        注册 MCP 服务器（同步连接、阻塞至完成）。
        返回：与传入顺序一一对应的注册结果列表，True=成功。
        """
        configs = [server_config] if isinstance(server_config, ToolServerConfig) else (server_config or [])
        if not configs:
            return []

        results: List[bool] = []
        for cfg in configs:
            try:
                await self._connect_and_register_server(cfg)
                results.append(True)
            except Exception as e:
                logger.exception(f"Register MCP server {cfg.server_name} failed: {e}")
                results.append(False)
        return results

    async def _connect_and_register_server(self, config: ToolServerConfig):
        try:
            client = self._create_client(config)
            connected = await client.connect()
            if not connected:
                logger.error(f"Failed to connect to MCP server: {config.server_name}")
                return

            self._mcp_clients[config.server_name] = client
            self._server_configs[config.server_name] = config

            tools = await client.list_tools()
            self._server_tool_infos[config.server_name] = tools

            for tool_info in tools:
                tool_id = tool_info.name
                mcp_tool = MCPTool(
                    mcp_client=client,
                    tool_name=tool_info.name,
                    server_name=config.server_name,
                )
                # 注册到 ToolMgr
                self.add_tool(tool_id, mcp_tool)
                logger.info(f"Registered MCP tool: {tool_id}")

        except Exception as e:
            logger.exception(f"Error registering MCP server {config.server_name}: {e}")

    def _create_client(self, config: ToolServerConfig) -> McpToolClient:
        if config.client_type == "sse":
            return SseClient(config.params, config.server_name)
        elif config.client_type == "stdio":
            return StdioClient(config.params, config.server_name)
        elif config.client_type == "playwright":
            return PlaywrightClient(config.params, config.server_name)
        else:
            raise ValueError(f"Unsupported MCP client type: {config.client_type}")

    async def remove_tool_server(self, tool_server_name: str):
        """移除 MCP 服务器"""
        if tool_server_name not in self._mcp_clients:
            logger.warning(f"MCP server '{tool_server_name}' not found.")
            return

        # 移除该服务器下的所有工具
        tools = self._server_tool_infos.pop(tool_server_name, [])
        for tool_info in tools:
            tool_id = f"{tool_server_name}.{tool_info.name}"
            self.remove_tool(tool_id)

        # 断开客户端连接 - 直接调用，不创建新任务
        client = self._mcp_clients.pop(tool_server_name)
        await client.disconnect()  # 直接 await，确保在同一个任务中
        self._server_configs.pop(tool_server_name, None)
        logger.info(f"Removed MCP server: {tool_server_name}")