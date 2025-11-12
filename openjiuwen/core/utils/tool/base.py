#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from abc import abstractmethod
from typing import Any, Optional

from openjiuwen.core.utils.llm.messages import ToolInfo, Function, Parameters
from openjiuwen.core.utils.tool.constant import Input, Output


class Tool:
    """tool class that defined the data types and content for LLM modules"""
    def __init__(self):
        pass

    @abstractmethod
    def invoke(self, inputs: Input, **kwargs) -> Output:
        """invoke the tool"""
        pass

    @abstractmethod
    async def ainvoke(self, inputs: Input, **kwargs) -> Output:
        """async invoke the tool"""
        pass

    @abstractmethod
    def get_tool_info(self) -> ToolInfo:
        """get tool info"""
        pass


class MCPTool(Tool):
    """MCP Tool class that wraps MCP server tools for LLM modules"""

    def __init__(self,
                 mcp_client: Any,  # McpToolClient or its subclasses
                 tool_name: str,
                 server_name: str = "mcp_server"):
        """
        Initialize MCP Tool

        Args:
            mcp_client: Instance of McpToolClient or its subclasses
            tool_name: Name of the MCP tool
            server_name: Name of the MCP server (for logging and identification)
        """
        super().__init__()
        self.mcp_client = mcp_client
        self.tool_name = tool_name
        self.server_name = server_name
        self._tool_info: Optional[ToolInfo] = None

    async def ainvoke(self, inputs: Input, **kwargs) -> Output:
        """Async invoke of the MCP tool"""
        try:
            # Prepare arguments for MCP tool call
            arguments = inputs if isinstance(inputs, dict) else {}

            result = await self.mcp_client.call_tool(
                tool_name=self.tool_name,
                arguments=arguments
            )
            return {"result": result}

        except Exception as e:
            return {"error": f"Tool invocation failed: {str(e)}"}

    def get_tool_info(self) -> ToolInfo:
        """Get tool information"""
        # If we haven't cached the tool info, create it
        if self._tool_info is None:
            # Create a Function object with the tool information
            function = Function(
                name=self.tool_name,
                description=f"MCP tool from {self.server_name}",
                parameters=Parameters(
                    type="object",
                    properties={},
                    required=[]
                )
            )

            # Create and cache ToolInfo
            self._tool_info = ToolInfo(
                type="function",
                function=function
            )

        return self._tool_info
