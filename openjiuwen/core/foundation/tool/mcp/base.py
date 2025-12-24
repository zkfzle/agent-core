#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import Any

from pydantic import Field

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.foundation.tool.base import Tool
from openjiuwen.core.foundation.tool.constant import Input, Output
from openjiuwen.core.foundation.tool.schema import ToolInfo


class McpToolInfo(ToolInfo):
    input_schema: dict = Field(default_factory=dict)
    server_name: str = Field(default="")


class MCPTool(Tool):
    """MCP Tool class that wraps MCP server tools for LLM modules"""

    def __init__(self,
                 mcp_client: Any,  # McpToolClient or its subclasses
                 tool_info: McpToolInfo):
        """
        Initialize MCP Tool

        Args:
            mcp_client: Instance of McpToolClient or its subclasses
            tool_name: Name of the MCP tool
            server_name: Name of the MCP server (for logging and identification)
        """
        super().__init__()
        self.mcp_client = mcp_client
        self._tool_info = tool_info

    def invoke(self, inputs: Input, **kwargs) -> Output:
        """invoke of the MCP tool"""
        raise JiuWenBaseException(
            error_code=StatusCode.PLUGIN_UNEXPECTED_ERROR.code, message="mcp tool only support ainvoke"
        )

    async def ainvoke(self, inputs: Input, **kwargs) -> Output:
        """Async invoke of the MCP tool"""
        try:
            # Prepare arguments for MCP tool call
            arguments = inputs if isinstance(inputs, dict) else {}

            result = await self.mcp_client.call_tool(
                tool_name=self._tool_info.name,
                arguments=arguments
            )
            return {"result": result}

        except Exception as e:
            return {"error": f"Tool invocation failed: {str(e)}"}

    def get_tool_info(self) -> ToolInfo:
        """Get tool information"""
        return self._tool_info
