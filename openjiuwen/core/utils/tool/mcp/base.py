from abc import ABC, abstractmethod
from typing import Any, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from openjiuwen.core.utils.tool.base import Tool
from openjiuwen.core.utils.tool.schema import Parameters, ToolInfo
from openjiuwen.core.utils.tool.constant import Input, Output
from openjiuwen.core.common.logging import logger
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode

from pydantic import BaseModel


class ToolServerConfig(BaseModel):
    server_name: str
    params: Any
    client_type: str


NO_TIMEOUT = -1


class McpToolInfo(BaseModel):
    name: str
    description: str = ''
    schema: dict


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
            # Create and cache ToolInfo
            self._tool_info = ToolInfo(
                name=self.tool_name,
                description=f"MCP tool from {self.server_name}",
                parameters=Parameters(
                    type="object",
                    properties={},
                    required=[]
                )
            )

        return self._tool_info

class McpToolClient(ABC):
    def __init__(self, server_path: str):
        self._server_path = server_path

    @abstractmethod
    async def connect(self, *, timeout: float = NO_TIMEOUT) -> bool:
        pass

    @abstractmethod
    async def disconnect(self, *, timeout: float = NO_TIMEOUT) -> bool:
        pass

    @abstractmethod
    async def list_tools(self, *, timeout: float = NO_TIMEOUT) -> List[McpToolInfo]:
        pass

    @abstractmethod
    async def call_tool(self, tool_name, arguments: dict, *, timeout: float = NO_TIMEOUT) -> Any:
        pass

    @abstractmethod
    async def get_tool_info(self, tool_name: str, *, timeout: float = NO_TIMEOUT) -> Optional[McpToolInfo]:
        pass

class SseClient(McpToolClient):
    """SSE (Server-Sent Events) transport based MCP client"""

    def __init__(self, server_path: str, name: str):
        super().__init__(server_path)
        self._name = name
        self._client = None
        self._session = None
        self._read = None
        self._write = None

    async def connect(self, *, timeout: float = NO_TIMEOUT) -> bool:
        try:
            actual_timeout = timeout if timeout != NO_TIMEOUT else 60.0
            self._client = sse_client(self._server_path, timeout=actual_timeout)
            self._read, self._write = await self._client.__aenter__()
            self._session = ClientSession(
                self._read, self._write, sampling_callback=None
            )
            await self._session.__aenter__()
            await self._session.initialize()
            logger.info(f"SSE client connected successfully to {self._server_path}")
            return True

        except Exception as e:
            logger.error(f"SSE connection failed to {self._server_path}: {e}")
            await self.disconnect()
            return False

    async def disconnect(self, *, timeout: float = NO_TIMEOUT) -> bool:
        """Close SSE connection"""
        try:
            if self._session:
                await self._session.__aexit__(None, None, None)
                self._session = None

            if self._client:
                await self._client.__aexit__(None, None, None)
                self._client = None
                self._read = None
                self._write = None

            logger.info("SSE client disconnected successfully")
            return True
        except Exception as e:
            logger.error(f"SSE disconnection failed: {e}")
            return False

    async def list_tools(self, *, timeout: float = NO_TIMEOUT) -> List[McpToolInfo]:
        """List available tools via SSE"""
        if not self._session:
            raise RuntimeError("Not connected to SSE server")

        try:
            tools_response = await self._session.list_tools()
            tools_list = [
                McpToolInfo(
                    name=f"{self._name}.{tool.name}",
                    description=getattr(tool, "description", ""),
                    schema=getattr(tool, "inputSchema", {})
                )
                for tool in tools_response.tools
            ]
            logger.info(f"Retrieved {len(tools_list)} tools from SSE server")
            return tools_list
        except Exception as e:
            logger.error(f"Failed to list tools via SSE: {e}")
            raise

    async def call_tool(self, tool_name: str, arguments: dict, *, timeout: float = NO_TIMEOUT) -> Any:
        """Call tool via SSE"""
        if not self._session:
            raise RuntimeError("Not connected to SSE server")

        try:
            logger.info(f"Calling tool '{tool_name}' via SSE with arguments: {arguments}")
            tool_result = await self._session.call_tool(tool_name, arguments=arguments)
            # Extract text content from tool result
            result_content = None
            if tool_result.content and len(tool_result.content) > 0:
                result_content = tool_result.content[-1].text
            logger.info(f"Tool '{tool_name}' call completed via SSE")
            return result_content
        except Exception as e:
            logger.error(f"Tool call failed via SSE: {e}")
            raise

    async def get_tool_info(self, tool_name: str, *, timeout: float = NO_TIMEOUT) -> Optional[McpToolInfo]:
        """Get specific tool info via SSE"""
        tools = await self.list_tools(timeout=timeout)
        for tool in tools:
            if tool.name == tool_name:
                logger.debug(f"Found tool info for '{tool_name}' via SSE")
                return tool
        logger.warning(f"Tool '{tool_name}' not found via SSE")
        return None

class StdioClient(McpToolClient):
    """Stdio transport based MCP client"""

    def __init__(self, server_path: str, name: str):
        super().__init__(server_path)
        self._name = name
        self._client = None
        self._session = None
        self._read = None
        self._write = None

    async def connect(self, *, timeout: float = NO_TIMEOUT) -> bool:
        """Establish Stdio connection to the tool server"""
        try:
            # server_path should be StdioServerParameters for stdio client
            if not isinstance(self._server_path, StdioServerParameters):
                raise ValueError("StdioClient requires StdioServerParameters as server_path")

            self._client = stdio_client(self._server_path)
            self._read, self._write = await self._client.__aenter__()
            self._session = ClientSession(self._read, self._write, sampling_callback=None)
            await self._session.__aenter__()
            await self._session.initialize()
            logger.info("Stdio client connected successfully")
            return True
        except Exception as e:
            logger.error(f"Stdio connection failed: {e}")
            await self.disconnect()
            return False

    async def disconnect(self, *, timeout: float = NO_TIMEOUT) -> bool:
        """Close Stdio connection"""
        try:
            if self._session:
                await self._session.__aexit__(None, None, None)
                self._session = None

            if self._client:
                await self._client.__aexit__(None, None, None)
                self._client = None
                self._read = None
                self._write = None

            logger.info("Stdio client disconnected successfully")
            return True
        except Exception as e:
            logger.error(f"Stdio disconnection failed: {e}")
            return False

    async def list_tools(self, *, timeout: float = NO_TIMEOUT) -> List[McpToolInfo]:
        """List available tools via Stdio"""
        if not self._session:
            raise RuntimeError("Not connected to Stdio server")

        try:
            tools_response = await self._session.list_tools()
            tools_list = [
                McpToolInfo(
                    name=f"{self._name}.{tool.name}",
                    description=getattr(tool, "description", ""),
                    schema=getattr(tool, "inputSchema", {})
                )
                for tool in tools_response.tools
            ]
            logger.info(f"Retrieved {len(tools_list)} tools from Stdio server")
            return tools_list
        except Exception as e:
            logger.error(f"Failed to list tools via Stdio: {e}")
            raise

    async def call_tool(self, tool_name: str, arguments: dict, *, timeout: float = NO_TIMEOUT) -> Any:
        """Call tool via Stdio"""
        if not self._session:
            raise RuntimeError("Not connected to Stdio server")

        try:
            logger.info(f"Calling tool '{tool_name}' via Stdio with arguments: {arguments}")
            tool_result = await self._session.call_tool(tool_name, arguments=arguments)
            # Extract text content from tool result
            result_content = None
            if tool_result.content and len(tool_result.content) > 0:
                result_content = tool_result.content[-1].text
            logger.info(f"Tool '{tool_name}' call completed via Stdio")
            return result_content
        except Exception as e:
            logger.error(f"Tool call failed via Stdio: {e}")
            raise

    async def get_tool_info(self, tool_name: str, *, timeout: float = NO_TIMEOUT) -> Optional[McpToolInfo]:
        """Get specific tool info via Stdio"""
        tools = await self.list_tools(timeout=timeout)
        for tool in tools:
            if tool.name == tool_name:
                logger.debug(f"Found tool info for '{tool_name}' via Stdio")
                return tool
        logger.warning(f"Tool '{tool_name}' not found via Stdio")
        return None

class PlaywrightClient(McpToolClient):
    """Playwright browser session based MCP client"""

    def __init__(self, server_path: str, name: str):
        super().__init__(server_path)
        self._name = name
        self._client = None
        self._session = None
        self._read = None
        self._write = None

    async def connect(self, *, timeout: float = NO_TIMEOUT) -> bool:
        """Establish connection to Playwright MCP server"""
        try:
            # Determine client type based on server_path type
            if isinstance(self._server_path, StdioServerParameters):
                self._client = stdio_client(self._server_path)
                logger.debug("Using Stdio transport for Playwright client")
            elif isinstance(self._server_path, str) and self._server_path.startswith(("http://", "https://")):
                self._client = sse_client(self._server_path)
                logger.debug("Using SSE transport for Playwright client")
            else:
                raise ValueError(f"Unsupported server_path type: {type(self._server_path)}")

            self._read, self._write = await self._client.__aenter__()
            self._session = ClientSession(self._read, self._write, sampling_callback=None)
            await self._session.__aenter__()
            await self._session.initialize()
            logger.info("Playwright client connected successfully")
            return True
        except Exception as e:
            logger.error(f"Playwright connection failed: {e}")
            await self.disconnect()
            return False

    async def disconnect(self, *, timeout: float = NO_TIMEOUT) -> bool:
        """Close Playwright connection"""
        try:
            if self._session:
                await self._session.__aexit__(None, None, None)
                self._session = None

            if self._client:
                await self._client.__aexit__(None, None, None)
                self._client = None
                self._read = None
                self._write = None

            logger.info("Playwright client disconnected successfully")
            return True
        except Exception as e:
            logger.error(f"Playwright disconnection failed: {e}")
            return False

    async def list_tools(self, *, timeout: float = NO_TIMEOUT) -> List[McpToolInfo]:
        """List available browser tools"""
        if not self._session:
            raise RuntimeError("Not connected to Playwright server")

        try:
            tools_response = await self._session.list_tools()
            tools_list = [
                McpToolInfo(
                    name=f"{self._name}.{tool.name}",
                    description=getattr(tool, "description", ""),
                    schema=getattr(tool, "inputSchema", {})
                )
                for tool in tools_response.tools
            ]
            logger.info(f"Retrieved {len(tools_list)} browser tools from Playwright server")
            return tools_list
        except Exception as e:
            logger.error(f"Failed to list browser tools: {e}")
            raise

    async def call_tool(self, tool_name: str, arguments: dict, *, timeout: float = NO_TIMEOUT) -> Any:
        """Call browser tool"""
        if not self._session:
            raise RuntimeError("Not connected to Playwright server")

        try:
            logger.info(f"Calling browser tool '{tool_name}' with arguments: {arguments}")
            tool_result = await self._session.call_tool(tool_name, arguments=arguments)
            # Extract text content from tool result
            result_content = None
            if tool_result.content and len(tool_result.content) > 0:
                result_content = tool_result.content[-1].text
            logger.info(f"Browser tool '{tool_name}' call completed")
            return result_content
        except Exception as e:
            logger.error(f"Browser tool call failed: {e}")
            raise

    async def get_tool_info(self, tool_name: str, *, timeout: float = NO_TIMEOUT) -> Optional[McpToolInfo]:
        """Get specific browser tool info"""
        tools = await self.list_tools(timeout=timeout)
        for tool in tools:
            if tool.name == tool_name:
                logger.debug(f"Found browser tool info for '{tool_name}'")
                return tool
        logger.warning(f"Browser tool '{tool_name}' not found")
        return None
