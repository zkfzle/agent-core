import json
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Union, Iterable, List, Any, Dict

from fastmcp.experimental.utilities.openapi.director import RequestDirector
from fastmcp.experimental.utilities.openapi import (
    HTTPRoute,
    extract_output_schema_from_responses,
    format_simple_description,
    parse_openapi_to_http_routes,
)
from fastmcp.experimental.server.openapi import OpenAPITool
from fastmcp.tools.tool import ToolResult
from jsonschema_path import SchemaPath
import httpx
from openjiuwen.core.common.logging import logger
from openjiuwen.core.utils.tool.mcp.base import NO_TIMEOUT, McpToolInfo, McpToolClient


class ToolManager:
    def __init__(self):
        self.tools: dict[str, OpenAPITool] = {}

    async def get_tool(self, tool_name: str) -> OpenAPITool | None:
        if tool_name not in self.tools.keys():
            return None
        tool = self.tools[tool_name]
        return tool

    async def get_tools(self) -> dict[str, OpenAPITool]:
        return self.tools

    async def call_tool(self, key: str, arguments: dict[str, any]) -> ToolResult:
        tool = await self.get_tool(key)
        if tool is None:
            return ToolResult(None, None)
        try:
            return await tool.run(arguments)
        except Exception as e:
            logger.exception(f"{e} when calling openapi_tool {key}")
            raise ValueError(f"{e} when calling openapi_tool {key}")

class OpenApiClient(McpToolClient):
    def __init__(self, server_path: Union[str,Iterable[str]], name: str):
        super().__init__(str(server_path))
        self._director = None
        self._spec = None
        self.openapi_spec = None
        self._name = name
        self._client = httpx.AsyncClient()
        self._tool_manager = ToolManager()
        self._used_names: defaultdict[str, int] = defaultdict(int)

        if isinstance(server_path, str):
            self._server_path = [server_path]
        else:
            self._server_path = server_path

    def _generate_tool_name(self, route: HTTPRoute) -> str:
        """Generate a tool name from the route and client_name."""
        if route.operation_id:
            name = route.operation_id.split("__")[0]
        else:
            name = route.summary or f"{route.method}_{route.path}"

        # Truncate to 64 characters maximum
        if len(name) > 64:
            name = name[:64]

        return f"{name}"

    def _get_unique_name(self, name: str) -> str:
        """
        Ensure the tool_name is unique

        Args:
            name: the original name

        Returns:
            str: A unique name
        """
        # record this tool_name
        self._used_names[name] += 1
        # if tool_name is used, add number to tool_name
        if self._used_names[name] == 1:
            return name
        else:
            new_name = f"{name}_{self._used_names[name]}"
            logger.debug(
                f"tool_ame collision: '{name}' already used,using '{new_name}' instead. "
            )

        return new_name

    def _create_openapi_tool(
            self,
            route: HTTPRoute,
            name: str,
            tags: set[str],
            timout: float,
    ):
        """create an OpenAPITool"""
        # Use pre-calculated schema from route
        combined_schema = route.flat_param_schema

        # Extract output schema from OpenAPI responses
        output_schema = extract_output_schema_from_responses(
            route.responses,
            route.response_schemas,
            route.openapi_version,
        )

        # Get a unique tool name
        tool_name = self._get_unique_name(name)

        base_description = (
                route.description
                or route.summary
                or f"Executes {route.method} {route.path}"
        )

        # Use simplified description formatter for tools
        enhanced_description = format_simple_description(
            base_description=base_description,
            parameters=route.parameters,
            request_body=route.request_body,
        )

        tool = OpenAPITool(
            client=self._client,
            route=route,
            director=self._director,
            name=tool_name,
            description=enhanced_description,
            parameters=combined_schema,
            output_schema=output_schema,
            tags=set(route.tags or []) | tags,
            timeout=timout,
        )

        # Register the tool by directly assigning to the tools dictionary
        self._tool_manager.tools[tool_name] = tool

    async def connect(self, *, timeout: float = NO_TIMEOUT) -> bool:
        for server_path in self._server_path:
            self.openapi_spec = load_conf(server_path)

            try:
                self._spec = SchemaPath.from_dict(self.openapi_spec)
                self._director = RequestDirector(self._spec)
            except Exception as e:
                logger.error(f"Invalid openapi spec: {e}")
                return False

            http_routes = parse_openapi_to_http_routes(self.openapi_spec)

            # start to convert openapi to mcp_tool
            for route in http_routes:
                tool_name = self._generate_tool_name(route)
                tool_tags = set(route.tags or [])
                self._create_openapi_tool(route, tool_name, tool_tags, timeout)

        return True

    async def disconnect(self, *, timeout: float = NO_TIMEOUT)->bool:
       await self._client.aclose()
       return True

    async def list_tools(self, *, timeout: float = NO_TIMEOUT) -> List[McpToolInfo]:
        tools = await self._tool_manager.get_tools()
        tools_info = []

        for tool_name, tool in tools.items():
            tools_info.append(McpToolInfo(
                name=tool_name,
                description=tool.description,
                input_schema=tool.parameters,
                )
            )
        return tools_info

    async def call_tool(self, tool_name, arguments: dict, *, timeout: float = NO_TIMEOUT) -> Any:
        try:
            tool_result = await self._tool_manager.call_tool(tool_name, arguments)
            return tool_result.to_mcp_result()
        except Exception as e:
            logger.error(f"Tool call failed via openapi: {e}")
            raise

    async def get_tool_info(self, tool_name: str, *, timeout: float = NO_TIMEOUT) -> Any:
        tool = await self._tool_manager.get_tool(tool_name)
        return McpToolInfo(
            name=tool_name,
            description=tool.description,
            input_schema=tool.parameters,
        )


def load_conf(file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON or YAML file and return its content as a dict.

    Parameters
    ----------
    file : str | pathlib.path
        Path to the file to be loaded.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    ValueError
        If the extension is invalid or the root element is not a dict.
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError / yaml.YAMLError
        If parsing fails.
    """
    path = Path(file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as fp:
        if suffix == ".json":
            data = json.load(fp)
        elif suffix in {".yaml", ".yml"}:
            # 如需安全加载，可用 yaml.safe_load
            data = yaml.safe_load(fp)
        else:
            raise ValueError(f"仅支持 .json / .yaml / .yml，当前扩展名：{suffix}")

    if not isinstance(data, dict):
        raise ValueError("配置文件根元素必须是 dict 类型")
    return data