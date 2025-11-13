from abc import ABC, abstractmethod
from typing import Any, List, Optional

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
