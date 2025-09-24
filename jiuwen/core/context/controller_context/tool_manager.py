from typing import List, Optional, Tuple

from jiuwen.core.utils.llm.messages import ToolInfo
from jiuwen.core.utils.tool.base import Tool
from jiuwen.core.utils.tool.service_api.restful_api import RestfulApi
from jiuwen.core.context.controller_context.thread_safe_dict import ThreadSafeDict


class ToolMgr:
    def __init__(self) -> None:
        self._tools: ThreadSafeDict[str, Tool] = ThreadSafeDict()
        self._tool_infos: ThreadSafeDict[str, ToolInfo] = ThreadSafeDict()

    def add_tool(self, tool_id: str, tool: Tool) -> None:
        self._tools[tool_id] = tool
        self._tool_infos[tool_id] = tool.get_tool_info()

    def add_tools(self, tools: List[Tuple[str, Tool]]):
        if not tools:
            return
        for id, tool in tools:
            if isinstance(tool, RestfulApi):
                self._tools.update({id: tool})
                self._tool_infos.update({id: tool.get_tool_info()})

    def find_tool_by_name(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_tool(self, tool_id: str) -> Tool:
        return self._tools.get(tool_id)

    def remove_tool(self, tool_id: str):
        self._tool_infos.pop(tool_id, None)
        return self._tools.pop(tool_id, None) is not None

    def get_tool_infos(self, tool_id: List[str]):
        if not tool_id:
            return []
        return [self._tool_infos.get(id) for id in tool_id]
