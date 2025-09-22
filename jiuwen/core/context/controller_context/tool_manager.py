from typing import List, Optional, Tuple
from jiuwen.core.utils.tool.base import Tool
from jiuwen.core.utils.tool.service_api.restful_api import RestfulApi
from jiuwen.core.context.controller_context.thread_safe_dict import ThreadSafeDict


class ToolMgr:
    def __init__(self) -> None:
        self._tools: ThreadSafeDict[str, Tool] = ThreadSafeDict()

    def add_tool(self, tool_id: str, tool: Tool) -> None:
        self._tools[tool_id] = tool

    def add_tools(self, tools: List[Tuple[str, Tool]]):
        if not tools:
            return
        for id, tool in tools:
            if isinstance(tool, RestfulApi):
                self._tools.update({id: tool})

    def find_tool_by_name(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_tool(self, tool_id: str) -> Tool:
        return self._tools.get(tool_id)

    def remove_tool(self, tool_id: str):
        return self._tools.pop(tool_id, None) is not None