from pathlib import Path

from openjiuwen.core.foundation.tool import ToolCard, tool
from openjiuwen.core.sys_operation.base import SysOperation

view_file_tool_card = ToolCard(
    id="_internal_view_file",
    name="view_file",
    description="查看指定文件路径的文件内容",
    input_params={
        "type": "object",
        "properties": {
            "file_path": {
                "description": "文件路径",
                "type": "string",
            }
        },
        "required": ["file_path"],
    }
)


@tool(
    card=view_file_tool_card
)
def view_file(file_path: str, sandbox_id: str = None):
    text = SysOperation().read_file(Path(file_path), "text", sandbox_id)
    return text
