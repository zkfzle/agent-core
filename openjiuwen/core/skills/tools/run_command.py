from openjiuwen.core.foundation.tool import ToolCard, tool
from openjiuwen.core.runner import Runner
from openjiuwen.core.sys_operation.base import SysOperation

run_command_tool_card = ToolCard(
    id="_internal_run_command",
    name="run_command",
    description="在linux终端执行bash命令",
    input_params={
        "type": "object",
        "properties": {
            "bash_command": {
                "description": "一条或多条bash命令",
                "type": "string",
            }
        },
        "required": ["bash_command"],
    }
)


@tool(
    card=run_command_tool_card
)
def run_command(command: str, sandbox_id: str = None):
    res = SysOperation().run_command(command, sandbox_id)
    return res
