from openjiuwen.core.foundation.tool import ToolCard, tool
from openjiuwen.core.sys_operation.base import SysOperation

execute_python_code_tool_card = ToolCard(
    id="_internal_execute_python_code",
    name="execute_python_code",
    description="执行python代码",
    input_params={
        "type": "object",
        "properties": {
            "code_block": {
                "description": "要执行的python代码",
                "type": "string",
            }
        },
        "required": ["code_block"],
    }
)


@tool(
    card=execute_python_code_tool_card
)
def execute_python_code(code_block: str, sandbox_id: str = None):
    text = SysOperation().run_python_code(code_block, sandbox_id)
    return text
