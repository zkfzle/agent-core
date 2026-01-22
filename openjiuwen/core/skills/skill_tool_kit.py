from openjiuwen.core.foundation.tool import ToolCard, LocalFunction
from openjiuwen.core.runner import Runner
from openjiuwen.core.single_agent.agent import BaseAgent


class SkillToolKit:
    def __init__(self, sys_operation_id):
        self._sys_operation_id = sys_operation_id

    def create_view_file_tool(self):
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

        def view_file(file_path):
            sys_operation = Runner().resource_mgr.get_sys_operation(self._sys_operation_id)
            res = sys_operation.code().read_file(file_path)
            return str(res)

        return LocalFunction(
            card=view_file_tool_card,
            func=view_file
        )

    def create_execute_python_code_tool(self):
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

        def execute_python_code(code_block):
            sys_operation = Runner().resource_mgr.get_sys_operation(self._sys_operation_id)
            res = sys_operation.code().execute_code(code_block)
            return str(res)

        return LocalFunction(
            card=execute_python_code_tool_card,
            func=execute_python_code
        )

    def create_execute_command_tool(self):
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

        def run_command(code_block):
            sys_operation = Runner().resource_mgr.get_sys_operation(self._sys_operation_id)
            res = sys_operation.code().execute_code(code_block)
            return str(res)

        return LocalFunction(
            card=run_command_tool_card,
            func=run_command
        )

    def add_skill_tools(self, agent: BaseAgent):
        execute_python_code_tool = self.create_execute_python_code_tool()
        execute_command_tool = self.create_execute_command_tool()
        view_file_tool = self.create_view_file_tool()
        Runner().resource_mgr.add_tool(execute_python_code_tool)
        Runner().resource_mgr.add_tool(execute_command_tool)
        Runner().resource_mgr.add_tool(view_file_tool)
        agent.ability_kit.add(execute_python_code_tool.card)
        agent.ability_kit.add(execute_command_tool.card)
        agent.ability_kit.add(view_file_tool.card)

