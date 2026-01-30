import asyncio
import re
from typing import List, Dict, Any

from openjiuwen.core.runner.runner import Runner, resource_mgr
from openjiuwen.core.utils.llm.messages import AIMessage, HumanMessage

from src.common.react_utils import ActionSpace, Action, ActionResult
from src.tools import ToolResult, ClickResult, SearchResult, get_tool_categories
from src.utils import AsyncCodeExecutor


class BaseSpace(ActionSpace):
    def __init__(self, working_dir: str, tools: List[str] = None, agents: List[str] = None):
        self.working_dir = working_dir
        self.tools = tools or []
        self.agents = agents or []

        self.actions = {
            "final": self._handle_final_action,
            "default": self._handle_default_action,
        }

        self.cache = []

    @staticmethod
    async def _handle_final_action(action: Action) -> ActionResult:
        return ActionResult(
            is_finished=True,
            message=HumanMessage(content=action.content),
        )

    @staticmethod
    async def _handle_default_action(action: Action) -> ActionResult:
        return ActionResult(
            is_finished=False,
            message=HumanMessage(
                content=f"Unknown action_type '{action.action}'. Please respond using the required XML tags."
            ),
        )

    def get_api_descriptions(self) -> str:
        desc = 'The usage of tool calling: `tool_result = call_tool(tool_name=\'tool_name\', **kwargs)`. (you can use custom variable names for the tool result)\n\n'
        desc += 'Below are the tools and their descriptions:\n\n'
        if self.agents is not None:
            # Todo 适配新接口
            description = "Tool: Deep Search\n"
            "Description: run comprehensive web searches (news, filings, research, etc.) "
            "to gather evidence for a given task.\n"
            "Parameters: query:str (describe exactly what information is needed; "
            "avoid loose keyword lists).\n"
            desc += f"- Tool: {"deepsearch_agent"}\nDescription: {description}\n\n"
        for tool in self.tools:
            tool_instance = resource_mgr.tool().get_tool(tool)
            parameters = [{"name": p.name, "description": p.description, "type": p.type, "required": p.required} for
                          p in tool_instance.params]
            desc += f"- Tool: {tool}\nDescription: {tool_instance.description}\nParameters: {parameters}\n\nOutput: \n\n"
        desc += 'The result of each tool is a variable, please use `print` to print the result.'
        return desc


    def parse_llm_response(self, message: AIMessage) -> List[Action]:
        response = message.content
        response = response.replace("<thinking>", "\n").replace("</thinking>", "\n")
        response = response.replace("<think>", "\n").replace("</think>", "\n")
        pattern = re.compile(r"<([\w_]+)>(.*?)</\1>", re.DOTALL)
        matches = list(pattern.finditer(response))

        if not matches:
            return []
        match = matches[-1]

        tag_name = match.group(1)
        if tag_name == 'execute':
            tag_name = 'code'
        if tag_name == 'final_result':
            tag_name = 'final'
        content_string = match.group(2).strip()  # Remove surrounding whitespace

        return [
            Action(
                action=tag_name,
                content=content_string
            )
        ]

    async def execute(self, action: Action) -> ActionResult:
        return await self.actions[action.action](action)


class CodeSpace(BaseSpace):
    def __init__(self, working_dir: str, tools: List[str] = None, agents: List[str] = None):
        super().__init__(working_dir, tools, agents)
        self.actions["code"] = self._handle_code_action
        self.code_executor = AsyncCodeExecutor(working_dir)

    async def _handle_code_action(self, action: Action) -> ActionResult:

        def format_execution_result(result: Dict[str, Any]) -> str:
            feedback = []

            if result["error"] is False:
                feedback.append("Code execution: success\n")

                if result["stdout"]:
                    feedback.append(f"Console output:\n{result['stdout']}\n\n")

                if result.get("variables"):
                    feedback.append("New variables:")
                    for var_name, var_info in result["variables"].items():
                        feedback.append(f"  - {var_name}: {var_info}")
                if result.get("additional_notes"):
                    feedback.append(f"Additional notes: {result['additional_notes']}\n")
            else:
                feedback.append("Code execution: failed\n")
                if result["stderr"]:
                    feedback.append(f"Error message: {result['stderr']}\n")
                if result["stdout"]:
                    feedback.append(f"Partial output: {result['stdout']}\n")
            return "\n".join(feedback)

        code_result = await self.code_executor.execute(code=action.content)
        return ActionResult(
            message=HumanMessage(content=format_execution_result(code_result)),
        )