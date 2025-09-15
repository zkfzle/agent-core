"""格式化工具类模块"""
import ast
import json
from typing import List

from jiuwen.core.utils.llm.messages import BaseMessage, ToolInfo, Function, Parameters
from jiuwen.agent.common.schema import WorkflowSchema, PluginSchema


class FormatUtils:
    """输入输出格式化工具类"""

    @classmethod
    def format_input_parameters(cls, inputs: dict) -> Parameters:
        """格式化输入参数为Parameters对象"""
        properties, required_parameters = dict(), list()
        if inputs.get("properties"):
            for key, value in inputs.get("properties").items():
                if value.get("required", False):
                    required_parameters.append(key)
                param_type = value.get("type", "").lower()
                if param_type in ["array", "object"]:
                    nested_result = dict()
                    cls._recursive_format_nested_params(param_type, value.get("properties", dict()), nested_result)
                    properties[key] = nested_result
                else:
                    properties[key] = dict(description=value.get("description", ""), type=param_type)
        parameters = Parameters(properties=properties, required=required_parameters)
        return parameters

    @classmethod
    def _recursive_format_nested_params(cls, param_type, properties, output):
        """递归格式化嵌套参数"""
        pass

    @staticmethod
    def format_workflows_metadata(workflows_metadata: List[WorkflowSchema]) -> List[ToolInfo]:
        """格式化工作流元数据为工具信息"""
        result = []
        for workflow in workflows_metadata:
            parameters = FormatUtils.format_input_parameters(workflow.inputs)
            function = Function(name=workflow.name, description=workflow.description, parameters=parameters)
            result.append(ToolInfo(function=function))
        return result

    @staticmethod
    def format_plugins_metadata(plugins_metadata: List[PluginSchema]) -> List[ToolInfo]:
        """格式化插件元数据为工具信息"""
        result = []
        for plugin in plugins_metadata:
            parameters = FormatUtils.format_input_parameters(plugin.inputs)
            function = Function(name=plugin.name, description=plugin.description, parameters=parameters)
            result.append(ToolInfo(function=function))
        return result

    @staticmethod
    def create_llm_inputs(system_prompt: List[BaseMessage], chat_history: List[BaseMessage]) -> List[BaseMessage]:
        """创建LLM输入消息列表

        Args:
            system_prompt: 系统提示消息列表
            chat_history: 对话历史消息列表（已包含当前用户输入）

        Returns:
            完整的LLM输入消息列表
        """
        from jiuwen.core.utils.llm.messages import HumanMessage

        # 创建新的消息列表，避免修改原始chat_history
        result_messages = []

        # 添加系统提示（如果chat_history中没有system消息）
        if not chat_history or chat_history[0].role != "system":
            result_messages.extend(system_prompt)

        # 添加对话历史
        result_messages.extend(chat_history)

        return result_messages

    @staticmethod
    def json_loads(arguments: str) -> dict:
        """安全的JSON解析"""
        result = dict()
        try:
            result = json.loads(arguments, strict=False)
        except json.JSONDecodeError:
            try:
                result = ast.literal_eval(arguments)
            except (SyntaxError, AttributeError, ValueError):
                pass
        return result