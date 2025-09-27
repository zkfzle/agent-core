"""ReAct Controller 工具方法"""

import copy
from typing import List, Dict, Any, Optional

from jiuwen.agent.common.enum import SubTaskType
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.task.sub_task import SubTask
from jiuwen.core.agent.controller.base import ControllerOutput, ControllerInput
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.utils.format.format_utils import FormatUtils
from jiuwen.core.utils.llm.messages import BaseMessage, ToolInfo, ToolCall, AIMessage, HumanMessage, ToolMessage
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.common.logging import logger
from pydantic import Field, ConfigDict


class ReActControllerInput(ControllerInput):
    model_config = ConfigDict(extra='allow')


class ReActControllerOutput(ControllerOutput):
    """ReAct Controller输出类"""
    should_continue: bool = Field(default=False)
    llm_output: Optional[AIMessage] = Field(default=None)
    sub_tasks: List[SubTask] = Field(default_factory=list)


class ReActControllerUtils:
    """ReAct Controller 通用工具方法类"""

    @staticmethod
    def get_tools_info(config: AgentConfig) -> List[ToolInfo]:
        """获取可用工具信息"""
        tool_info_list = []
        tool_info_list.extend(FormatUtils.format_workflows_metadata(config.workflows))
        tool_info_list.extend(FormatUtils.format_plugins_metadata(config.plugins))
        return tool_info_list

    @staticmethod
    def format_llm_inputs(
            inputs,
            chat_history: List[BaseMessage],
            config: AgentConfig
    ) -> List[BaseMessage]:
        """格式化LLM输入"""
        if isinstance(inputs.query, InteractiveInput):
            user_fields = copy.deepcopy(inputs.model_dump())
            user_fields.pop("query")
        else:
            user_fields = inputs.model_dump()

        system_prompt = (Template(
            name=config.prompt_template_name,
            content=config.prompt_template
        ).format(user_fields).to_messages())

        return FormatUtils.create_llm_inputs(system_prompt, chat_history)

    @staticmethod
    def parse_llm_output(response: BaseMessage, config: AgentConfig) -> "ReActControllerOutput":
        """解析LLM输出并生成行动计划"""
        sub_tasks = ReActControllerUtils.create_sub_tasks_from_tool_calls(
            response.tool_calls, config
        )
        should_continue = len(sub_tasks) > 0
        return ReActControllerOutput(
            should_continue=should_continue,
            llm_output=response,
            sub_tasks=sub_tasks
        )

    @staticmethod
    def create_sub_tasks_from_tool_calls(
            tool_calls: List[ToolCall],
            config: AgentConfig
    ) -> List[SubTask]:
        """从工具调用创建SubTask"""
        if not tool_calls:
            return []

        result = []
        for tool_call in tool_calls:
            sub_task_type = ReActControllerUtils.determine_sub_task_type(
                tool_call.function.name, config
            )
            result.append(SubTask(
                id=tool_call.id,
                func_name=tool_call.function.name,
                func_args=FormatUtils.json_loads(tool_call.function.arguments),
                sub_task_type=sub_task_type
            ))
        return result

    @staticmethod
    def determine_sub_task_type(tool_name: str, config: AgentConfig) -> SubTaskType:
        """确定SubTask类型"""
        # 检查workflow
        for workflow in config.workflows:
            if tool_name == workflow.name:
                return SubTaskType.WORKFLOW

        # 检查plugin
        for plugin in config.plugins:
            if tool_name == plugin.name:
                return SubTaskType.PLUGIN

        raise JiuWenBaseException(5000, f"未找到工具调用类型: {tool_name}")

    @staticmethod
    def is_interaction_result(exec_result: Any) -> bool:
        """检查是否为交互中断结果"""
        return (isinstance(exec_result, dict) and
                exec_result.get("error") and
                isinstance(exec_result.get("value"), list))

    @staticmethod
    def create_interrupt_result(e, tool_name: str) -> Dict[str, Any]:
        """创建中断结果"""
        return {
            "error": True,
            "value": e.message,
            "tool_name": tool_name
        }

    @staticmethod
    def validate_execution_inputs(exec_result: Any, sub_task_result: Any) -> bool:
        """验证执行结果"""
        # 可以在这里添加更多的验证逻辑
        return exec_result is not None

    @staticmethod
    def should_add_user_message(query: str, context_engine: ContextEngine, runtime: Runtime) -> bool:
        """判断是否需要添加用户消息"""
        agent_context = context_engine.get_agent_context(runtime.session_id())
        last_message = agent_context.get_latest_message()

        if not last_message:
            return True

        if last_message.role == 'tool':
            logger.info("Skipping user message - post-tool-call request")
            return False

        if last_message.role == 'user' and last_message.content == query:
            logger.info("Skipping duplicate user message")
            return False

        return True

    @staticmethod
    def add_user_message(query: Any, context_engine: ContextEngine, runtime: Runtime):
        """添加用户消息到上下文"""
        if ReActControllerUtils.should_add_user_message(query, context_engine, runtime):
            agent_context = context_engine.get_agent_context(runtime.session_id())
            user_message = HumanMessage(content=query)
            agent_context.add_message(user_message)
            logger.info(f"Added user message: {query}")

    @staticmethod
    def add_ai_message(ai_message: AIMessage, context_engine: ContextEngine, runtime: Runtime):
        """添加AI消息到上下文"""
        if ai_message:
            agent_context = context_engine.get_agent_context(runtime.session_id())
            agent_context.add_message(ai_message)

    @staticmethod
    def add_tool_results(completed_tasks: List[SubTask], context_engine: ContextEngine, runtime: Runtime):
        """添加工具执行结果到上下文"""
        if not completed_tasks:
            logger.warning("No completed sub tasks to add to chat history")
            return

        agent_context = context_engine.get_agent_context(runtime.session_id())
        logger.info(f"Adding {len(completed_tasks)} tool results to chat history")

        for sub_task in completed_tasks:
            if sub_task.result:
                tool_message = ToolMessage(content=sub_task.result, tool_call_id=sub_task.id)
                agent_context.add_message(tool_message)
                logger.info(f"Added tool result: {sub_task.func_name}")
            else:
                logger.warning(f"Sub task {sub_task.func_name} has no result")

    @staticmethod
    def get_chat_history(context_engine: ContextEngine, runtime: Runtime, config: AgentConfig) -> List[BaseMessage]:
        """获取最新的对话历史"""
        agent_context = context_engine.get_agent_context(runtime.session_id())
        chat_history = agent_context.get_messages()
        max_rounds = config.constrain.reserved_max_chat_rounds
        return chat_history[-2 * max_rounds:]
