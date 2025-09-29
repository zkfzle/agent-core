from typing import List, Union, Any, Dict, AsyncIterator
import json

from jiuwen.agent.common.enum import SubTaskType
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.controller.base import Controller
from jiuwen.core.agent.controller.utils import WorkflowControllerOutput, WorkflowControllerInput
from jiuwen.core.agent.handler.base import AgentHandler, AgentHandlerInputs
from jiuwen.core.agent.task.sub_task import SubTask
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.logging import logger
from jiuwen.core.context.controller_context.workflow_manager import generate_workflow_key
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.runtime.interaction.base import AgentInterrupt
from jiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.utils.llm.messages import HumanMessage, AIMessage
from jiuwen.core.utils.llm.messages_chunk import BaseMessageChunk
from jiuwen.core.workflow.base import Workflow


class WorkflowState:
    """Workflow状态管理类"""

    def __init__(self, runtime: Runtime):
        self._runtime = runtime

    def is_interrupted(self) -> bool:
        track_state = self._runtime.get_state("workflow_state")
        return (track_state and
                track_state.get("status") == "interrupted" and
                track_state.get("sub_tasks") is not None)

    def get_interrupted_sub_tasks(self) -> List[SubTask]:
        state_data = self._runtime.get_state("workflow_state") or {}
        return state_data.get("sub_tasks", [])

    def save_interrupt_state(self, sub_tasks: List[SubTask]):
        self._runtime.update_state({
            "workflow_state": {
                "status": "interrupted",
                "sub_tasks": sub_tasks
            }
        })

    def get_current_status(self) -> str:
        """获取当前状态"""
        track_state = self._runtime.get_state("workflow_state")
        if not track_state:
            return "normal"
        return track_state.get("status", "normal")

    def set_status(self, status: str, sub_tasks: List[SubTask] = None):
        """设置状态"""
        self._runtime.update_state({
            "workflow_state": {
                "status": status,
                "sub_tasks": sub_tasks or []
            }
        })


class WorkflowController(Controller):
    def __init__(self, config: AgentConfig, context_engine: ContextEngine, runtime: Runtime):
        super().__init__(config)
        self._context_engine = context_engine
        self._runtime = runtime
        self._state = WorkflowState(runtime)
        self._agent_handler = None

    def set_agent_handler(self, agent_handler: AgentHandler):
        """设置Agent处理器"""
        self._agent_handler = agent_handler

    @staticmethod
    def _filter_inputs(schema: dict, user_data: dict) -> dict:
        """
        根据 schema 过滤并校验用户输入
        :param schema:   workflow.inputs 的 schema，形如 {"query": {"type": "string", "required": True}}
        :param user_data: 用户实际传入的数据，形如 {"query": "你好", "foo": "bar"}
        :return: 仅保留 schema 中声明的字段
        :raises KeyError: 缺失必填字段时抛出
        """
        if not schema:
            return {}

        required_fields = {
            k for k, v in schema.items()
            if isinstance(v, dict) and v.get("required") is True
        }

        filtered = {}
        for k in schema:
            if k not in user_data:
                if k in required_fields:
                    raise KeyError(f"缺少必填参数: {k}")
                continue
            filtered[k] = user_data[k]

        return filtered

    def _find_workflow(self, inputs: AgentHandlerInputs) -> Workflow:
        context = inputs.context
        workflow_name = inputs.name
        workflow_metadata = self._agent_handler.search_workflow_metadata_by_workflow_name(workflow_name)
        workflow_id = generate_workflow_key(workflow_metadata.id, workflow_metadata.version)
        workflow = context.get_workflow(workflow_id)
        return workflow

    def _add_msg_to_chat_histroy(self, message: Union[HumanMessage, AIMessage]):
        workflow_context = self._context_engine.get_workflow_context(workflow_id=self._config.workflows[0].id,
                                                                     session_id=self._runtime.session_id())
        workflow_context.add_message(message)

    def invoke(
            self, inputs: Dict, context
    ) -> WorkflowControllerOutput:
        if len(self._config.workflows) > 1:
            raise NotImplementedError("Multi-workflow not implemented yet")

        workflow = self._config.workflows[0]

        filtered_inputs = self._filter_inputs(
            schema=workflow.inputs or {},
            user_data=inputs
        )

        sub_tasks = [
            SubTask(
                sub_task_type=SubTaskType.WORKFLOW,
                func_name=workflow.name,
                func_id=f"{workflow.id}_{workflow.version}",
                func_args=filtered_inputs,
            )
        ]

        user_message = HumanMessage(content=inputs.get("query"))
        self._add_msg_to_chat_histroy(user_message)
        logger.info(f"Added user message to chat history: {inputs.get('query')}")

        return WorkflowControllerOutput(is_task=True, sub_tasks=sub_tasks)

    async def stream(self,
                     inputs: WorkflowControllerInput,
                     context: Runtime
                     ) -> AsyncIterator[Union[BaseMessageChunk, WorkflowControllerOutput]]:
        pass

    def should_continue(self, output: WorkflowControllerOutput) -> bool:
        """
        当且仅当 output 是 Task 时继续下一轮
        """
        return not output.is_task

    def handle_workflow_results(self, results):
        if self._config.is_single_workflow:
            return results[self._config.workflows[0].name]
        raise Exception("Multi-workflow not implemented yet")

    @staticmethod
    def _validate_inputs(inputs: Dict):
        """验证输入"""
        if isinstance(inputs.get("query"), InteractiveInput):
            raise JiuWenBaseException(5000, "Non-interrupt status data format error.")

    async def execute(self, inputs: Dict) -> Dict | list:
        """主执行流程 - 简化的workflow执行"""
        logger.info(f"Starting Workflow execution with inputs: {inputs}")

        # 输入验证（仅在非中断恢复时进行）
        if not self._state.is_interrupted():
            self._validate_inputs(inputs)

        # 检查是否需要处理中断恢复
        if self._state.is_interrupted():
            return await self._resume_task(inputs)

        # 执行workflow
        return await self._run_workflow(inputs)

    async def _resume_task(self, inputs: Dict) -> Dict | list:
        """恢复中断的任务"""
        if not isinstance(inputs.get("query"), InteractiveInput):
            raise JiuWenBaseException(5000, "Interrupt status data format error.")

        logger.info(f"Processing interrupt recovery: {inputs}")

        # 获取中断的任务
        sub_tasks = self._state.get_interrupted_sub_tasks()
        if not sub_tasks:
            self._state.set_status("normal")
            return await self._run_workflow(inputs)

        # 更新第一个任务的参数
        sub_tasks[0].func_args = inputs.get("query", "")

        # 执行恢复的任务
        result = await self._execute_workflow_task(sub_tasks[0])

        # 检查是否为交互中断结果
        if result and hasattr(result, 'state') and result.state.value == "INPUT_REQUIRED":
            # 是中断状态
            interrupt_data_list = []
            for output_scheme in result.result:
                await self._runtime.write_stream(output_scheme)
                interrupt_data_list.append(output_scheme)
            return interrupt_data_list
        else:
            # 恢复正常状态并返回结果
            self._state.set_status("normal")
            final_result = self.handle_workflow_results({sub_tasks[0].func_name: result})
            return {"output": final_result, "result_type": "answer"}

    async def _run_workflow(self, inputs: Dict) -> Dict | list:
        """执行workflow主流程"""
        # 生成sub_tasks
        controller_output: WorkflowControllerOutput = self.invoke(inputs, None)

        if not controller_output.sub_tasks:
            return {"output": "No tasks to execute", "result_type": "answer"}

        # 执行第一个sub_task
        sub_task = controller_output.sub_tasks[0]
        result = await self._execute_workflow_task(sub_task)

        # 检查是否为交互中断结果
        if result and hasattr(result, 'state') and result.state.value == "INPUT_REQUIRED":
            # 是中断状态，保存状态
            self._state.save_interrupt_state([sub_task])
            interrupt_data_list = []
            for output_scheme in result.result:
                await self._runtime.write_stream(output_scheme)
                interrupt_data_list.append(output_scheme)
            return interrupt_data_list
        else:
            # 正常完成
            final_result = self.handle_workflow_results({sub_task.func_name: result})
            return {"output": final_result, "result_type": "answer"}

    async def _execute_workflow_task(self, sub_task: SubTask) -> Any:
        """执行单个workflow任务"""
        try:
            inputs = AgentHandlerInputs(
                context=self._runtime,
                name=sub_task.func_name,
                arguments=sub_task.func_args
            )
            workflow = self._find_workflow(inputs)

            # 创建workflow runtime并执行
            workflow_runtime = inputs.context.create_workflow_runtime()
            result = await workflow.invoke(inputs.arguments, workflow_runtime)

            # 处理WorkflowOutput对象
            if hasattr(result, 'result') and hasattr(result, 'state'):
                # 对于WorkflowOutput，存储完整对象以便后续处理状态
                sub_task.result = result
                return result
            else:
                # 对于其他对象，尝试序列化，如果失败则直接存储
                try:
                    sub_task.result = json.dumps(result, ensure_ascii=False)
                except TypeError:
                    sub_task.result = result
                return result

        except AgentInterrupt as e:
            # 插件执行失败时，添加失败信息
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(f"Sub task {sub_task.func_name} failed: {error_msg}")

            error_result = {
                "error": True,
                "id": "0",
                "value": e.message,
                "message": error_msg,
                "tool_name": sub_task.func_name
            }
            sub_task.result = json.dumps(error_result, ensure_ascii=False)
            return error_result
