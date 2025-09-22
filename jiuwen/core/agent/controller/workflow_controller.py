from typing import List, Union, Iterator, Any, Dict, AsyncIterator

from pydantic import Field

from jiuwen.agent.common.enum import SubTaskType, WorkflowAgentStatus, WorkflowAgentEvent
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.controller.base import Controller, ControllerOutput, ControllerInput
from jiuwen.core.agent.handler.base import AgentHandler, AgentHandlerInputs
from jiuwen.core.agent.state_machine.workflow_agent_state_machine import WorkflowAgentStateMachine
from jiuwen.core.agent.task.sub_task import SubTask
from jiuwen.core.agent.task.task_context import AgentRuntime
from jiuwen.core.common.logging import logger
from jiuwen.core.context.controller_context.controller_context_manager import ControllerContextMgr
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.runtime.runtime import WorkflowRuntime
from jiuwen.core.utils.llm.messages_chunk import BaseMessageChunk
from jiuwen.core.workflow.base import Workflow


class Message:
    ...


class WorkflowControllerOutput(ControllerOutput):
    sub_tasks: List[SubTask] = Field(default_factory=list)
    messages: Any = Field(default_factory=list)


class WorkflowControllerInput(ControllerInput):
    workflow_inputs: dict = Field(default_factory=dict)


class WorkflowController(Controller):
    def __init__(self, config: AgentConfig, context_mgr: ControllerContextMgr,
                 context_engine: ContextEngine, runtime: WorkflowRuntime):
        super().__init__(config, context_mgr)
        self._context_engine = context_engine
        self._runtime = runtime
        self._state_machine = WorkflowAgentStateMachine(runtime)
        self._setup_state_handlers()

    def _setup_state_handlers(self):
        # 注册同步状态处理器
        self._state_machine.register_state_handler(WorkflowAgentStatus.INITIALIZED, self._handle_initialized_state)
        self._state_machine.register_state_handler(WorkflowAgentStatus.COMPLETED, self._handle_completed_state)
        self._state_machine.register_state_handler(WorkflowAgentStatus.INTERRUPTED, self._handle_interrupted_state)

        # 注册流式状态处理器
        self._state_machine.register_state_handler(WorkflowAgentStatus.INITIALIZED, self._stream_handle_initialized_state,
                                                   is_stream=True)
        self._state_machine.register_state_handler(WorkflowAgentStatus.COMPLETED, self._stream_handle_completed_state,
                                                   is_stream=True)
        self._state_machine.register_state_handler(WorkflowAgentStatus.INTERRUPTED, self._stream_handle_interrupted_state,
                                                   is_stream=True)

    async def _handle_initialized_state(self, inputs:Dict):
        current_inputs = inputs
        while True:
            controller_output: WorkflowControllerOutput = self.invoke(current_inputs, None)
            results = {}

            from jiuwen.core.agent.task.task_context import AgentRuntime
            temp_context = AgentRuntime(trace_id=self._runtime.session_id())
            temp_context.set_controller_context_manager(self._context_mgr)

            if controller_output.sub_tasks:
                self._state_machine.update_state_data({
                    "sub_tasks": controller_output.sub_tasks
                })
                for sub_task in controller_output.sub_tasks:
                    inputs = AgentHandlerInputs(context=temp_context, name=sub_task.func_name,
                                                arguments=sub_task.func_args)
                    result = await self._agent_handler.invoke(sub_task.sub_task_type, inputs)
                    results[sub_task.func_name] = result
            if not self.should_continue(controller_output):
                output = self.handle_workflow_results(results)
                self._state_machine.set_current_status(WorkflowAgentStatus.COMPLETED)
                self._state_machine.set_current_event(WorkflowAgentEvent.USER_INVOKE)
                result = await self._state_machine.handle_state(self._state_machine._current_status, output)
                logger.info(f"State handler result: {result}")

                self._state_machine.update_state_data({
                    "sub_tasks": controller_output.sub_tasks,
                    "final_result": result
                })
                return result
            current_inputs = results

    async def _handle_completed_state(self, output):
        self._state_machine.update_state_data({"final_result": output})
        logger.info(f"update final result {output}")
        return output

    async def _handle_interrupted_state(self):
        pass

    async def _stream_handle_initialized_state(self, inputs:Dict):
        current_inputs = inputs
        while True:
            controller_output: WorkflowControllerOutput = self.invoke(current_inputs, None)
            results = {}
            from jiuwen.core.agent.task.task_context import AgentRuntime
            temp_context = AgentRuntime(trace_id=self._runtime.session_id())
            temp_context.set_controller_context_manager(self._context_mgr)

            if controller_output.sub_tasks:
                for sub_task in controller_output.sub_tasks:
                    if sub_task.sub_task_type != SubTaskType.WORKFLOW:
                        logger.error("Only support workflow sub task in workflow agent")
                        continue
                    inputs = AgentHandlerInputs(context=temp_context, name=sub_task.func_name,
                                                arguments=sub_task.func_args)
                    workflow = self._find_workflow(inputs)
                    async for result in workflow.stream(inputs.arguments, inputs.context.create_workflow_runtime()):
                        if hasattr(result, 'type') and result.type == 'workflow_final':
                            final_result = result
                        else:
                            yield result

            if not self.should_continue(controller_output) and final_result is not None:
                self._state_machine.set_current_status(WorkflowAgentStatus.COMPLETED)
                self._state_machine.set_current_event(WorkflowAgentEvent.USER_INVOKE)
                async for result in self._state_machine.handle_stream_state(self._state_machine.get_current_status(),
                                                                        final_result):
                    yield result
                break
            else:
                current_inputs = results

    async def _stream_handle_completed_state(self, output):
        self._state_machine.update_state_data({"final_result": output})
        yield output

    async def _stream_handle_interrupted_state(self):
        pass

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
        context_manager = context.controller_context_manager()
        workflow_manager = context_manager.workflow_mgr
        workflow_metadata = self._agent_handler.search_workflow_metadata_by_workflow_name(workflow_name)
        workflow = workflow_manager.find_workflow_by_id_and_version(workflow_metadata.id,
                                                                    workflow_metadata.version)
        return workflow

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

        return WorkflowControllerOutput(is_task=True, sub_tasks=sub_tasks)

    async def stream(self,
                     inputs: WorkflowControllerInput,
                     context: AgentRuntime
                     ) -> AsyncIterator[Union[BaseMessageChunk, WorkflowControllerOutput]]:
        pass

    def should_continue(self, output: WorkflowControllerOutput) -> bool:
        """
        当且仅当 output 是 Task 时继续下一轮
        """
        return not output.is_task

    def set_agent_handler(self, agent_handler: AgentHandler):
        """设置Agent处理器"""
        self._agent_handler = agent_handler

    def set_is_single_workflow(self, is_single_workflow: bool):
        self._is_single_workflow = is_single_workflow

    def handle_workflow_results(self, results):
        if self._config.is_single_workflow:
            return results[self._config.workflows[0].name]
        raise Exception("Multi-workflow not implemented yet")

    async def execute(self, inputs: Dict) -> Dict:
        logger.info(f"Starting Workflow Controller execution with inputs: {inputs}")
        self._state_machine.set_current_event(WorkflowAgentEvent.USER_INVOKE)

        while not self._state_machine.is_completed():
            current_status = self._state_machine.get_current_status()
            logger.info(f"Current status: {current_status}")

            if not self._state_machine.can_handle_status(current_status):
                logger.error(f"Cannot handle status: {current_status}")
                break

            result = await self._state_machine.handle_state(current_status, inputs)
            logger.info(f"State handler result: {result}")

            # 如果返回了最终结果，直接返回
            if isinstance(result, dict) and result.get("result_type") in ["answer", "question"]:
                logger.info(f"Returning final result: {result}")
                return result

            # 返回最终结果
        final_result = self._state_machine.get_final_result()
        logger.info(f"Final execution result: {final_result}")
        return {"output": final_result, "result_type": "answer"}

    async def stream_execute(self, inputs: Dict):
        """流式执行ReAct迭代流程"""
        self._state_machine.set_current_event(WorkflowAgentEvent.USER_INVOKE)

        while not self._state_machine.is_completed():
            current_status = self._state_machine.get_current_status()

            if not self._state_machine.can_handle_status(current_status, is_stream=True):
                logger.error(f"Cannot handle stream status: {current_status}")
                break

            async for result in self._state_machine.handle_stream_state(current_status, inputs):
                yield result

                # 如果是最终结果，结束流式处理
                if isinstance(result, dict) and result.get("result_type") in ["answer", "question"]:
                    return