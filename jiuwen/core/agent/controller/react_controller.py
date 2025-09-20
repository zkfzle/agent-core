"""Controller of ReActAgent"""
from typing import List, Dict, Any, Optional, AsyncIterator, Union

from pydantic import Field

from jiuwen.agent.common.enum import SubTaskType, ReActStatus, ReActEvent
from jiuwen.core.agent.controller.base import ControllerOutput, ControllerInput, Controller
from jiuwen.core.agent.handler.base import AgentHandler
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.task.sub_task import SubTask
from jiuwen.core.agent.task.task_context import AgentRuntime
from jiuwen.core.agent.state_machine.react_state_machine import ReActStateMachine
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.context.controller_context.controller_context_manager import ControllerContextMgr
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.runtime.runtime import WorkflowRuntime
from jiuwen.core.utils.format.format_utils import FormatUtils
from jiuwen.core.utils.llm.messages import BaseMessage, ToolInfo, HumanMessage, AIMessage, \
    ToolCall, UsageMetadata
from jiuwen.core.utils.llm.messages_chunk import BaseMessageChunk
from jiuwen.core.utils.llm.model_utils.model_factory import ModelFactory
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.core.graph.interrupt.interactive_input import InteractiveInput
from jiuwen.core.common.logging import logger


class ReActControllerOutput(ControllerOutput):
    should_continue: bool = Field(default=False)
    llm_output: Optional[AIMessage] = Field(default=None)
    sub_tasks: List[SubTask] = Field(default_factory=list)


class ReActControllerInput(ControllerInput):
    user_fields: Dict[str, Any] = Field(default_factory=dict, alias="userFields")


class ReActController(Controller):
    def __init__(self, config: AgentConfig, context_mgr: ControllerContextMgr,
                 context_engine: ContextEngine, runtime: WorkflowRuntime):
        super().__init__(config, context_mgr)
        self._context_engine = context_engine
        self._runtime = runtime
        self._state_machine = ReActStateMachine(runtime)
        self._model = self._init_model()
        self._agent_handler = None
        self._setup_state_handlers()

    def _setup_state_handlers(self):
        """设置状态处理器"""
        # 注册同步状态处理器
        self._state_machine.register_state_handler(ReActStatus.INITIALIZED, self._handle_initialized_state)
        self._state_machine.register_state_handler(ReActStatus.LLM_RESPONSE, self._handle_llm_response_state)
        self._state_machine.register_state_handler(ReActStatus.TOOL_INVOKED, self._handle_tool_invoked_state)
        self._state_machine.register_state_handler(ReActStatus.COMPLETED, self._handle_completed_state)
        self._state_machine.register_state_handler(ReActStatus.INTERRUPTED, self._handle_interrupted_state)

        # 注册流式状态处理器
        self._state_machine.register_state_handler(ReActStatus.INITIALIZED, self._stream_handle_initialized_state, is_stream=True)
        self._state_machine.register_state_handler(ReActStatus.LLM_RESPONSE, self._stream_handle_llm_response_state, is_stream=True)
        self._state_machine.register_state_handler(ReActStatus.TOOL_INVOKED, self._stream_handle_tool_invoked_state, is_stream=True)
        self._state_machine.register_state_handler(ReActStatus.COMPLETED, self._stream_handle_completed_state, is_stream=True)
        self._state_machine.register_state_handler(ReActStatus.INTERRUPTED, self._stream_handle_interrupted_state, is_stream=True)

    async def  execute(self, inputs: Dict) -> Dict:
        """执行完整的ReAct迭代流程"""
        logger.info(f"Starting ReAct execution with inputs: {inputs}")
        self._state_machine.set_current_event(ReActEvent.USER_INVOKE)

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
        self._state_machine.set_current_event(ReActEvent.USER_INVOKE)

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

    def set_agent_handler(self, agent_handler: AgentHandler):
        """设置Agent处理器"""
        self._agent_handler = agent_handler

    # 状态处理方法
    async def _handle_initialized_state(self, inputs: Dict) -> Dict:
        """处理初始化状态"""
        controller_output = await self.invoke(ReActControllerInput(**inputs), None)
        self._state_machine.set_current_event(ReActEvent.USER_INVOKE)
        self._state_machine.set_current_status(ReActStatus.LLM_RESPONSE)
        return await self._handle_llm_response_state(inputs, controller_output)

    async def _handle_llm_response_state(self, inputs: Dict, controller_output: ReActControllerOutput = None) -> Dict:
        """处理LLM响应状态"""
        if controller_output is None:
            controller_output = await self.invoke(ReActControllerInput(**inputs), None)

        logger.info(f"LLM response contains {len(controller_output.sub_tasks)} sub_tasks")
        logger.info(f"LLM response content: {controller_output.llm_output.content if controller_output.llm_output else 'None'}")
        logger.info(f"Should continue: {controller_output.should_continue}")
        for i, task in enumerate(controller_output.sub_tasks):
            logger.info(f"SubTask {i}: {task.func_name} ({task.sub_task_type}) with args: {task.func_args}")

        if controller_output.should_continue:
            # 更新状态数据 - 在执行前先保存
            self._state_machine.update_state_data({
                "llm_output": controller_output.llm_output.model_dump() if controller_output.llm_output else None,
                "sub_tasks": controller_output.sub_tasks  # 直接存储对象列表
            })

            # 创建一个临时的TaskContext，因为新架构中不再依赖TaskContext
            # 但为了保持与AgentHandler的兼容性，我们需要传递一个context对象
            from jiuwen.core.agent.task.task_context import AgentRuntime
            temp_context = AgentRuntime(trace_id=self._runtime.session_id())
            temp_context.set_controller_context_manager(self._context_mgr)

            # 直接传递 controller_output.sub_tasks 而不是从状态机获取
            completed_sub_tasks, exec_result = await self._execute_sub_tasks(temp_context, controller_output.sub_tasks)
            self._state_machine.set_current_event(ReActEvent.INVOKE_TOOL)
            return await self._handle_sub_task_result(exec_result, inputs, completed_sub_tasks)
        else:
            # 更新状态数据
            final_result = controller_output.llm_output.content if controller_output.llm_output else ""
            logger.info(f"Setting final result: {final_result}")

            # 先获取当前状态数据，然后更新
            current_state_data = self._state_machine.get_state_data()
            logger.info(f"Current state data before update: {current_state_data}")

            self._state_machine.update_state_data({
                "llm_output": controller_output.llm_output.model_dump() if controller_output.llm_output else None,
                "sub_tasks": controller_output.sub_tasks,
                "final_result": final_result
            })

            # 验证更新后的状态数据
            updated_state_data = self._state_machine.get_state_data()
            logger.info(f"State data after update: {updated_state_data}")

            self._state_machine.set_current_event(ReActEvent.FINISH)
            self._state_machine.set_current_status(ReActStatus.COMPLETED)
            return await self._handle_completed_state(inputs, controller_output)

    async def _handle_tool_invoked_state(self, inputs: Dict, completed_sub_tasks: List[SubTask] = None) -> Dict:
        """处理工具调用状态"""
        controller_output = await self.invoke(ReActControllerInput(**inputs), None)
        self._state_machine.set_current_event(ReActEvent.INVOKE_TOOL_FINISHED)
        self._state_machine.set_current_status(ReActStatus.LLM_RESPONSE)
        return await self._handle_llm_response_state(inputs, controller_output)

    async def _handle_completed_state(self, inputs: Dict, controller_output: ReActControllerOutput = None) -> Dict:
        """处理完成状态"""
        logger.info("Entering _handle_completed_state")

        # 检查状态机中的数据
        state_data = self._state_machine.get_state_data()
        logger.info(f"State data in _handle_completed_state: {state_data}")

        if controller_output and controller_output.llm_output:
            final_result = controller_output.llm_output.content
            logger.info(f"Setting final result from controller_output: {final_result}")
            self._state_machine.update_state_data({"final_result": final_result})

        final_result = self._state_machine.get_final_result()
        logger.info(f"Retrieved final result from state machine: {final_result}")

        # 如果状态机中的结果为空，但controller_output有内容，直接使用controller_output的内容
        if not final_result and controller_output and controller_output.llm_output and controller_output.llm_output.content:
            final_result = controller_output.llm_output.content
            logger.info(f"Using controller_output content as fallback: {final_result}")

        return {"output": final_result, "result_type": "answer"}

    async def _handle_interrupted_state(self, inputs: Dict) -> Dict:
        """处理中断状态"""
        user_input = InteractiveInput()
        state_data = self._state_machine.get_state_data()
        interrupt_state = state_data.get("interrupt_state", {})

        user_input.update(interrupt_state.get("interrupt_component_id", ""), inputs.get("query"))

        # 更新SubTask参数
        sub_tasks = state_data.get("sub_tasks", [])
        logger.info(f"Retrieved {len(sub_tasks)} sub_tasks from interrupted state")
        if sub_tasks:
            # 直接修改 SubTask 对象的 func_args
            sub_tasks[0].func_args = user_input
            self._state_machine.update_state_data({"sub_tasks": sub_tasks})
        else:
            logger.error("No sub_tasks found in interrupted state!")

        # 创建一个临时的TaskContext
        from jiuwen.core.agent.task.task_context import AgentRuntime
        temp_context = AgentRuntime(trace_id=self._runtime.session_id())
        temp_context.set_controller_context_manager(self._context_mgr)

        # 直接传递 sub_tasks 而不是让 _execute_sub_tasks 从状态机获取
        completed_sub_tasks, exec_result = await self._execute_sub_tasks(temp_context, sub_tasks)
        return await self._handle_sub_task_result(exec_result, inputs, completed_sub_tasks)

    # 流式状态处理方法
    async def _stream_handle_initialized_state(self, inputs: Dict):
        """流式处理初始化状态"""
        controller_stream = self.stream(ReActControllerInput(**inputs), None)
        controller_output = None

        async for item in controller_stream:
            if isinstance(item, BaseMessageChunk):
                if item.content:
                    yield {"output": item.content, "result_type": "partial"}
            elif isinstance(item, ReActControllerOutput):
                controller_output = item
                break

        self._state_machine.set_current_event(ReActEvent.USER_INVOKE)
        self._state_machine.set_current_status(ReActStatus.LLM_RESPONSE)
        async for result in self._stream_handle_llm_response_state(inputs, controller_output):
            yield result

    async def _stream_handle_llm_response_state(self, inputs: Dict, controller_output: ReActControllerOutput = None):
        """流式处理LLM响应状态"""
        if controller_output is None:
            # 这里应该是流式调用，但为了简化，先用同步调用
            controller_output = await self.invoke(ReActControllerInput(**inputs), None)

        logger.info(f"Stream LLM response contains {len(controller_output.sub_tasks)} sub_tasks")

        if controller_output.should_continue:
            # 更新状态数据
            self._state_machine.update_state_data({
                "llm_output": controller_output.llm_output.model_dump() if controller_output.llm_output else None,
                "sub_tasks": controller_output.sub_tasks
            })

            # 创建一个临时的TaskContext
            from jiuwen.core.agent.task.task_context import AgentRuntime
            temp_context = AgentRuntime(trace_id=self._runtime.session_id())
            temp_context.set_controller_context_manager(self._context_mgr)

            # 直接传递 controller_output.sub_tasks
            completed_sub_tasks, exec_result = await self._execute_sub_tasks(temp_context, controller_output.sub_tasks)
            self._state_machine.set_current_event(ReActEvent.INVOKE_TOOL)
            async for result in self._stream_handle_sub_task_result(exec_result, inputs, completed_sub_tasks):
                yield result
        else:
            # 更新状态数据
            self._state_machine.update_state_data({
                "llm_output": controller_output.llm_output.model_dump() if controller_output.llm_output else None,
                "sub_tasks": controller_output.sub_tasks
            })
            self._state_machine.set_current_event(ReActEvent.FINISH)
            self._state_machine.set_current_status(ReActStatus.COMPLETED)
            async for result in self._stream_handle_completed_state(inputs, controller_output):
                yield result

    async def _stream_handle_tool_invoked_state(self, inputs: Dict, completed_sub_tasks: List[SubTask] = None):
        """流式处理工具调用状态"""
        controller_stream = self.stream(ReActControllerInput(**inputs), None)
        controller_output = None

        async for item in controller_stream:
            if isinstance(item, BaseMessageChunk):
                if item.content:
                    yield {"output": item.content, "result_type": "partial"}
            elif isinstance(item, ReActControllerOutput):
                controller_output = item
                break

        self._state_machine.set_current_event(ReActEvent.INVOKE_TOOL_FINISHED)
        self._state_machine.set_current_status(ReActStatus.LLM_RESPONSE)
        async for result in self._stream_handle_llm_response_state(inputs, controller_output):
            yield result

    async def _stream_handle_completed_state(self, inputs: Dict, controller_output: ReActControllerOutput = None):
        """流式处理完成状态"""
        if controller_output and controller_output.llm_output:
            final_result = controller_output.llm_output.content
            self._state_machine.update_state_data({"final_result": final_result})

        final_result = self._state_machine.get_final_result()
        yield {"output": final_result, "result_type": "answer"}

    async def _stream_handle_interrupted_state(self, inputs: Dict):
        """流式处理中断状态"""
        user_input = InteractiveInput()
        state_data = self._state_machine.get_state_data()
        interrupt_state = state_data.get("interrupt_state", {})

        user_input.update(interrupt_state.get("interrupt_component_id", ""), inputs.get("query"))

        # 更新SubTask参数
        sub_tasks = state_data.get("sub_tasks", [])
        if sub_tasks:
            # 直接修改 SubTask 对象的 func_args
            sub_tasks[0].func_args = user_input
            self._state_machine.update_state_data({"sub_tasks": sub_tasks})

        # 创建一个临时的TaskContext
        from jiuwen.core.agent.task.task_context import AgentRuntime
        temp_context = AgentRuntime(trace_id=self._runtime.session_id())
        temp_context.set_controller_context_manager(self._context_mgr)

        # 直接传递 sub_tasks
        completed_sub_tasks, exec_result = await self._execute_sub_tasks(temp_context, sub_tasks)
        async for result in self._stream_handle_sub_task_result(exec_result, inputs, completed_sub_tasks):
            yield result

    async def _handle_sub_task_result(self, exec_result, inputs: Dict, completed_sub_tasks: List[SubTask]) -> Dict:
        """处理SubTask执行结果"""
        logger.info(f"Handling sub task result: {exec_result}")
        logger.info(f"Completed sub tasks: {[task.func_name for task in completed_sub_tasks]}")

        if isinstance(exec_result, list) and exec_result[0].get('type') == '__interaction__':
            # 处理交互请求
            interrupt_data = {
                "interrupt_component_id": exec_result[0].get('payload').get('id'),
                "question": exec_result[0].get('payload').get('value')
            }
            self._state_machine.update_state_data({"interrupt_state": interrupt_data})
            self._state_machine.set_current_status(ReActStatus.INTERRUPTED)
            return {"output": interrupt_data["question"], "result_type": "question"}
        else:
            # 处理正常结果
            logger.info("Adding tool results to chat history before state transition")
            await self._update_chat_history_with_tool_results(completed_sub_tasks)
            self._state_machine.set_current_status(ReActStatus.TOOL_INVOKED)
            return await self._handle_tool_invoked_state(inputs, completed_sub_tasks)

    async def _stream_handle_sub_task_result(self, exec_result, inputs: Dict, completed_sub_tasks: List[SubTask]):
        """流式处理SubTask执行结果"""
        if isinstance(exec_result, list) and exec_result[0].get('type') == '__interaction__':
            # 处理交互请求
            interrupt_data = {
                "interrupt_component_id": exec_result[0].get('payload')[0],
                "question": exec_result[0].get('payload')[1]
            }
            self._state_machine.update_state_data({"interrupt_state": interrupt_data})
            self._state_machine.set_current_status(ReActStatus.INTERRUPTED)
            yield {"output": interrupt_data["question"], "result_type": "question"}
        else:
            # 处理正常结果
            await self._update_chat_history_with_tool_results(completed_sub_tasks)
            self._state_machine.set_current_status(ReActStatus.TOOL_INVOKED)
            async for result in self._stream_handle_tool_invoked_state(inputs, completed_sub_tasks):
                yield result

    async def _execute_sub_tasks(self, context: 'AgentRuntime', sub_tasks: List[SubTask] = None):
        """执行SubTask列表"""
        if sub_tasks is None:
            # 如果没有直接传递，则从状态机获取
            state_data = self._state_machine.get_state_data()
            sub_tasks = state_data.get("sub_tasks", [])
            logger.info(f"Retrieved {len(sub_tasks)} sub_tasks from state_data")
        else:
            logger.info(f"Using directly passed {len(sub_tasks)} sub_tasks")

        completed_sub_tasks = []
        exec_result = None

        logger.info(f"Executing {len(sub_tasks)} sub tasks")

        if not sub_tasks:
            logger.warning("No sub_tasks found in state_data!")
            return completed_sub_tasks, None

        for i, sub_task in enumerate(sub_tasks):
            # sub_task 现在直接是 SubTask 对象，不需要重建
            if not isinstance(sub_task, SubTask):
                logger.error(f"SubTask {i} is not a SubTask instance: {type(sub_task)}")
                continue

            logger.info(f"Executing sub task: {sub_task.func_name} with args: {sub_task.func_args}")

            try:
                # 执行SubTask
                from jiuwen.core.agent.handler.base import AgentHandlerInputs
                inputs = AgentHandlerInputs(context=context, name=sub_task.func_name, arguments=sub_task.func_args)
                exec_result = await self._agent_handler.invoke(sub_task.sub_task_type, inputs)
                logger.info(f"Sub task {sub_task.func_name} result: {exec_result}")

                # 更新结果
                import json
                sub_task.result = json.dumps(exec_result, ensure_ascii=False)
            except Exception as e:
                # 插件执行失败时，添加失败信息
                error_msg = f"Tool execution failed: {str(e)}"
                logger.error(f"Sub task {sub_task.func_name} failed: {error_msg}")

                import json
                error_result = {
                    "error": True,
                    "message": error_msg,
                    "tool_name": sub_task.func_name
                }
                sub_task.result = json.dumps(error_result, ensure_ascii=False)
                exec_result = error_result

            completed_sub_tasks.append(sub_task)

        return completed_sub_tasks, exec_result

    async def _update_chat_history_with_tool_results(self, completed_sub_tasks: List[SubTask]):
        """更新对话历史，添加工具调用结果"""
        if not completed_sub_tasks:
            logger.warning("No completed sub tasks to add to chat history")
            return

        from jiuwen.core.utils.llm.messages import ToolMessage

        agent_context = self._context_engine.get_agent_context(self._runtime.session_id())
        logger.info(f"Adding {len(completed_sub_tasks)} tool results to chat history")

        # 为每个完成的SubTask创建ToolMessage并添加到对话历史
        for sub_task in completed_sub_tasks:
            if sub_task.result:  # 确保有执行结果
                tool_message = ToolMessage(content=sub_task.result, tool_call_id=sub_task.id)
                agent_context.add_message(tool_message)
                logger.info(f"Added tool result to chat history: {sub_task.func_name} -> {sub_task.result[:100]}...")
            else:
                logger.warning(f"Sub task {sub_task.func_name} has no result to add")

        # 调试：打印当前对话历史
        current_messages = agent_context.get_messages()
        logger.info(f"Current chat history length: {len(current_messages)}")
        for i, msg in enumerate(current_messages[-5:]):  # 只打印最后5条消息
            content_preview = ""
            if hasattr(msg, 'content') and msg.content:
                content_preview = msg.content[:50]
            elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                content_preview = f"tool_calls: {len(msg.tool_calls)}"
            logger.info(f"Message {i}: {msg.role} - {content_preview}...")

    def _update_llm_response_to_context(self, llm_output: AIMessage):
        """更新LLM响应到上下文"""
        if llm_output:
            agent_context = self._context_engine.get_agent_context(self._runtime.session_id())
            agent_context.add_message(llm_output)

    async def invoke(self, inputs: ReActControllerInput, context: AgentRuntime) -> ReActControllerOutput:
        # 只在初始请求时添加用户输入到对话历史
        # 在工具调用后的请求中，对话历史已经包含了用户消息和工具结果
        agent_context = self._context_engine.get_agent_context(self._runtime.session_id())

        # 检查对话历史中是否已经包含当前用户输入
        last_message = agent_context.get_latest_message()
        should_add_user_message = True

        if last_message:
            # 如果最后一条消息是工具消息，说明这是工具调用后的请求，不需要再添加用户消息
            if last_message.role == 'tool':
                should_add_user_message = False
                logger.info("Skipping user message addition - this is a post-tool-call request")
            # 如果最后一条消息已经是相同的用户输入，也不需要重复添加
            elif last_message.role == 'user' and last_message.content == inputs.query:
                should_add_user_message = False
                logger.info("Skipping user message addition - same message already exists")

        if should_add_user_message:
            user_message = HumanMessage(content=inputs.query)
            agent_context.add_message(user_message)
            logger.info(f"Added user message to chat history: {inputs.query}")

        tools = self._format_tools_info()
        chat_history = self._get_latest_chat_history()
        llm_inputs = self._format_llm_inputs(inputs, chat_history)

        result = await self._invoke_llm_and_parse_output(llm_inputs, tools)
        self._update_llm_response_to_context(result.llm_output)
        return result

    async def stream(self,
                     inputs: ReActControllerInput,
                     context: AgentRuntime
                     ) -> AsyncIterator[Union[BaseMessageChunk, ReActControllerOutput]]:
        # 只在初始请求时添加用户输入到对话历史
        # 在工具调用后的请求中，对话历史已经包含了用户消息和工具结果
        agent_context = self._context_engine.get_agent_context(self._runtime.session_id())

        # 检查对话历史中是否已经包含当前用户输入
        last_message = agent_context.get_latest_message()
        should_add_user_message = True

        if last_message:
            # 如果最后一条消息是工具消息，说明这是工具调用后的请求，不需要再添加用户消息
            if last_message.role == 'tool':
                should_add_user_message = False
                logger.info("Skipping user message addition - this is a post-tool-call request")
            # 如果最后一条消息已经是相同的用户输入，也不需要重复添加
            elif last_message.role == 'user' and last_message.content == inputs.query:
                should_add_user_message = False
                logger.info("Skipping user message addition - same message already exists")

        if should_add_user_message:
            user_message = HumanMessage(content=inputs.query)
            agent_context.add_message(user_message)
            logger.info(f"Added user message to chat history: {inputs.query}")

        tools = self._format_tools_info()
        chat_history = self._get_latest_chat_history()
        llm_inputs = self._format_llm_inputs(inputs, chat_history)

        response = AIMessage()
        async for chunk in self._stream_llm(llm_inputs, tools):
            yield chunk
            if self._check_if_last_chunk(chunk):
                self._transform_chunk_to_ai_message(chunk, response)

        result = self._parse_llm_output(response)
        self._update_llm_response_to_context(result.llm_output)
        yield result

    def _get_latest_chat_history(self) -> List[BaseMessage]:
        """获取最新的对话历史"""
        # 从context engine获取对话历史
        agent_context = self._context_engine.get_agent_context(self._runtime.session_id())
        chat_history = agent_context.get_messages()
        chat_history = chat_history[-2 * self._config.constrain.reserved_max_chat_rounds:]
        return chat_history

    def _format_tools_info(self) -> List[ToolInfo]:
        tool_info_list: List[ToolInfo] = list()
        workflows_metadata = self._config.workflows
        plugins_metadata = self._config.plugins
        tool_info_list.extend(FormatUtils.format_workflows_metadata(workflows_metadata))
        tool_info_list.extend(FormatUtils.format_plugins_metadata(plugins_metadata))
        return tool_info_list

    async def _invoke_llm_and_parse_output(self, llm_inputs: List[BaseMessage], tools: List[ToolInfo]) -> ReActControllerOutput:
        try:
            response = await self._model.ainvoke(self._config.model.model_info.model_name, llm_inputs, tools)
        except Exception as e:
            raise JiuWenBaseException(
                error_code=StatusCode.INVOKE_LLM_FAILED.code,
                message=StatusCode.INVOKE_LLM_FAILED.errmsg
            ) from e

        return self._parse_llm_output(response)

    async def _stream_llm(self, llm_inputs: List[BaseMessage], tools: List[ToolInfo]) -> AsyncIterator[BaseMessageChunk]:
        try:
            async for chunk in self._model.astream(self._config.model.model_info.model_name, llm_inputs, tools):
                if self._check_if_valid_chunk(chunk):
                    yield chunk

        except Exception as e:
            import traceback
            tmp = traceback.format_exc()

            raise JiuWenBaseException(
                error_code=StatusCode.INVOKE_LLM_FAILED.code,
                message=StatusCode.INVOKE_LLM_FAILED.errmsg
            ) from e

    def _format_system_prompt_template(self, user_fields):
        """格式化系统提示模板"""
        return (Template(name=self._config.prompt_template_name, content=self._config.prompt_template)
                .format(user_fields)
                .to_messages())

    def _init_model(self):
        return ModelFactory().get_model(model_provider=self._config.model.model_provider, api_base=self._config.model.model_info.api_base,
                                        api_key=self._config.model.model_info.api_key)

    def _update_llm_response_to_context(self, llm_output: AIMessage):
        """更新LLM响应到上下文"""
        if llm_output:
            # 获取agent context并添加消息
            agent_context = self._context_engine.get_agent_context(self._runtime.session_id())
            agent_context.add_message(llm_output)

    def _format_sub_tasks(self, tool_calls: List[ToolCall]) -> List[SubTask]:
        if not tool_calls:
            return []
        result = []
        for tool_call in tool_calls:
            tool_call_id = tool_call.id
            tool_call_info = tool_call.function
            tool_call_name = tool_call_info.name
            tool_call_args = FormatUtils.json_loads(tool_call_info.arguments)
            tool_call_type = self._check_sub_task_type(tool_call_name)
            result.append(
                SubTask(id=tool_call_id, func_name=tool_call_name, func_args=tool_call_args, sub_task_type=tool_call_type))
        return result

    def _check_sub_task_type(self, tool_call_name: str) -> SubTaskType:
        result = SubTaskType.UNDEFINED
        workflows_metadata = self._config.workflows
        for workflow in workflows_metadata:
            if tool_call_name == workflow.name:
                result = SubTaskType.WORKFLOW
                break
        if result == SubTaskType.UNDEFINED:
            plugins_metadata = self._config.plugins
            for plugin in plugins_metadata:
                if tool_call_name == plugin.name:
                    result = SubTaskType.PLUGIN
                    break
        if result == SubTaskType.UNDEFINED:
            raise JiuWenBaseException()
        return result

    def _format_llm_inputs(self, inputs: ReActControllerInput, chat_history: List[BaseMessage]):
        user_fields = inputs.user_fields
        system_prompt = self._format_system_prompt_template(user_fields)
        return FormatUtils.create_llm_inputs(system_prompt, chat_history)

    def _parse_llm_output(self, response: BaseMessage):
        llm_output = response
        sub_tasks = self._format_sub_tasks(llm_output.tool_calls)
        should_continue = isinstance(sub_tasks, list) and len(sub_tasks) > 0
        return ReActControllerOutput(should_continue=should_continue, llm_output=llm_output, sub_tasks=sub_tasks)

    @staticmethod
    def _check_if_last_chunk(chunk: BaseMessageChunk):
        if isinstance(chunk.usage_metadata, UsageMetadata):
            return chunk.usage_metadata.finish_reason in ["stop", "tool_calls"]
        return False

    @staticmethod
    def _transform_chunk_to_ai_message(chunk: BaseMessageChunk, message: AIMessage):
        message.content = chunk.content
        message.tool_calls = chunk.tool_calls
        message.usage_metadata = chunk.usage_metadata

    @staticmethod
    def _check_if_valid_chunk(chunk):
        return (chunk.content or (hasattr(chunk, "reason_content") and chunk.reason_content) or
                (hasattr(chunk, "tool_calls") and chunk.tool_calls))
