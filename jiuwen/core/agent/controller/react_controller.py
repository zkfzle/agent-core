"""Controller of ReActAgent"""
from typing import List, Dict, Optional, Union, Any
import json

from jiuwen.core.agent.controller.base import Controller
from jiuwen.core.agent.handler.base import AgentHandler, AgentHandlerInputs
from jiuwen.agent.config.base import AgentConfig
from jiuwen.core.agent.task.sub_task import SubTask
from jiuwen.core.runtime.interaction.base import AgentInterrupt
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.context_engine.engine import ContextEngine
from jiuwen.core.stream.base import OutputSchema
from jiuwen.core.utils.llm.hash_util import generate_key
from jiuwen.core.utils.llm.model_utils.model_factory import ModelFactory
from jiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from jiuwen.core.common.logging import logger
from jiuwen.core.agent.controller.utils import ReActControllerUtils, ReActControllerOutput, ReActControllerInput
from jiuwen.agent.common.enum import ReActControllerStatus


class ReActState:
    """ReAct状态管理类"""

    def __init__(self, runtime: Runtime):
        self._runtime = runtime

    def is_interrupted(self) -> bool:
        track_state = self._runtime.get_state("react_state")
        return (track_state and
                track_state.get("status") == ReActControllerStatus.INTERRUPTED.value and
                track_state.get("sub_tasks") is not None)

    def get_interrupted_sub_tasks(self) -> List[SubTask]:
        state_data = self._runtime.get_state("react_state") or {}
        return state_data.get("sub_tasks", [])

    def save_interrupt_state(self, sub_tasks: List[SubTask]):
        self._runtime.update_state({
            "react_state": {
                "status": ReActControllerStatus.INTERRUPTED.value,
                "sub_tasks": sub_tasks
            }
        })

    def get_current_status(self) -> ReActControllerStatus:
        """获取当前状态"""
        track_state = self._runtime.get_state("react_state")
        if not track_state:
            return ReActControllerStatus.NORMAL

        status_value = track_state.get("status", ReActControllerStatus.NORMAL.value)
        try:
            return ReActControllerStatus(status_value)
        except ValueError:
            # 如果状态值不是有效的枚举值，返回默认状态
            return ReActControllerStatus.NORMAL

    def set_status(self, status: ReActControllerStatus, sub_tasks: List[SubTask] = None):
        """设置状态"""
        self._runtime.update_state({
            "react_state": {
                "status": status.value,
                "sub_tasks": sub_tasks or []
            }
        })


class ReActController(Controller):
    """优化的ReAct控制器 - 清晰的Reason→Act→Observe→Decide循环"""

    def __init__(self, config: AgentConfig, context_engine: ContextEngine, runtime: Runtime):
        super().__init__(config)
        self._context_engine = context_engine
        self._runtime = runtime
        self._model = self._init_model()

        # 组件初始化
        self._state = ReActState(runtime)
        self._agent_handler = None

    def _init_model(self):
        model_id = generate_key(
            self._config.model.model_info.api_key,
            self._config.model.model_info.api_base,
            self._config.model.model_provider
        )

        model = self._runtime.get_model(model_id=model_id)

        if model is None:
            model = ModelFactory().get_model(
                model_provider=self._config.model.model_provider,
                api_base=self._config.model.model_info.api_base,
                api_key=self._config.model.model_info.api_key
            )
            self._runtime.add_model(model_id=model_id, model=model)

        return self._runtime.get_model(model_id=model_id)

    @staticmethod
    def _validate_inputs(inputs: Dict):
        """验证输入"""
        if isinstance(inputs.get("query"), InteractiveInput):
            raise JiuWenBaseException(5000, "Non-interrupt status data format error.")

    def set_agent_handler(self, agent_handler: AgentHandler):
        """设置Agent处理器"""
        self._agent_handler = agent_handler

    async def execute(self, inputs: Dict) -> Dict:
        """主执行流程 - 统一的ReAct循环"""
        logger.info(f"Starting ReAct execution with inputs: {inputs}")

        # 输入验证（仅在非中断恢复时进行）
        if not self._state.is_interrupted():
            self._validate_inputs(inputs)

        # 执行ReAct主循环
        return await self._run_react_loop(inputs)

    async def _run_react_loop(self, inputs: Dict) -> Dict | list:
        """核心ReAct循环：Reason→Act→Observe→Decide，包含中断恢复处理"""

        # 标准ReAct循环
        for iteration in range(self._config.constrain.max_iteration):
            logger.info(f"ReAct iteration {iteration + 1}")

            # 首先检查是否需要处理中断恢复
            if self._state.is_interrupted():
                interrupt_data = await self._resume_task(inputs)
                if interrupt_data is not None:
                    return interrupt_data
                # 中断恢复完成后，继续到下一次迭代进行reason总结
                continue

            # 1. Reason: LLM推理生成计划
            plan_result = await self.reason(inputs)

            # 2. Decide: 判断是否需要继续
            if not plan_result.should_continue:
                self._state.set_status(ReActControllerStatus.COMPLETED)  # 设置完成状态
                final_result = {"output": plan_result.llm_output.content, "result_type": "answer"}
                await self._runtime.write_stream(OutputSchema(type="answer", index=0, payload=final_result))
                return final_result

            # 3. Act: 执行工具调用
            completed_tasks, exec_result = await self.act(plan_result.sub_tasks)

            # 4. Observe: 观察结果并更新历史
            interrupt_data = await self.observe(completed_tasks, exec_result)
            if interrupt_data is not None:
                return interrupt_data

        # 设置超时状态并返回超时结果
        self._state.set_status(ReActControllerStatus.TIMEOUT)
        timeout_result = {"output": "执行超过最大迭代次数", "result_type": "answer"}
        await self._runtime.write_stream(OutputSchema(type="answer", index=0, payload=timeout_result))
        return timeout_result

    async def _resume_task(self, inputs: Dict) -> Optional[Dict]:
        """恢复中断的任务

        Args:
            inputs: 输入数据，包含InteractiveInput

        Returns:
            如果任务完成并需要返回结果，返回结果字典；否则返回None继续循环
        """
        if not isinstance(inputs.get("query"), InteractiveInput):
            raise JiuWenBaseException(5000, "Interrupt status data format error.")

        logger.info(f"Processing interrupt recovery within ReAct loop: {inputs}")

        # 添加用户输入
        for _, query in inputs.get("query").user_inputs.items():
            ReActControllerUtils.add_user_message(query, self._context_engine, self._runtime)

        # 获取中断的任务
        sub_tasks = self._state.get_interrupted_sub_tasks()
        if not sub_tasks:
            self._state.set_status(ReActControllerStatus.NORMAL)
            return None

        # 更新第一个任务的参数
        sub_tasks[0].func_args = inputs.get("query", "")

        # 执行恢复的任务
        completed_tasks, exec_result = await self.act(sub_tasks)

        # 观察结果并更新历史
        interrupt_data = await self.observe(completed_tasks, exec_result)

        # 如果观察阶段返回了交互结果，直接返回
        if interrupt_data is not None:
            return interrupt_data

        # 恢复正常状态
        self._state.set_status(ReActControllerStatus.NORMAL)
        return None

    async def act(self, sub_tasks: List[SubTask]) -> tuple[List[SubTask], Any]:
        """Act: 执行工具 - 执行SubTask列表，返回(完成的任务, 执行结果)"""
        if not sub_tasks:
            return [], None

        completed_tasks = []
        exec_result = None

        for sub_task in sub_tasks:
            try:
                inputs = AgentHandlerInputs(
                    context=self._runtime,
                    name=sub_task.func_name,
                    arguments=sub_task.func_args
                )
                exec_result = await self._agent_handler.invoke(sub_task.sub_task_type, inputs)
                sub_task.result = json.dumps(exec_result, ensure_ascii=False)
                completed_tasks.append(sub_task)

            except AgentInterrupt as e:
                interrupt_result = ReActControllerUtils.create_interrupt_result(e, sub_task.func_name)
                sub_task.result = interrupt_result
                exec_result = interrupt_result

                # 保存中断状态
                self._state.save_interrupt_state(sub_tasks)
                break

        return completed_tasks, exec_result

    async def observe(self, completed_tasks: List[SubTask], exec_result: Any = None) -> Any | None:
        """Observe: 观察结果并更新历史"""
        # 检查是否为交互中断结果
        if exec_result and ReActControllerUtils.is_interaction_result(exec_result):
            # 处理交互请求 - 写入流式输出
            interrupt_data_list = []
            for output_scheme in exec_result.get("value", []):
                await self._runtime.write_stream(output_scheme)
                interrupt_data_list.append(output_scheme)
            return interrupt_data_list

        # 更新历史 - 这里会将中断恢复任务的结果也加入对话历史
        ReActControllerUtils.add_tool_results(completed_tasks, self._context_engine, self._runtime)
        return None

    # === ReAct推理引擎 ===
    async def reason(self, inputs: Union[Dict, ReActControllerInput],
                     context: Optional[Runtime] = None) -> ReActControllerOutput:
        """推理阶段：分析情况并生成行动计划"""
        # 转换为ReActControllerInput
        if isinstance(inputs, dict):
            controller_input = ReActControllerInput(**inputs)
        else:
            controller_input = inputs

        # 统一处理用户消息 - 包括普通查询和InteractiveInput
        if not isinstance(controller_input.query, InteractiveInput):
            ReActControllerUtils.add_user_message(controller_input.query, self._context_engine, self._runtime)

        # 准备LLM输入
        tools = self._runtime.get_tool_info()
        chat_history = ReActControllerUtils.get_chat_history(self._context_engine, self._runtime, self._config)
        llm_inputs = ReActControllerUtils.format_llm_inputs(controller_input, chat_history, self._config)
        logger.info(f"React llm inputs: {llm_inputs}")
        # 调用LLM
        try:
            response = await self._model.ainvoke(
                self._config.model.model_info.model_name,
                llm_inputs,
                tools
            )
        except Exception as e:
            raise JiuWenBaseException(
                error_code=StatusCode.INVOKE_LLM_FAILED.code,
                message=StatusCode.INVOKE_LLM_FAILED.errmsg
            ) from e

        # 解析结果
        result = ReActControllerUtils.parse_llm_output(response, self._config)
        ReActControllerUtils.add_ai_message(result.llm_output, self._context_engine, self._runtime)
        logger.info(f"React llm output: {result.llm_output}")
        return result
