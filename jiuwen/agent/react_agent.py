#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
"""ReActAgent"""
import json
from typing import Dict, Iterator, Any, List

from jiuwen.agent.common.enum import ControllerType, ReActStatus, ReActEvent
from jiuwen.agent.common.schema import WorkflowSchema, PluginSchema
from jiuwen.agent.config.react_config import ReActAgentConfig
from jiuwen.core.agent.agent import Agent
from jiuwen.core.agent.controller.react_controller import ReActController, ReActControllerOutput, ReActControllerUtils, \
    ReActControllerInput
from jiuwen.core.agent.handler.base import AgentHandlerImpl, AgentHandlerInputs
from jiuwen.agent.state.react_state import ReActState
from jiuwen.core.agent.task.sub_task import SubTask
from jiuwen.core.agent.task.task import Task
from jiuwen.core.agent.task.task_context import TaskContext
from jiuwen.core.common.logging import logger
from jiuwen.core.component.common.configs.model_config import ModelConfig
from jiuwen.core.context.controller_context.controller_context_manager import ControllerContextMgr
from jiuwen.core.graph.interrupt.interactive_input import InteractiveInput
from jiuwen.core.utils.llm.messages import ToolMessage
from jiuwen.core.utils.tool.base import Tool
from jiuwen.core.workflow.base import Workflow

REACT_AGENT_STATE_KEY = "react_agent_state"


def create_react_agent_config(agent_id: str,
                              agent_version: str,
                              description: str,
                              workflows: List[WorkflowSchema],
                              plugins: List[PluginSchema],
                              model: ModelConfig,
                              prompt_template: List[Dict]):
    config = ReActAgentConfig(id=agent_id,
                              version=agent_version,
                              description=description,
                              workflows=workflows,
                              plugins=plugins,
                              model=model,
                              prompt_template=prompt_template)
    return config


def create_react_agent(agent_config: ReActAgentConfig,
                       workflows: List[Workflow] = None,
                       tools: List[Tool] = None):
    agent = ReActAgent(agent_config)
    agent.bind_workflows(workflows)
    agent.bind_tools(tools)
    return agent


class ReActAgent(Agent):
    def __init__(self, agent_config: ReActAgentConfig):
        super().__init__(agent_config)
        self._state = None

        self.fsm_state_map: Dict = {
                              ReActStatus.INITIALIZED: self._handle_event_in_initialized,
                              ReActStatus.LLM_RESPONSE: self._handle_event_in_llm_response,
                              ReActStatus.TOOL_INVOKED: self._handle_event_in_tool_invoked,
                              ReActStatus.COMPLETED: self._handle_event_in_completed,
                              ReActStatus.INTERRUPTED: self._handle_event_in_interrupted
                             }
        self.fsm_event: ReActEvent = ReActEvent.NO_EVENT

    def _init_controller(self):
        if self._config.controller_type != ControllerType.ReActController:
            raise NotImplementedError("")
        return ReActController(self._config, self._controller_context_manager)

    def _init_agent_handler(self):
        return AgentHandlerImpl(self._config)

    def _init_controller_context_manager(self) -> ControllerContextMgr:
        return ControllerContextMgr(self._config)

    async def _handle_sub_task_result(self, exec_result, inputs: Dict, context: TaskContext,
                                    completed_sub_tasks: List[SubTask]):
        if isinstance(exec_result, list) and exec_result[0].get('type') == '__interaction__':
            self._state.interrupt_state.update(exec_result[0].get('payload')[0], exec_result[0].get('payload')[1])
            self._state.status = ReActStatus.INTERRUPTED
            self._store_state_to_context(context)
            return dict(output=self._state.interrupt_state.question, result_type="question")
        else:
            self._state.status = ReActStatus.TOOL_INVOKED
            self._store_state_to_context(context)
            return await self.fsm_state_map[self._state.status](inputs, context, completed_sub_tasks)

    async def _handle_event_in_initialized(self, inputs: Dict, context: TaskContext):
        logger.info(f"Enter in state: {self._state.status} with event: {self.fsm_event}")
        controller_output = self._controller.invoke(ReActControllerInput(**inputs), context)
        self.fsm_event = ReActEvent.USER_INVOKE
        self._state.status = ReActStatus.LLM_RESPONSE
        self._store_state_to_context(context)
        return await self.fsm_state_map[self._state.status](inputs, context, controller_output)

    async def _handle_event_in_llm_response(self, inputs: Dict, context: TaskContext, controller_output: ReActControllerOutput):
        logger.info(f"Enter in state: {self._state.status} with event: {self.fsm_event}")
        self._state.handle_llm_response_event(controller_output.llm_output, controller_output.sub_tasks)
        if controller_output.should_continue:
            completed_sub_tasks, exec_result = await self._execute_sub_tasks(context)
            self.fsm_event = ReActEvent.INVOKE_TOOL
            return await self._handle_sub_task_result(exec_result, inputs, context, completed_sub_tasks)
        else:
            self.fsm_event = ReActEvent.FINISH
            self._state.status = ReActStatus.COMPLETED
            self._store_state_to_context(context)
            return await self.fsm_state_map[self._state.status](inputs, context, controller_output)

    async def _handle_event_in_tool_invoked(self, inputs: Dict, context: TaskContext, completed_sub_tasks: List[SubTask]):
        logger.info(f"Enter in state: {self._state.status} with event: {self.fsm_event}")
        self._state.handle_tool_invoked_event(completed_sub_tasks)
        controller_output = self._controller.invoke(ReActControllerInput(**inputs), context)
        self.fsm_event = ReActEvent.INVOKE_TOOL_FINISHED
        self._state.status = ReActStatus.LLM_RESPONSE
        self._store_state_to_context(context)
        return await self.fsm_state_map[self._state.status](inputs, context, controller_output)

    async def _handle_event_in_completed(self, inputs: Dict, context: TaskContext, controller_output: ReActControllerOutput):
        logger.info(f"Enter in state: {self._state.status} with event: {self.fsm_event}")
        self._state.handle_react_completed_event(controller_output.llm_output.content)
        self._store_state_to_context(context)
        return dict(output=self._state.final_result, result_type="answer")

    async def _handle_event_in_interrupted(self, inputs: Dict, context: TaskContext):
        logger.info(f"Enter in state: {self._state.status} with event: {self.fsm_event}")
        user_input = InteractiveInput()
        user_input.update(self._state.interrupt_state.interrupt_component_id, inputs.get("query"))
        # 追问模式下只有一个sub_task,func_args要指定是user_input
        if len(self._state.sub_tasks) == 0:
            logger.error(f"sub_task is empty when _invoke_from_interrupted.")
            return dict(output=f"Fatal Error! Status:{self._state.status}", result_type="answer")
        self._state.sub_tasks[0].func_args = user_input
        completed_sub_tasks, exec_result = await self._execute_sub_tasks(context)
        return await self._handle_sub_task_result(exec_result, inputs, context, completed_sub_tasks)

    async def invoke(self, inputs: Dict) -> Dict:
        # 如果input是追问的问题，那么用户要保证conversation_id一致
        task: Task = self._task_manager.create_task(inputs.get("conversation_id"))
        context = task.context
        context.set_controller_context_manager(self._controller_context_manager)
        self._load_state_from_context(context)
        self.fsm_event = ReActEvent.USER_INVOKE

        if (self._state.status != ReActStatus.INITIALIZED) and (self._state.status != ReActStatus.INTERRUPTED):
            logger.error(f"state {self._state.status} is incorrect in invoke")
            return dict(output=f"Fatal Error! Status:{self._state.status}", result_type="answer")
        self._store_state_to_context(context)
        return await self.fsm_state_map[self._state.status](inputs, context)

    async def stream(self, inputs: Dict) -> Iterator[Any]:
        pass

    async def _execute_sub_tasks(self, context: TaskContext):
        to_exec_sub_tasks = self._state.sub_tasks
        completed_sub_tasks = []
        exec_result = None
        for st in to_exec_sub_tasks:
            inputs = AgentHandlerInputs(context=context, name=st.func_name, arguments=st.func_args)
            exec_result = await self._agent_handler.invoke(st.sub_task_type, inputs)
            st.result = json.dumps(exec_result, ensure_ascii=False)
            completed_sub_tasks.append(st)
        self._update_chat_history_in_context(completed_sub_tasks, context)
        return completed_sub_tasks, exec_result

    def _load_state_from_context(self, context: TaskContext):
        state_dict = context.state().get(REACT_AGENT_STATE_KEY)
        if state_dict:
            self._state = ReActState.deserialize(state_dict)
        else:
            self._state = ReActState()

    def _store_state_to_context(self, context):
        state_dict = self._state.serialize()
        context.state().update({REACT_AGENT_STATE_KEY: state_dict})

    @staticmethod
    def _update_chat_history_in_context(completed_sub_tasks: List[SubTask], context):
        current_messages = ReActControllerUtils.get_dialogue_history_from_context(context)
        tool_messages = [ToolMessage(content=sub_task.result, tool_call_id=sub_task.id) for sub_task in completed_sub_tasks]
        ReActControllerUtils.set_dialogue_history_to_context(current_messages + tool_messages, context)
