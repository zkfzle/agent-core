#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
"""State of ReActAgent"""
from typing import Optional, List

from pydantic import BaseModel, Field

from jiuwen.agent.common.enum import ReActStatus
from jiuwen.core.agent.task.sub_task import SubTask
from jiuwen.core.utils.llm.messages import AIMessage

class InterruptState(BaseModel):
    interrupt_component_id: str = Field(default="")
    question: str = Field(default="")

    def update(self, interrupt_component_id: str, question: str):
        self.interrupt_component_id = interrupt_component_id
        self.question = question

class ReActState(BaseModel):
    current_iteration: int = Field(default=0)
    status: ReActStatus = Field(default=ReActStatus.INITIALIZED)
    llm_output: Optional[AIMessage] = Field(default=None)
    sub_tasks: List[SubTask] = Field(default_factory=list)
    final_result: str = Field(default="")
    interrupt_state: InterruptState = Field(default=InterruptState())

    @classmethod
    def deserialize(cls, state_dict):
        return cls.model_validate(state_dict)

    def serialize(self) -> dict:
        return self.model_dump()

    def handle_llm_response_event(self, llm_response: AIMessage, sub_tasks: List[SubTask]):
        self.llm_output = llm_response
        self.sub_tasks = sub_tasks
        self.status = ReActStatus.LLM_RESPONSE

    def handle_tool_invoked_event(self, sub_tasks: List[SubTask]):
        self.sub_tasks = sub_tasks
        self.status = ReActStatus.TOOL_INVOKED

    def handle_react_completed_event(self, final_result: str):
        self.final_result = final_result
        self.status = ReActStatus.COMPLETED

    def handle_interrupted_event(self, sub_tasks: List[SubTask]):
        self.sub_tasks = sub_tasks
        self.status = ReActStatus.INTERRUPTED

    def increment_iteration(self):
        self.current_iteration += 1
