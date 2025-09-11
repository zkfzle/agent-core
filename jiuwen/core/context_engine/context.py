#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import Optional, Any, Dict, Union, List

from jiuwen.core.context_engine.base import Context, ContextVariable, ContextOwner
from jiuwen.core.context_engine.accessor.accessor import ContextAccessor
from jiuwen.core.context_engine.execute.executor import ContextExecutor
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.core.utils.llm.messages import BaseMessage


class ContextImpl(Context):
    def __init__(self,
                 owner: ContextOwner,
                 executor: ContextExecutor,
                 accessor: ContextAccessor):
        self._owner = owner
        self._executor: ContextExecutor = executor
        self._accessor: ContextAccessor = accessor

    def add_message(self,
                    message: BaseMessage,
                    tags: Optional[Dict[str, str]] = None):
        history = self._accessor.history()
        history.add_message(message, owner=[self._owner], tags=tags)

    def get_messages(self,
                    num: int = -1,
                    tags: Optional[Dict[str, str]] = None) -> List[BaseMessage]:
        history = self._accessor.history()
        messages = history.get_messages(num, owner=self._owner, tags=tags)
        return messages

    def get_variable(self,
                     name: str) -> Optional[ContextVariable]:
        variables = self._accessor.variables()
        return variables.get(name)

    def set_variable(self,
                     name: str,
                     value: ContextVariable):
        variables = self._accessor.variables()
        return variables.set(name, value)

    def assemble(self,
                 user_input: str,
                 system_prompt: Union[str, Template],
                 variables: Optional[Dict[str, ContextVariable]] = None,
                 **kwargs) -> Union[List[BaseMessage], str]:
        context_window = self._accessor.create_context_window(self._owner)
        context_window.user_input = user_input
        context_window.system_prompt = system_prompt
        context_window.variables.update(variables or {})
        return self._executor.assemble(context_window, kwargs.get("config"))

    def assemble_by_pipeline(self,
                             user_input: str,
                             system_prompt: Union[str, Template],
                             variables: Optional[Dict[str, ContextVariable]] = None,
                             **kwargs) -> Union[List[BaseMessage], str]:
        context_window = self._accessor.create_context_window(self._owner)
        context_window.user_input = user_input
        context_window.system_prompt = system_prompt
        context_window.variables.update(variables or {})
        output = self._executor.run_process_pipeline(context_window)
        return output.full_prompt

    def get_compressed_history(self,
                               config: Optional[Dict[str, Any]] = None,
                               owner: Optional[ContextOwner] = None) -> List[BaseMessage]:
        pass


class AgentContext(ContextImpl):
    def __init__(self,
                 owner: ContextOwner,
                 executor: ContextExecutor,
                 accessor: ContextAccessor):
        super().__init__(owner, executor, accessor)


class WorkflowContext(ContextImpl):
    def __init__(self,
                 owner: ContextOwner,
                 executor: ContextExecutor,
                 accessor: ContextAccessor):
        super().__init__(owner, executor, accessor)
