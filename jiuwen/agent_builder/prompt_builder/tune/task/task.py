#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from abc import ABC, abstractmethod
from typing import Optional, List

from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.utils.llm.messages import BaseMessage, AIMessage, SystemMessage
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.agent_builder.prompt_builder.base import ModelMixin
from jiuwen.agent_builder.prompt_builder.tune.base import Case


class BaseTask(ABC):
    @abstractmethod
    def forward(self, case: Case, **kwargs) -> Optional[AIMessage]:
        pass


class PromptTask(BaseTask, ModelMixin):
    def __init__(self,
                 model: BaseChatModel,
                 model_name: str,
                 prompt: str,
                 **kwargs) -> None:

        super().__init__(model, model_name)
        self._prompt = prompt

    def update_prompt(self, prompt: str):
        self._prompt = prompt

    def get_prompt(self):
        return self._prompt

    def forward(self, case: Case, **kwargs) -> Optional[AIMessage]:
        variables = case.variables
        messages: List[BaseMessage] = [
            SystemMessage(
                content=Template(content=self._prompt).format(variables).content
            )
        ]
        messages.extend(case.messages)
        return self._model.invoke(self._model_name, messages, tools=case.tools)