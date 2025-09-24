#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from abc import abstractmethod
from typing import List, Optional, Union, Tuple

from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.agent_builder.prompt_builder.base import ModelMixin
from jiuwen.agent_builder.prompt_builder.tune.base import Case
from jiuwen.agent_builder.prompt_builder.tune.dataset.case_loader import CaseLoader


class BaseOptimizer(ModelMixin):
    def __init__(self,
                 model: BaseChatModel,
                 model_name: str,
                 **kwargs
                 ):
        super().__init__(model, model_name)

    @abstractmethod
    def optimize(self,
                 original_prompt: str,
                 case_loader: CaseLoader
                 ) -> Optional[Tuple[str, List[Case]]]:
        pass

    def pre_optimize(self,
                     original_prompt: str,
                     case_loader: CaseLoader
                     ) -> Optional[Tuple[str, List[Case]]]:
        return original_prompt, []