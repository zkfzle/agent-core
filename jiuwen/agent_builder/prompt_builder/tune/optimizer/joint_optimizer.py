# -*-coding: utf-8 -*-

"""
prompt optimization evaluators
"""

import random
from typing import List, Optional, Tuple

from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.agent_builder.prompt_builder.tune.base import Case, TuneConstant
from jiuwen.agent_builder.prompt_builder.tune.dataset.case_loader import CaseLoader
from jiuwen.agent_builder.prompt_builder.tune.optimizer.base import BaseOptimizer
from jiuwen.agent_builder.prompt_builder.tune.optimizer.instruction_optimizer import InstructionOptimizer
from jiuwen.agent_builder.prompt_builder.tune.optimizer.example_optimizer import ExampleOptimizer


class JointOptimizer(BaseOptimizer):
    def __init__(
            self,
            model: BaseChatModel,
            model_name: str,
            instruction_optimizer: Optional[InstructionOptimizer] = None,
            example_optimizer: Optional[ExampleOptimizer] = None,
            **kwargs
    ):
        super().__init__(model, model_name)

        if not isinstance(instruction_optimizer, InstructionOptimizer) \
                or not isinstance(example_optimizer, ExampleOptimizer):
            raise JiuWenBaseException(
                StatusCode.AGENT_BUILDER_CREATE_OPTIMIZER_ERROR.code,
                StatusCode.AGENT_BUILDER_CREATE_OPTIMIZER_ERROR.errmsg.format(
                    error_msg=f"type of example optimizer or instruction optimizer is invalid"
                )
            )

        self._num_retires = kwargs.get("num_retires", TuneConstant.DEFAULT_LLM_CALL_RETRY_NUM)

        self._instruction_optimizer = (
            instruction_optimizer
            if instruction_optimizer
            else InstructionOptimizer(model, model_name, num_retires=self._num_retires)
        )

        self._example_optimizer = (
            example_optimizer
            if example_optimizer
            else ExampleOptimizer(model, model_name, num_retires=self._num_retires)
        )

    def optimize(self, original_prompt: str, case_loader: CaseLoader) -> Optional[Tuple[str, List[Case]]]:
        """optimize instruction"""
        need_optimize_example = self._example_optimizer._num_examples > 0
        is_optimize_instruction = random.choice([True, False]) if need_optimize_example else True
        return self._instruction_optimizer.optimize(original_prompt, case_loader) \
            if is_optimize_instruction \
            else self._example_optimizer.optimize(original_prompt, case_loader)

    def pre_optimize(self, original_prompt: str, case_loader: CaseLoader) -> Optional[Tuple[str, List[Case]]]:
        _, examples = self._example_optimizer.pre_optimize(original_prompt, case_loader)
        instruction, _ = self._instruction_optimizer.pre_optimize(original_prompt, case_loader)
        return instruction, examples