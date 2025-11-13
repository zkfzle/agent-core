# -*-coding: utf-8 -*-

"""
prompt optimization evaluators
"""

import random
import copy
from typing import List, Dict, Optional

from openjiuwen.core.utils.llm_call.base import LLMCall
from openjiuwen.core.utils.llm.base import BaseChatModel
from openjiuwen.agent_builder.prompt_builder.tune.base import TuneConstant, EvaluatedCase
from openjiuwen.agent_builder.prompt_builder.tune.optimizer.base import BaseOptimizer
from openjiuwen.agent_builder.prompt_builder.tune.optimizer.instruction_optimizer import InstructionOptimizer
from openjiuwen.agent_builder.prompt_builder.tune.optimizer.example_optimizer import ExampleOptimizer


class JointOptimizer(BaseOptimizer):
    def __init__(
            self,
            model: BaseChatModel,
            model_name: str,
            parameters: Optional[Dict[str, LLMCall]] = None,
            num_examples: int = TuneConstant.DEFAULT_EXAMPLE_NUM,
            ):
        self._instruction_optimizer = InstructionOptimizer(
            model, model_name, copy.deepcopy(parameters)
        )
        self._example_optimizer = ExampleOptimizer(
            model, model_name, copy.deepcopy(parameters), num_examples
        )
        super().__init__(parameters)
        self._model = model,
        self._model_name = model_name,
        self._is_optimize_instruction: bool = True

    def bind_parameter(self, parameters: Dict[str, LLMCall]):
        super().bind_parameter(parameters)
        self._example_optimizer.bind_parameter(copy.deepcopy(parameters))
        self._instruction_optimizer.bind_parameter(copy.deepcopy(parameters))

    def _backward(self,
                 evaluated_cases: List[EvaluatedCase] ,
                 ):
        self._example_optimizer.init_examples(evaluated_cases)
        self._select_optimize_strategy()
        for name, param in self._parameters.items():
            if self._is_optimize_instruction:
                self._instruction_optimizer.backward(evaluated_cases)
                backward_params = self._instruction_optimizer.parameters()
            else:
                self._example_optimizer.backward(evaluated_cases)
                backward_params = self._example_optimizer.parameters()
            param.set_gradient("system_prompt", backward_params.get(name).get_gradient("system_prompt"))
            param.set_gradient("user_prompt", backward_params.get(name).get_gradient("user_prompt"))

    def _update(self):
        if self._is_optimize_instruction:
            self._instruction_optimizer._update()
        instr_parameters = self._instruction_optimizer.parameters()
        for name, param in self._parameters.items():
            if not param.llm_call.get_freeze_user_prompt():
                optimized_prompt = self._example_optimizer._format_prompt(
                    instr_parameters.get(name).llm_call.get_user_prompt(),
                    self._example_optimizer.parameters().get(name, {}).get_gradient("user_prompt")
                )
                param.llm_call.update_user_prompt(optimized_prompt)
            elif not param.llm_call.get_freeze_system_prompt():
                optimized_prompt = self._example_optimizer._format_prompt(
                    instr_parameters.get(name).llm_call.get_system_prompt(),
                    self._example_optimizer.parameters().get(name, {}).get_gradient("system_prompt")
                )
                param.llm_call.update_system_prompt(optimized_prompt)

    def _select_optimize_strategy(self):
        need_optimize_example = self._example_optimizer._num_examples > 0
        self._is_optimize_instruction = random.choice([True, False]) if need_optimize_example else True
