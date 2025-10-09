# -*-coding: utf-8 -*-

"""
prompt optimization evaluators
"""

import random
from typing import List, Optional

from jiuwen.core.agent.agent import Agent
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.agent_builder.prompt_builder.tune.base import TuneConstant, EvaluatedCase
from jiuwen.agent_builder.prompt_builder.tune.optimizer.base import BaseOptimizer
from jiuwen.agent_builder.prompt_builder.tune.optimizer.instruction_optimizer import InstructionOptimizer
from jiuwen.agent_builder.prompt_builder.tune.optimizer.example_optimizer import ExampleOptimizer


class JointOptimizer(BaseOptimizer):
    def __init__(
            self,
            agent: Agent,
            model: BaseChatModel,
            model_name: str,
            num_examples: int = TuneConstant.DEFAULT_EXAMPLE_NUM,
            **kwargs
    ):
        super().__init__(agent)
        self._model = model,
        self._model_name = model_name,
        self._instruction_optimizer = InstructionOptimizer(agent, model, model_name)
        self._example_optimizer = ExampleOptimizer(agent, model, model_name, num_examples)

    def _backward(self,
                 evaluated_cases: List[EvaluatedCase] ,
                 ) -> Optional[Agent]:
        need_optimize_example = self._example_optimizer._num_examples > 0
        is_optimize_instruction = random.choice([True, False]) if need_optimize_example else True
        self._example_optimizer.init_examples(evaluated_cases)
        for name, param in self._parameters.items():
            if is_optimize_instruction:
                self._instruction_optimizer.backward(evaluated_cases)
                backward_params = self._instruction_optimizer.parameters()
            else:
                self._example_optimizer.backward(evaluated_cases)
                backward_params = self._example_optimizer.parameters()
            param.llm_call = backward_params.get(name).llm_call
            param.set_gradient("system_prompt", backward_params.get(name).get_gradient("system_prompt"))

    def _update(self) -> Optional[Agent]:
        optimized_agent = self._instruction_optimizer._update()
        llm_calls = optimized_agent.get_llm_calls()
        for name, param in self._parameters.items():
            optimized_prompt = self._example_optimizer._format_prompt(
                llm_calls.get(name).get_system_prompt(),
                self._example_optimizer.parameters().get(name, {}).get_gradient("system_prompt")
            )
            llm_calls.get(name).update_system_prompt(optimized_prompt)
        return optimized_agent
