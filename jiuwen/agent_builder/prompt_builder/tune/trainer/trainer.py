#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import copy

import asyncio

from jiuwen.agent_builder.prompt_builder.tune.utils import TuneUtils
from jiuwen.core.agent.agent import Agent
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.common.logging import logger
from jiuwen.core.utils.llm_call.base import LLMCall
from jiuwen.agent_builder.prompt_builder.tune.base import EvaluatedCase, TuneConstant
from jiuwen.agent_builder.prompt_builder.tune.dataset.case_loader import CaseLoader
from jiuwen.agent_builder.prompt_builder.tune.evaluator.evaluator import BaseEvaluator
from jiuwen.agent_builder.prompt_builder.tune.optimizer.base import BaseOptimizer, TextualParameter


class Trainer:
    def __init__(self,
                 optimizer: BaseOptimizer,
                 evaluator: BaseEvaluator,
                 **kwargs
                 ):
        self._optimizer = optimizer
        self._evaluator = evaluator

        self._num_parallel = kwargs.get("num_parallel", TuneConstant.DEFAULT_PARALLEL_NUM)
        TuneUtils.validate_digital_parameter(self._num_parallel, "num_parallel",
                                             TuneConstant.MIN_PARALLEL_NUM, TuneConstant.MAX_PARALLEL_NUM)
        self._early_stop_score = kwargs.get("early_stop_score", TuneConstant.DEFAULT_EARLY_STOP_SCORE)
        TuneUtils.validate_digital_parameter(self._early_stop_score, "num_parallel",
                                             0.0, 1.0)

    def train(self,
              agent: Agent,
              train_cases: CaseLoader,
              **kwargs
              ) -> Optional[Agent]:
        from jiuwen.agent.chat_agent import ChatAgent
        if not isinstance(agent, ChatAgent):
            raise JiuWenBaseException(
                StatusCode.AGENT_BUILDER_AGENT_TRAINER_TRAIN_ERROR.code,
                StatusCode.AGENT_BUILDER_AGENT_TRAINER_TRAIN_ERROR.errmsg.format(
                    error_msg=f"trainer only support {type(ChatAgent)} right now"
                )
            )
        best_score = 0.0
        num_iterations = kwargs.get('num_iterations', TuneConstant.DEFAULT_ITERATION_NUM)
        score, evaluated_cases = self.evaluate(agent, train_cases)
        logger.info(f"train iteration: (baseline), score: {score}")
        self._pre_train(agent)
        for i in range(1, num_iterations + 1):
            self._optimizer.backward(evaluated_cases)
            self._optimizer.update()
            cur_parameters = copy.deepcopy(agent.get_llm_calls())
            self._update_agent(agent, self._optimizer.parameters())
            score, evaluated_cases = self.evaluate(agent, train_cases)
            self._update_agent(agent, cur_parameters)
            logger.info(f"train iteration: {i}, score: {score}")
            if score > best_score:
                best_score = score
                self._update_agent(agent, self._optimizer.parameters())
            if best_score >= self._early_stop_score:
                break
        return agent

    def evaluate(self,
                 agent: Agent,
                 cases: CaseLoader,
                 ) -> Tuple[float, List[EvaluatedCase]]:
        if not cases.get_cases():
            return 0.0, []
        predicts = self.predict(agent, cases)
        evaluated_cases = self._evaluator.batch_evaluate(cases.get_cases(), predicts)
        score = sum(case.score for case in evaluated_cases) / len(evaluated_cases) \
            if evaluated_cases else 0.0
        return score, evaluated_cases

    def predict(self,
                agent: Agent,
                cases: CaseLoader
                ) -> List[Dict]:

        num_workers = min(self._num_parallel, cases.size())
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            predicts = executor.map(
                asyncio.run,
                [agent.invoke(case.inputs) for case in cases.get_cases()]
            )
            return list(predicts)

    def _pre_train(self, agent: Agent):
        parameters = copy.deepcopy(agent.get_llm_calls())
        self._optimizer.bind_parameter(parameters)

    def _update_agent(self, agent: Agent, parameters: Dict[str, TextualParameter | LLMCall]):
        agent_parameters = agent.get_llm_calls()
        for name, llm_call in agent_parameters.items():
            param = parameters.get(name)
            if not param:
                continue
            if isinstance(param, TextualParameter):
                llm_call.update_system_prompt(param.llm_call.get_system_prompt().content)
                llm_call.update_user_prompt(param.llm_call.get_user_prompt().content)
            elif isinstance(param, LLMCall):
                llm_call.update_system_prompt(param.get_system_prompt().content)
                llm_call.update_user_prompt(param.get_user_prompt().content)