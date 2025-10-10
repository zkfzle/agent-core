#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import asyncio

from jiuwen.agent_builder.prompt_builder.tune.utils import TuneUtils
from jiuwen.core.agent.agent import Agent
from jiuwen.core.common.logging import logger
from jiuwen.agent_builder.prompt_builder.tune.base import EvaluatedCase, TuneConstant
from jiuwen.agent_builder.prompt_builder.tune.dataset.case_loader import CaseLoader
from jiuwen.agent_builder.prompt_builder.tune.evaluator.evaluator import BaseEvaluator
from jiuwen.agent_builder.prompt_builder.tune.optimizer.base import BaseOptimizer


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
        self._num_parallel = kwargs.get("early_stop_score", TuneConstant.DEFAULT_EARLY_STOP_SCORE)
        TuneUtils.validate_digital_parameter(self._num_parallel, "num_parallel",
                                             0.0, 1.0)

    def train(self,
              agent: Agent,
              train_cases: CaseLoader,
              **kwargs
              ) -> Optional[Agent]:
        cur_agent = agent
        best_agent = agent
        best_score = 0.0
        num_iterations = kwargs.get('num_iterations', TuneConstant.DEFAULT_ITERATION_NUM)
        score, evaluated_cases = self.evaluate(cur_agent, train_cases)
        logger.info(f"train iteration: (baseline), score: {score}")
        for i in range(1, num_iterations + 1):
            self._optimizer.backward(evaluated_cases)
            cur_agent = self._optimizer.update()
            score, evaluated_cases = self.evaluate(cur_agent, train_cases)
            logger.info(f"train iteration: {i}, score: {score}")
            if score > best_score:
                best_score = score
                best_agent = cur_agent
        return best_agent

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
