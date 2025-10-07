#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from typing import List, Tuple, Dict, Optional

import asyncio

from jiuwen.core.common.logging import logger
from jiuwen.agent_builder.prompt_builder.tune.base import EvaluatedCase, TuneConstant
from jiuwen.core.agent.agent import Agent
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
        async def run_forward_tasks():
            forward_tasks = [agent.invoke(case.inputs) for case in cases.get_cases()]
            return [await task for task in forward_tasks]
        return asyncio.run(run_forward_tasks())
