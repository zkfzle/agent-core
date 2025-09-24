#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import List, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

from jiuwen.agent_builder.prompt_builder.tune.utils import TuneUtils
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.common.logging import logger
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.agent_builder.prompt_builder.tune.base import Case, EvaluatedCase, TuneConstant, PromptOptimizeProgress, \
    PromptOptimizeHistory
from jiuwen.agent_builder.prompt_builder.tune.dataset.case_loader import CaseLoader
from jiuwen.agent_builder.prompt_builder.tune.evaluator.evaluator import BaseEvaluator
from jiuwen.agent_builder.prompt_builder.tune.optimizer.base import BaseOptimizer
from jiuwen.agent_builder.prompt_builder.tune.task.task import PromptTask


class PromptTrainer:
    def __init__(self,
                 task: PromptTask,
                 optimizer: BaseOptimizer,
                 evaluator: BaseEvaluator,
                 case_loader: CaseLoader,
                 **kwargs
                 ):
        self._task = task
        self._optimizer = optimizer
        self._evaluator = evaluator
        self._case_loader = case_loader
        self._variable = []
        self._num_parallel = kwargs.get("num_parallel", TuneConstant.DEFAULT_NUM_PARALLEL)
        self._early_stop_accuracy = kwargs.get("early_stop_accuracy", TuneConstant.DEFAULT_EARLY_STOP_ACCURACY)

    def train(self,
              num_iterations: int = TuneConstant.DEFAULT_ITERATION_NUM,
              **kwargs
              ) -> PromptOptimizeProgress:
        progress_data = PromptOptimizeProgress(
            instruction=self._task.get_prompt(),
            best_prompt=self._task.get_prompt(),
            status=TuneConstant.TASK_RUNNING,
        )
        try:
            self._evaluate_original_prompt(progress_data)
            self._preprocess_before_train(progress_data)
            self._train_iteratively(progress_data, num_iterations)
        except Exception as e:
            progress_data.status = PromptOptimizeProgress.STATUS_FAILED
            progress_data.exception_info = f"Training task failed, reason: {str(e)}"
        return progress_data

    def evaluate(self, cases: List[Case]) -> Tuple[float, List[EvaluatedCase]]:
        return self._batch_forward_and_evaluate(self._task.get_prompt(), cases)

    @staticmethod
    def _get_variables_from_prompt(prompt: str, case_loader: CaseLoader):
        variables_in_prompt = [(prompt.find(f"{{{{{v}}}}}"), v) for v in case_loader.get_variable_keys()]
        return [v[1] for v in sorted(variables_in_prompt, key=lambda x: x[0], reverse=False)]

    def _preprocess_before_train(self, progress_data: PromptOptimizeProgress):
        self._variable = self._get_variables_from_prompt(progress_data.instruction, self._case_loader)
        progress_data.instruction = Template(
            content=progress_data.instruction
        ).format(dict([(v, v) for v in self._variable])).content
        instruction, examples = self._optimizer.pre_optimize(progress_data.instruction, self._case_loader)
        progress_data.instruction = instruction
        progress_data.examples = examples

    def _evaluate_original_prompt(self, progress_data: PromptOptimizeProgress):
        accuracy, evaluated_cases = self._batch_forward_and_evaluate(
            self._task.get_prompt(),
            self._case_loader.get_cases()
        )
        progress_data.best_accuracy = accuracy
        self._case_loader.update_bad_cases([case for case in evaluated_cases if case.score == 0])
        progress_data.history.append(
            PromptOptimizeHistory(
                optimized_prompt=self._task.get_prompt(),
                accuracy=accuracy,
                iteration_round=0
            )
        )

    def _train_iteratively(self,
                           progress_data: PromptOptimizeProgress,
                           num_iteration: int,
                           start_iteration: int = 0):
        if start_iteration >= num_iteration:
            return
        for iter in range(start_iteration, num_iteration):
            progress_data.iteration_round = iter
            logger.info(f"Start training task at iteration {iter}")
            optimized_instruction, optimized_examples = self._optimizer.optimize(
                progress_data.instruction, self._case_loader
            )
            if not optimized_instruction:
                raise JiuWenBaseException(
                    StatusCode.AGENT_BUILDER_PROMPT_OPTIMIZE_ERROR.code,
                    StatusCode.AGENT_BUILDER_PROMPT_OPTIMIZE_ERROR.errmsg.format(
                        error_msg="Optimize failed."
                    )
                )
            full_prompt = self._assemble_prompt_with_examples(optimized_instruction, optimized_examples)
            accuracy, evaluated_cases = self._batch_forward_and_evaluate(
                full_prompt,
                self._case_loader.get_cases()
            )
            if accuracy > progress_data.best_accuracy:
                progress_data.best_accuracy = accuracy
                progress_data.best_prompt = full_prompt
                progress_data.instruction = optimized_instruction
                progress_data.examples = optimized_examples or progress_data.examples
                self._case_loader.update_bad_cases([
                    case for case in evaluated_cases if case.score == 0
                ])
            progress_data.history.append(
                PromptOptimizeHistory(
                    optimized_prompt=full_prompt,
                    accuracy=accuracy,
                    iteration_round=iter
                )
            )
            if not self._chack_continue_each_iteration(progress_data):
                break
        progress_data.status = TuneConstant.TASK_FINISHED
        logger.info(f"Finished training task at iteration {iter}")

    def _chack_continue_each_iteration(self, progress_data: PromptOptimizeProgress) -> bool:
        if progress_data.best_accuracy >= self._early_stop_accuracy:
            return False
        return True

    def _batch_forward_and_evaluate(self, prompt: str, cases: List[Case]) -> Tuple[float, List[EvaluatedCase]]:
        last_prompt = self._task.get_prompt()
        self._task.update_prompt(prompt)
        evaluated_cases = []
        num_workers = min(self._num_parallel, len(cases))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self._forward_and_evaluate, case)
                for case in cases
            ]
            for future in as_completed(futures):
                try:
                    eval_case = future.result()
                    evaluated_cases.append(eval_case)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    logger.error(f"Evaluation failed, reason: {str(e)}")
            if not evaluated_cases:
                raise JiuWenBaseException(
                    StatusCode.AGENT_BUILDER_PROMPT_OPTIMIZE_CASE_VALIDATION_ERROR.code,
                    StatusCode.AGENT_BUILDER_PROMPT_OPTIMIZE_CASE_VALIDATION_ERROR.errmsg.format(
                        error_msg="Get empty evaluation result"
                    )
                )
            total_score = sum([case.score or 0 for case in evaluated_cases])
            accuracy = total_score / len(evaluated_cases)
            self._task.update_prompt(last_prompt)
            return accuracy, evaluated_cases

    def _forward_and_evaluate(self, case: Case) -> EvaluatedCase:
        predict = self._task.forward(case)
        score, reason = self._evaluator.evaluate(case, predict)
        evaluated_case = EvaluatedCase(**case.model_dump())
        evaluated_case.answer = predict
        evaluated_case.score = score
        evaluated_case.reason = reason
        return evaluated_case

    def _assemble_prompt_with_examples(self, instruction: str, examples: List[Case]):
        variable_section = "\n".join([f"【{key}】: {{{{{key}}}}}" for key in self._variable])
        examples_section = TuneUtils.convert_cases_to_examples(examples)
        return "".join([
            instruction,
            f"\n## 示例\n{examples_section}" if examples_section else "",
            f"\n\n用户输入:\n{variable_section}" if variable_section else ""
        ])