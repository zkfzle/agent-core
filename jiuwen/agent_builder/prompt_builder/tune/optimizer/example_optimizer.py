# -*- coding: utf-8 -*-
"""
prompt optimization evaluators
"""
import copy
import random
from typing import List, Optional, Tuple

from jiuwen.core.common.logging import logger
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.agent_builder.prompt_builder.tune.base import Case, TuneConstant
from jiuwen.agent_builder.prompt_builder.tune.utils import TuneUtils
from jiuwen.agent_builder.prompt_builder.tune.dataset.case_loader import CaseLoader
from jiuwen.agent_builder.prompt_builder.tune.optimizer.base import BaseOptimizer


EXAMPLE_SELECTION_TEMPLATE = Template(content="""作为提示词优化专家,我的任务是帮助代理高效且成功地完成任务。
当前任务描述:
[任务描述]{{task_description}}
请从以下回答错误的数据或正确但又代表性的示例集合中选择最具代表性的{{num_examples}}个示例,以解决上述任务中的任何问题。
当前的错误示例集是
{{examples}}

选择出最具代表性示例集的标号,用列表形式输出,输出格式为:
```list
[索引1, 索引2,...]
```
例如输出3个示例:
```list
[0, 2, 4]
```
1. 输出的索引列表必须满足{{num_examples}}个
2. 输出必须被'```list```'包裹

[请选择示例]
""")


class ExampleOptimizer(BaseOptimizer):
    def __init__(self, model: BaseChatModel, model_name: str, num_examples: int = TuneConstant.DEFAULT_EXAMPLE_NUM,
                 **kwargs):
        super().__init__(model, model_name)
        self._num_examples = num_examples

    def optimize(self, original_prompt: str, case_loader: CaseLoader) -> Optional[Tuple[str, List[Case]]]:
        """optimize instruction"""
        if self._num_examples <= 0:
            return original_prompt, []
        selected_examples = self._select_best_examples(original_prompt, case_loader)
        return original_prompt, selected_examples

    def pre_optimize(self, original_prompt: str, case_loader: CaseLoader) -> Optional[Tuple[str, List[Case]]]:
        return original_prompt, self._select_examples_before_optimize(case_loader)

    def _select_examples_before_optimize(self, case_loader: CaseLoader):
        """prepare few-shot examples"""
        return self._sample_example(self._num_examples, case_loader)

    def _sample_example(self, num_examples: int, case_loader: CaseLoader) -> Optional[List[Case]]:
        """sample example"""
        dataset = case_loader.get_cases()
        error_cases = case_loader.get_bad_cases()
        if num_examples >= len(dataset):
            return copy.deepcopy(dataset)
        sampled_examples = []
        if error_cases:
            num_error_examples = min(num_examples, len(error_cases))
            sampled_examples.extend(random.sample(error_cases, num_error_examples))

        if len(sampled_examples) < num_examples:
            num_remaining_examples = num_examples - len(sampled_examples)
            remaining_examples = [ex for ex in dataset if ex not in sampled_examples]
            sampled_examples.extend(random.sample(remaining_examples, num_remaining_examples))
        else:
            sampled_examples.extend(random.sample(dataset, num_examples))

        return sampled_examples

    def _select_best_examples(self, original_prompt: str, case_loader: CaseLoader) -> List[Case]:
        """select best examples"""
        if self._num_examples <= 0:
            return []

        num_selected_examples = min(len(case_loader.get_bad_cases()), self._num_examples)
        pre_selected_examples = self._sample_examples_from_cases(case_loader, num_selected_examples)
        examples_string = "\n".join(
            f"index: {i}\n"
            f"question: {TuneUtils.get_input_string_from_case(example)}\n"
            f"assistant answer: {example.label.content}"
            for i, example in enumerate(pre_selected_examples)
        )

        messages = EXAMPLE_SELECTION_TEMPLATE.format(
            dict(task_description=original_prompt, num_examples=num_selected_examples, examples=examples_string)
        ).to_messages()

        try:
            response = self._model.invoke(self._model_name, messages).content
            return self._extract_selected_examples_from_response(response, case_loader.get_bad_cases())

        except Exception as e:
            logger.warning(f"Error occur while selecting best examples: {e}")
            return []

    @staticmethod
    def _sample_examples_from_cases(case_loader: CaseLoader, num_examples: int) -> List[Case]:
        if num_examples >= case_loader.size():
            return copy.deepcopy(case_loader.get_cases())

        error_cases = case_loader.get_bad_cases()
        if len(error_cases) >= num_examples:
            return error_cases

        if len(error_cases) > TuneConstant.DEFAULT_MAX_NUM_SAMPLE_ERROR_CASES:
            error_cases = random.sample(
                error_cases,
                TuneConstant.DEFAULT_MAX_NUM_SAMPLE_ERROR_CASES
            )
        return error_cases

    @staticmethod
    def _extract_selected_examples_from_response(response: str, error_cases: List[Case]) -> List[Case]:
        best_example_list = TuneUtils.parse_list_from_llm_response(response)
        return [error_cases[index] for index in best_example_list]