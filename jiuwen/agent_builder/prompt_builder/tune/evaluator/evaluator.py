#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from abc import ABC, abstractmethod
from typing import Dict, Any, List

from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.agent_builder.prompt_builder.tune.base import Case, EvaluatedCase
from jiuwen.agent_builder.prompt_builder.tune.utils import TuneUtils


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self,
                 case: Case,
                 predict: Dict[str, Any]
                 ) -> EvaluatedCase:
        pass

    def batch_evaluate(self,
                       cases: List[Case],
                       predicts: List[Dict[str, Any]]
                       ) -> List[EvaluatedCase]:
        if len(cases) != len(predicts):
            raise JiuWenBaseException(
                StatusCode.AGENT_BUILDER_AGENT_EVALUATOR_EVALUATE_ERROR.code,
                StatusCode.AGENT_BUILDER_AGENT_EVALUATOR_EVALUATE_ERROR.errmsg.format(
                    error_msg=f"length of cases: {len(cases)} dose not equal with length of predicts: {len(predicts)} "
                )
            )
        evaluated_cases = []
        for case, predict in zip(cases, predicts):
            evaluated_cases.append(self.evaluate(case, predict))
        return evaluated_cases


LLM_METRIC_TEMPLATE = Template(content=
"""
你是一个答案校验专家，负责校验给定的模型回答和标准答案之间的含义和结论一致性。请根据以下标准判断模型回答是否与标准答案的含义和结论一致。

- 如果模型回答和标准答案含义一致，返回`true`。
- 如果模型回答和标准答案含义不一致，返回`false`。
- 注意区分对话和工具调用，两者通常不能按语意判断为一致
- 结合用户问题和标准答案，简要分析模型回答和标准答案不一致的理由

以下是用户补充的自定义校验规则，如果与上述规则冲突，则优先遵从用户自定义规则，请严格遵守：
{{user_metrics}}

输出JSON格式：
```json
{{
“result”: true/false,
"reason": "校验理由"
}}
```

[问题]：{question}

以下是需要比对的模型回答和标准答案：
[标准答案]：{{expected_answer}}

[模型回答]：{{model_answer}}

请校验并返回结果：
"""
)


class DefaultEvaluator(BaseEvaluator):
    def __init__(self,
                 model: BaseChatModel,
                 model_name: str,
                 metric: str = "",
                 ):
        super().__init__()
        self._model = model
        self._model_name = model_name
        self._metric_template = LLM_METRIC_TEMPLATE.format(
            dict(user_metrics=metric)
        )

    def evaluate(self,
                 case: Case,
                 predict: Dict[str, Any]
                 ) -> EvaluatedCase:
        messages = self._metric_template.format(
            dict(
                question=str(case.inputs),
                expected_answer=str(case.label),
                model_answer=str(predict),
            ),
        ).to_messages()
        evaluated_case = EvaluatedCase(case=case, answer=predict)
        try:
            response = self._model.invoke(self._model_name, messages).content
        except JiuWenBaseException:
            evaluated_case.reason = "Failed to evaluate case due to model error"
            return evaluated_case

        evaluated_result = TuneUtils.parse_json_from_llm_response(response)
        if not evaluated_result:
            evaluated_case.reason = "Failed to evaluate case due to parsing error"
            return evaluated_case
        result = evaluated_result.get("result", False)
        evaluated_case.reason = evaluated_result.get("reason", "")
        if result is True or (isinstance(result, str) and result.strip().lower() == "true"):
            evaluated_case.score = 1
            return evaluated_case
        return evaluated_case