#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from abc import ABC, abstractmethod
from typing import Tuple

from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.utils.llm.messages import BaseMessage
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.agent_builder.prompt_builder.base import ModelMixin
from jiuwen.agent_builder.prompt_builder.tune.base import Case
from jiuwen.agent_builder.prompt_builder.tune.utils import TuneUtils


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self,
                 case: Case,
                 predict: BaseMessage) -> Tuple[int, str]:
        pass


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


class DefaultEvaluator(BaseEvaluator, ModelMixin):
    def __init__(self,
                 model: BaseChatModel,
                 model_name: str,
                 metric: str = "",
                 **kwargs
                 ):
        super().__init__(model, model_name)
        self._metric_template = LLM_METRIC_TEMPLATE.format(
            dict(user_metrics=metric)
        )

    def evaluate(self,
                 case: Case,
                 predict: BaseMessage) -> Tuple[int, str]:
        messages = self._metric_template.format(
            dict(
                question=TuneUtils.get_input_string_from_case(case),
                expected_answer=TuneUtils.get_output_string_from_message(case.label),
                model_answer=TuneUtils.get_output_string_from_message(predict),
            ),
        ).to_messages()
        try:
            response = self._model.invoke(self._model_name, messages).content
        except JiuWenBaseException:
            return 0, "Failed to evaluate case due to model error"

        evaluated_result = TuneUtils.parse_json_from_llm_response(response)
        if not evaluated_result:
            return 0, "Failed to evaluate case due to parsing error"
        result = evaluated_result.get("result", False)
        reason = evaluated_result.get("reason", "")
        if result is True or (isinstance(result, str) and result.strip().lower() == "true"):
            return 1, reason
        return 0, reason