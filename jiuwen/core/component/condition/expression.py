#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import re

from jiuwen.core.component.condition.condition import Condition
from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.graph.executable import Input, Output


class ExpressionCondition(Condition):
    def __init__(self, expression: str):
        super().__init__()
        self._expression = expression

    def invoke(self, inputs: Input, context: BaseRuntime) -> Output:
        pattern = r'\$\{[^}]*\}'
        matches = re.findall(pattern, self._expression)
        inputs = {}
        for match in matches:
            inputs[match] = context.state().get_global(match[2:-1])
        return self._evaluate_expression(self._expression, inputs)

    def _evaluate_expression(self, expression, inputs) -> bool:
        expression = expression.replace("&&", " and ") \
            .replace("||", " or ") \
            .replace("not_in", " not in ") \
            .replace("length", "len")
        expression = re.sub(r'is_empty\(\s*\$\{(.*?)\}\s*\)', r'len(${\1}) == 0', expression)
        expression = re.sub(r'is_not_empty\(\s*\$\{(.*?)\}\s*\)', r'len(${\1}) > 0', expression)
        expression = re.sub(r'\btrue\b', r'True', expression)
        expression = re.sub(r'\bfalse\b', r'False', expression)

        expression = re.sub(r'\$\{(.*?)\}', lambda match: f'inputs["{match.group(0)}"]', expression)

        context = {
            "inputs": inputs
        }
        try:
            return eval(expression, context)
        except SyntaxError as e:
            raise e
        except Exception as e:
            raise e
