#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import re
import ast

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.component.condition.condition import Condition
from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.graph.executable import Input, Output


class ExpressionCondition(Condition):
    def __init__(self, expression: str):
        super().__init__()
        self._expression = expression

    def trace_info(self, runtime: BaseRuntime = None):
        return {
            "bool_expression": self._expression,
            "inputs": self._get_inputs(runtime)
        }

    def _get_inputs(self, runtime: BaseRuntime) -> dict:
        if len(self._expression) == 0 or runtime is None:
            return {}
        pattern = r'\$\{[^}]*\}'
        matches = re.findall(pattern, self._expression)
        inputs = {}
        for match in matches:
            inputs[match] = runtime.state().get_global(match[2:-1])
        return inputs

    def invoke(self, inputs: Input, runtime: BaseRuntime) -> Output:
        if len(self._expression) == 0:
            return True
        return self._evaluate_expression(self._expression, self._get_inputs(runtime))

    def _is_empty_replacement(self, match):
        placeholder = match.group(1)
        indices_part  = match.group(2) if match.group(2) else ""
        variable_access = f'${{{placeholder}}}{indices_part}'
        return f'({variable_access} is None) or (len({variable_access}) == 0)'

    def _is_not_empty_replacement(self, match):
        placeholder = match.group(1)
        indices_part  = match.group(2) if match.group(2) else ""
        variable_access = f'${{{placeholder}}}{indices_part}'
        return f'({variable_access} is not None) and (len({variable_access}) > 0)'

    def _evaluate_expression(self, raw_expression, inputs) -> bool:
        expression = raw_expression
        expression = expression.replace("&&", " and ") \
            .replace("||", " or ") \
            .replace("not_in", " not in ") \
            .replace("length", "len")

        expression = re.sub(r'is_empty\(\s*\$\{(.*?)\}\s*((?:\[[^\]]+\])*)\)', self._is_empty_replacement, expression)
        expression = re.sub(r'is_not_empty\(\s*\$\{(.*?)\}\s*((?:\[[^\]]+\])*)\)', self._is_not_empty_replacement,
                            expression)
        expression = re.sub(r'\btrue\b', r'True', expression)
        expression = re.sub(r'\bfalse\b', r'False', expression)

        processed_expression = re.sub(r'\$\{(.*?)\}', lambda match: f'inputs["{match.group(0)}"]', expression)
        
        try:
            parsed_expr = ast.parse(processed_expression, mode='eval')
            
            class SafeExprChecker(ast.NodeVisitor):
                def visit_Call(self, node):
                    # 只允许调用len函数
                    if isinstance(node.func, ast.Name) and node.func.id != 'len':
                        raise ValueError(f"Function calls other than 'len' are not allowed: {node.func.id}")
                    # 不允许任何属性访问（防止如os.system调用）
                    if isinstance(node.func, ast.Attribute):
                        raise ValueError(f"Attribute access is not allowed: {ast.unparse(node.func)}")
                    self.generic_visit(node)
                
                def visit_Name(self, node):
                    # 只允许访问inputs、len和True/False/None
                    if node.id not in ['inputs', 'len', 'True', 'False', 'None']:
                        raise ValueError(f"Variable access not allowed: {node.id}")
                
                def visit_Attribute(self, node):
                    # 不允许任何属性访问
                    raise ValueError(f"Attribute access is not allowed: {ast.unparse(node)}")
                
                def visit_Subscript(self, node):
                    # 只允许对inputs进行下标访问
                    if isinstance(node.value, ast.Name) and node.value.id == 'inputs':
                        self.generic_visit(node)
                    elif isinstance(node.value, ast.Subscript):
                        # 允许链式下标访问，如 inputs["a"][0]
                        self.generic_visit(node)
                    else:
                        raise ValueError(f"Subscript access only allowed for 'inputs': {ast.unparse(node)}")
            
            checker = SafeExprChecker()
            checker.visit(parsed_expr)

            eval_globals = {"__builtins__": {}}
            # 只允许访问inputs和len函数
            eval_locals = {
                'inputs': inputs,
                'len': len
            }
            
            result = eval(compile(parsed_expr, '<string>', 'eval'), eval_globals, eval_locals)
            
            if not isinstance(result, bool):
                raise SyntaxError(f"Expression result must be boolean, got: {type(result).__name__}")
            return result
        except (SyntaxError, TypeError) as e:
            raise JiuWenBaseException(StatusCode.EXPRESSION_CONDITION_SYNTAX_ERROR.code,
                                      StatusCode.EXPRESSION_CONDITION_SYNTAX_ERROR.errmsg.format(
                                          expression=expression,
                                          error_msg=f'{str(e)}, please check the expression inputs')) from e
        except Exception as e:
            raise JiuWenBaseException(StatusCode.EXPRESSION_CONDITION_EVAL_ERROR.code,
                                      StatusCode.EXPRESSION_CONDITION_EVAL_ERROR.errmsg.format(expression=expression,
                                                                                               error_msg=str(e))) from e
