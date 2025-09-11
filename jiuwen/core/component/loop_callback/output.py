#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Any

from jiuwen.core.component.loop_callback.loop_callback import LoopCallback
from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.runtime.utils import is_ref_path, extract_origin_key, NESTED_PATH_SPLIT
from jiuwen.core.graph.executable import Output


class OutputCallback(LoopCallback):
    def __init__(self, outputs_format: dict[str, Any], round_result_root: str = None,
                 result_root: str = None):
        self._outputs_format = outputs_format
        self._result_root = result_root
        self._round_result_root = round_result_root if round_result_root else "round"
        self._intermediate_loop_var_root = "intermediateLoopVar" + NESTED_PATH_SPLIT

    def _generate_results(self, results: list[(str, Any)]):
        for key, value in self._outputs_format.items():
            if isinstance(value, str) and is_ref_path(value):
                ref_str = extract_origin_key(value)
                results.append((ref_str, None, key))
            elif isinstance(value, dict):
                self._generate_results(results)

    def first_in_loop(self, runtime: BaseRuntime) -> Output:
        _results: list[(str, Any)] = []
        self._generate_results(_results)
        runtime.state().update({self._round_result_root: _results})
        return None

    def out_loop(self, runtime: BaseRuntime) -> Output:
        results: list[(str, Any)] = runtime.state().get(self._round_result_root)
        output = {}
        for result in results:
            output[result[-1]] = result[1]
        runtime.state().update(output)
        return output

    def start_round(self, runtime: BaseRuntime) -> Output:
        return None

    def end_round(self, runtime: BaseRuntime) -> Output:
        results: list[(str, Any)] = runtime.state().get(self._round_result_root)
        if not isinstance(results, list):
            raise RuntimeError("error results in round process")
        for value in results:
            path = value[0]
            if path.startswith(self._intermediate_loop_var_root):
                value[1] = runtime.state().get(path)
            elif isinstance(value, list):
                if value[1] is None:
                    value[1] = []
                value[1].append(runtime.state().get(path))
            else:
                raise RuntimeError("error process in loop: " + path + ", " + str(value))
        runtime.state().update({self._round_result_root: results})
        return None

