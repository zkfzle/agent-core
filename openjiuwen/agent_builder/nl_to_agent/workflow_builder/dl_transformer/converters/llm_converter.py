#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
from .base import BaseConverter
from ..models import InputsField
from ..converter_utils import ConverterUtils


class LLMConverter(BaseConverter):
    def _convert_specific_config(self):
        self.node.data.inputs = InputsField(
            inputParameters=self._convert_input_variables(self.node_data["parameters"]["inputs"]),
            llmParam=ConverterUtils.convert_llm_param(
                self.node_data["parameters"]["configs"]["system_prompt"],
                self.node_data["parameters"]["configs"]["user_prompt"]
            )
        )
        self.node.data.outputs = self._convert_outputs_field(self.node_data["parameters"]["outputs"])
