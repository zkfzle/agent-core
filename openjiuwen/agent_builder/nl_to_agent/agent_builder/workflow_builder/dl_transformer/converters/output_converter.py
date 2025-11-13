#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
from .base import BaseConverter
from ..models import InputsField


class OutputConverter(BaseConverter):
    def _convert_specific_config(self):
        self.node.data.inputs = InputsField(
            inputParameters=self._convert_input_variables(self.node_data["parameters"]["inputs"]),
            content=dict(type="template", content=self.node_data["parameters"]["configs"]["template"])
        )
