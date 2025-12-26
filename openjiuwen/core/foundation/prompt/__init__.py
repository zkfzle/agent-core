#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from openjiuwen.core.foundation.prompt.template import PromptTemplate, PromptTemplateCard

from openjiuwen.core.foundation.prompt.assemble.assembler import PromptAssembler


from openjiuwen.core.foundation.prompt.assemble.variables.variable import Variable

from openjiuwen.core.foundation.prompt.assemble.variables.textable import TextableVariable


_PROMPT_TEMPLATE_CLASSES = [
    "PromptTemplate",
    "PromptTemplateCard",
]

_PROMPT_ASSEMBLER_CLASSES = [
    "PromptAssembler",
]

_PROMPT_VARIABLE_CLASSES = [
    "Variable",
    "TextableVariable",
]

__all__ = (
        _PROMPT_TEMPLATE_CLASSES +
        _PROMPT_ASSEMBLER_CLASSES +
        _PROMPT_VARIABLE_CLASSES
)