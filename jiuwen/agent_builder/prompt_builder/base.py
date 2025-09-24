#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import Optional

from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode


class ModelMixin:
    def __init__(self,
                 model: BaseChatModel,
                 model_name: str
                 ):
        if not model or not model_name:
            raise JiuWenBaseException(
                error_code=StatusCode.AGENT_BUILDER_LLM_CONFIG_MISS_ERROR.code,
                message=StatusCode.AGENT_BUILDER_LLM_CONFIG_MISS_ERROR.errmsg.format(
                    error_msg=f"get empty model or model name: {model_name}"
                )
            )

        self._model: BaseChatModel = model
        self._model_name: str = model_name
