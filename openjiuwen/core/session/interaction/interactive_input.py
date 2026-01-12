# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

from openjiuwen.core.common.exception.errors import build_error
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.exception.exception import JiuWenBaseException

_sentinel = object()


class InteractiveInput(BaseModel):
    # key is id of interaction, value is input for the id
    user_inputs: Dict[str, Any] = Field(default_factory=dict)

    # input not bind to any id, used for the first interaction
    raw_inputs: Any = Field(default=None)

    def __init__(self, raw_inputs: Any = _sentinel):
        super().__init__(**{})
        if raw_inputs is None:
            raise build_error(
                StatusCode.WORKFLOW_INPUT_INVALID
            )
        if raw_inputs is _sentinel:
            self.raw_inputs = None
            return
        self.raw_inputs = raw_inputs

    def update(self, node_id: str, value: Any):
        if self.raw_inputs is not None:
            raise build_error(StatusCode.WORKFLOW_STATE_RUNTIME_ERROR)
        if node_id is None or value is None:
            raise build_error(
                StatusCode.WORKFLOW_INPUT_INVALID
            )
        self.user_inputs[node_id] = value
