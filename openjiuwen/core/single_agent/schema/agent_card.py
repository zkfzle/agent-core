# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from dataclasses import field
from typing import List

from openjiuwen.core.common import Param, BaseCard


class AgentCard(BaseCard):
    """Dataclass of Agent Card
    """
    input_params: List[Param] = field(default_factory=list)
    output_params: List[Param] = field(default_factory=list)
