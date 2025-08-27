#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from pydantic import BaseModel, Field
from typing import Dict

from jiuwen.core.context_engine.config import ContextEngineConfig

class ModelContextConfig(BaseModel):
    engine_configs: Dict[str, ContextEngineConfig] = Field(default={})

    def update_engine_config(self, node_id: str, engine_config: ContextEngineConfig):
        self.engine_configs[node_id] = engine_config

    def delete_engine_config(self, node_id: str):
        self.engine_configs.pop(node_id, None)