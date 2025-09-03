#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from jiuwen.core.context.model_context.base import ContextVariable
from jiuwen.core.context_engine.config import ContextEngineConfig


class ModelContextConfig(BaseModel):
    """conversation history config"""
    conversation_history_length: int = Field(default=20, ge=0)
    """variable config"""
    variables: List[ContextVariable] = Field(default=[])
    """context engine config"""
    engine_config: Optional[ContextEngineConfig] = Field(default=None)
    node_configs: Dict[str, ContextEngineConfig] = Field(default={})
    """memory config"""
    enable_memory: bool = Field(default=False)
    memory_config: Dict[str, Any] = Field(default={})

    def update_engine_config(self, engine_config: ContextEngineConfig):
        self.engine_config = engine_config

    def get_node_engine_config(self, node_id: str) -> ContextEngineConfig:
        return self.engine_config.get(node_id)