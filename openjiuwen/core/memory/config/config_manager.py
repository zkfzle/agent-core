#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from threading import Lock
from typing import Any, Dict, Optional
from pydantic import BaseModel

from .config import Config, MemoryConfig


class ConfigManger:
    def __init__(self, default_config: Config):
        """Initialize the Full Manager and receive default parameters"""
        self._default_config = default_config  # Default parameters(lowest priority)
        self._app_config: dict[str, MemoryConfig] = {}  # Application-level parameters(medium priority)
        self._applock = Lock()
        self._allowed_fields = set(self._default_config.__class__.model_fields.keys())

    def update_nested_model(self, model: BaseModel, updates: Dict[str, Any]) -> Config:
        """
        Recursively update nested Pydantic models
        """
        for key, value in updates.items():
            if key in model.model_fields:
                field = model.model_fields[key]
                if field.default_factory is not None and \
                issubclass(field.default_factory, BaseModel) and isinstance(value, dict):
                    nested_model = getattr(model, key)
                    setattr(model, key, self.update_nested_model(nested_model, value))
                else:
                    setattr(model, key, value)
        return model

    def set_app_config(self, app_id: str, config: MemoryConfig) -> None:
        """Set application-level parameters (will override default parameters)"""
        with self._applock:
            if app_id not in self._app_config:
                # 如果应用ID不存在，直接创建新配置
                self._app_config[app_id] = MemoryConfig()
            # 1. 处理 mem_variables（合并字典，新配置优先级更高）
            existing_mem_vars = self._app_config[app_id].mem_variables
            new_mem_vars = config.mem_variables
            if new_mem_vars:  # 只有当新配置包含 mem_variables 时才合并
                existing_mem_vars.update(new_mem_vars)
            # 2. 处理 enable_long_term_mem（只有当新配置不是默认值时才更新）
            if config.enable_long_term_mem is not MemoryConfig().enable_long_term_mem:
                self._app_config[app_id].enable_long_term_mem = config.enable_long_term_mem

    def get_param(self,
                  param_name: str,
                  app_id: Optional[str] = None,
                  request_config: Optional[dict[str, Any]] = None,
                  default: Any = None):
        """Obtain the effective value of a single parameter (by priority)"""
        # 1. First check the request-level parameters
        if request_config and param_name in request_config:
            return request_config[param_name]

        # 2. If the parameters name is not in the parameter list. return the specified default value
        if param_name not in self._allowed_fields:
            return default

        # 3. Check the application-level parameters
        with self._applock:
            if app_id and app_id in self._app_config:
                if param_name in self._app_config[app_id].__class__.model_fields:
                    return getattr(self._app_config[app_id], param_name)

        # 4. Check the defaule parameters
        if param_name in self._default_config.__class__.model_fields:
            return getattr(self._default_config, param_name)

        # 5. Return the specified default value when no parameters exist at all levels
        return default

    def get_config(self, app_id: Optional[str] = None, request_config: Optional[Dict[str, Any]] = None) -> Config:
        """
        Obtain the merged complete configuration (for batch acquisition scenarios)
        priority: Request-level > Application-level > Default-level
        """
        if not app_id or app_id not in self._app_config:
            config = self._default_config
        else:
            with self._applock:
                config = self._app_config.get(app_id)

        # Override the previous parameters with request-level parameters
        if request_config:
            merged = config.model_copy()
            merged = self.update_nested_model(merged, request_config)
            return merged
        return config
