#!/usr/bin/python3.11
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

import importlib
import logging
import os
from typing import Dict, Type

from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.utils.llm.model_utils.singleton import Singleton


class ModelFactory(metaclass=Singleton):

    def __init__(self):
        self.model_map: Dict[str, Type[BaseChatModel]] = {}
        self._initialize_models()

    def _initialize_models(self):
        core_model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'model_library'
        )
        self._load_model_dir(core_model_dir)
        custom_model_dir = os.getenv('MODEL_DIR')
        if custom_model_dir and os.path.exists(custom_model_dir):
            self._load_model_dir(custom_model_dir)

    @staticmethod
    def _load_models(model_dir: str) -> Dict[str, Type[BaseChatModel]]:
        model_dict = {}
        if not os.path.exists(model_dir):
            logging.warning(f"Model directory not found: {model_dir}")
            return model_dict

        try:
            py_files = [
                f for f in os.listdir(model_dir)
                if (f.endswith('.py') or f.endswith('.pyc')) and f != "__init__.py" and f != "__init__.pyc"
            ]
            for py_file in py_files:
                module_name = os.path.splitext(py_file)[0]
                module_path = os.path.join(model_dir, py_file)

                try:
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    if spec is None:
                        continue
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for name, obj in module.__dict__.items():
                        if (isinstance(obj, type) and issubclass(obj, BaseChatModel) and obj != BaseChatModel):
                            model_dict[module_name] = obj
                            logging.info(f"Loaded model: {module_name} -> {obj.__name__}")
                except Exception as e:
                    logging.error(f"Error loading module {py_file}: {str(e)}")
                    continue
        except Exception as e:
            raise Exception(f"module load error: {str(e)}")
        return model_dict

    def _load_model_dir(self, model_dir: str):
        model_dict = self._load_models(model_dir)
        self.model_map.update(model_dict)

    def get_model(self, model_provider: str, api_key: str, api_base: str,
                  max_retrie: int=3, timeout: int=60) -> BaseChatModel:
        model_cls = self.model_map.get(model_provider.lower())
        if not model_cls:
            available_models = ", ".join(self.model_map.keys())
            raise ValueError(f"Unavailable model provider: {model_provider}. Available models: {available_models}")
        return model_cls(api_key=api_key, api_base=api_base, max_retrie=max_retrie, timeout=timeout)