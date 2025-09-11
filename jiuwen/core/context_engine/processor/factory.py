#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import Union, Dict, Optional

from jiuwen.core.utils.llm.model_utils.singleton import Singleton
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.common.logging import logger
from jiuwen.core.context_engine.config import BaseProcessorConfig
from jiuwen.core.context_engine.processor.base import BaseContextProcessor


class ProcessorFactory(metaclass=Singleton):
    def __init__(self):
        self.__registered_processors = dict()
        self.__registered_processor_configs = dict()

    @classmethod
    def register(cls, processor_type: str, config_type):
        def register_processor_class(proc_cls):
            cls().__registered_processors[processor_type] = proc_cls
            cls().__registered_processor_configs[processor_type] = config_type
            return proc_cls

        return register_processor_class

    def create_processor(self,
                         config: Union[Dict, BaseProcessorConfig],
                         llm: Optional[BaseChatModel] = None
    ) -> Optional[BaseContextProcessor]:
        processor_type = (
            config.get("processor_type", "")
            if isinstance(config, dict)
            else config.processor_type
        )
        if processor_type not in self.__registered_processors:
            return None
        try:
            if isinstance(config, dict):
                config = self.__registered_processor_configs[processor_type](**config)
            processor = self.__registered_processors[processor_type](config)
            processor.bind_llm(llm)
        except Exception as e:
            logger.error(
                f"cannot create processor type {processor_type}, reason: {str(e)}"
            )
            return None
        return processor
