#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from typing import Optional, List, Union

from jiuwen.core.context_engine.accessor.accessor import ContextAccessor
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.utils.llm.messages import BaseMessage
from jiuwen.core.context_engine.base import ContextWindow
from jiuwen.core.context_engine.config import ContextEngineConfig
from jiuwen.core.context_engine.execute.asynch.async_executor import AsyncContextExecutor
from jiuwen.core.context_engine.execute.online.pipeline_executor import PipelineContextExecutor
from jiuwen.core.context_engine.processor.assemble.assembler import AssemblerProcessor, AssemblerConfig

class ContextExecutor:
    def __init__(self,
                 accessor: ContextAccessor,
                 llm: Optional[BaseChatModel] = None,
                 config: Optional[ContextEngineConfig] = None):
        self._pipeline_executor: Optional[PipelineContextExecutor] = None
        self._async_executor: Optional[AsyncContextExecutor] = None
        self._llm: Optional[BaseChatModel] = llm
        self._accessor = accessor
        self.build_from_config(config)

    def build_from_config(self, config: ContextEngineConfig):
        self._init_online_executor(config)
        self._init_async_executor(config)

    def assemble(self,
                 context_window: ContextWindow,
                 config: Optional[AssemblerConfig]
                 ) -> Union[str, List[BaseMessage]]:
        assembler = AssemblerProcessor(config or AssemblerConfig())
        output = assembler.run(context_window)
        return output.full_prompt

    def run_process_pipeline(self, inputs: ContextWindow):
        return self._pipeline_executor.run(inputs)

    def _init_async_executor(self, config: ContextEngineConfig):
        if not config:
            return
        self._async_executor = AsyncContextExecutor(
            config,
            self._accessor
        )
        self._async_executor.build_from_config(config, self._llm)

    def _init_online_executor(self, config: ContextEngineConfig):
        if not config:
            return
        self._pipeline_executor = PipelineContextExecutor(config)
        self._pipeline_executor.build_from_config(config, self._llm)
