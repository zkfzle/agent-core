#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import List, Optional

from jiuwen.core.common.logging import logger
from jiuwen.core.utils.llm.base import BaseChatModel
from jiuwen.core.context_engine.base import EngineInput, EngineOutput
from jiuwen.core.context_engine.processor.preprocess.base import PreprocessStage
from jiuwen.core.context_engine.processor.assemble.base import AssembleStage
from jiuwen.core.context_engine.processor.postprocess.base import PostprocessStage
from jiuwen.core.context_engine.processor.factory import ProcessorFactory
from jiuwen.core.context_engine.config import OnlineExecuteConfig


class OnlineProcessPipeline:
    def __init__(self, config: Optional[OnlineExecuteConfig] = None):
        self.__preprocess_pipeline: List[PreprocessStage] = []
        self.__assemble_pipeline: List[AssembleStage] = []
        self.__postprocess_pipeline: List[PostprocessStage] = []
        self.__config: Optional[OnlineExecuteConfig] = config

    def build_from_config(self, config: OnlineExecuteConfig, llm: Optional[BaseChatModel] = None):
        if not config:
            return

        for preprocessor_config in config.preprocess_stage:
            preprocessor = ProcessorFactory().create_processor(preprocessor_config)
            if not preprocessor or not isinstance(preprocessor, PreprocessStage):
                logger.warning(f"preprocessor type error: {preprocessor_config.processor_type}")
                continue
            if preprocessor.need_llm():
                preprocessor.bind_llm(llm)
            self.__preprocess_pipeline.append(preprocessor)

        for assemble_config in config.assemble_stage:
            assembler = ProcessorFactory().create_processor(assemble_config)
            if not assembler or not isinstance(assembler, AssembleStage):
                logger.warning(f"assembler type error: {assemble_config.processor_type}")
                continue
            if assembler.need_llm():
                assembler.bind_llm(llm)
            self.__assemble_pipeline.append(assembler)

        for postprocess_config in config.postprocess_stage:
            postprocessor = ProcessorFactory().create_processor(postprocess_config)
            if not postprocessor or not isinstance(postprocessor, PostprocessStage):
                logger.warning(f"postprocessor type error: {postprocess_config.processor_type}")
                continue
            if postprocessor.need_llm():
                postprocessor.bind_llm(llm)
            self.__postprocess_pipeline.append(postprocessor)

    def run(self, input_data: EngineInput) -> EngineOutput:
        preproc_input = input_data
        preproc_output = self.__run_preprocess(preproc_input)
        assemble_output = self.__run_assemble(preproc_output)
        postproc_output = self.__run_postprocess(assemble_output)
        return postproc_output

    def __run_preprocess(self, preproc_data: EngineInput) -> EngineOutput:
        for preprocessor in self.__preprocess_pipeline:
            preproc_data: EngineOutput = preprocessor.run(preproc_data)
        return EngineOutput.from_input(preproc_data)

    def __run_assemble(self, assemble_data: EngineInput) -> EngineOutput:
        for assembler in self.__assemble_pipeline:
            assemble_data: EngineOutput = assembler.run(assemble_data)
        return EngineOutput.from_input(assemble_data)

    def __run_postprocess(self, postproc_data: EngineInput) -> EngineOutput:
        for postprocessor in self.__postprocess_pipeline:
            postproc_data: EngineOutput = postprocessor.run(postproc_data)
        return EngineOutput.from_input(postproc_data)
