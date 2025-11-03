#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import asyncio
from typing import AsyncIterator, TypedDict, Union, AsyncGenerator

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.common.logging import logger
from jiuwen.core.component.base import WorkflowComponent
from jiuwen.core.context_engine.base import Context
from jiuwen.core.graph.executable import Input, Output
from jiuwen.core.runtime.base import ComponentExecutable
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.runtime.utils import get_value_by_nested_path
from jiuwen.core.utils.common.dict_utils import extract_leaf_nodes, format_path
from jiuwen.core.utils.common.verify_utils import TemplateUtils
from jiuwen.core.utils.config.user_config import UserConfig

STREAM_CACHE_KEY = "_stream_cache_key"


class EndConfig(TypedDict):
    responseTemplate: str


class End(ComponentExecutable, WorkflowComponent):
    def __init__(self, conf: Union[EndConfig, dict] = None):
        super().__init__()
        self.conf = conf
        self.template = None
        if conf is not None and conf.get("responseTemplate") is not None:
            template = conf["responseTemplate"]
            if not isinstance(template, str):
                raise JiuWenBaseException(StatusCode.WORKFLOW_END_CREATE_VALUE.code,
                                          message=StatusCode.WORKFLOW_END_CREATE_VALUE.errmsg.format(
                                              reason="`responseTemplate` type error, is not str"))
            if template != "":
                self.template = TemplateProcessor(template)

    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        if self.template:
            answer = self.template.render(inputs)
            output = {}
        else:
            answer = ""
            output = {k: v for k, v in inputs.items() if v is not None} if isinstance(inputs, dict) else inputs
        return {
            "responseContent": answer,
            "output": output
        }

    async def stream(self, inputs: Input, runtime: Runtime, context: Context) -> AsyncIterator[Output]:
        logger.debug(f"end component stream method inputs: {inputs}")
        try:
            if self.template is not None:
                generator = self.template.render_stream(inputs)
                async for frame in generator:
                    logger.debug(f"rendering stream frame: {frame}")
                    yield dict(answer=frame)
            else:
                for key, value in inputs.items():
                    yield dict(output={key: value})

        except Exception as e:
            if UserConfig.is_sensitive():
                logger.info("stream output error")
            else:
                logger.error("stream output error: {}".format(e))

    async def transform(self, inputs: Input, runtime: Runtime, context: Context) -> AsyncIterator[Output]:
        logger.debug(f"end component transform method inputs: {inputs}")
        if self.template is not None:
            generator = self.template.render_stream(inputs)
            async for frame in generator:
                logger.debug(f"rendering transform frame: {frame}")
                yield dict(answer=frame)
        else:
            for (path, value) in extract_leaf_nodes(inputs):
                if isinstance(value, AsyncGenerator):
                    async for frame in value:
                        yield dict(output={format_path(path): frame})
                else:
                    yield dict(output={format_path(path): value})


class TemplateProcessor:
    def __init__(self, template: str):
        self.template = template
        response_list = TemplateUtils.render_template_to_list(template)
        self.segments = response_list
        self.variables_positions: set[int] = set()
        self.current_position = 0
        for pos, res in enumerate(response_list):
            if res.startswith("{{") and res.endswith("}}"):
                self.variables_positions.add(pos)
                self.segments[pos] = res[2:-2]

        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition()

    def current_position(self) -> int:
        return self.current_position

    def get_current_segment(self) -> str:
        return self._get_segment(self.current_position)

    def _get_segment(self, pos: int) -> str:
        if pos >= len(self.segments):
            return ""
        return self.segments[pos]

    def should_render(self) -> bool:
        return self.current_position in self.variables_positions

    def advance_position(self) -> int:
        self.current_position += 1
        return self.current_position

    def render(self, inputs: dict) -> str:
        return TemplateUtils.render_template(self.template, inputs)

    async def render_stream(self, inputs: dict) -> AsyncGenerator:
        should_wait = False
        while True:
            if should_wait:
                async with self.condition:
                    try:
                        await asyncio.wait_for(self.condition.wait(), timeout=0.2)  # TODO: set timeout by config
                    except asyncio.TimeoutError as e:
                        logger.error(f"render template stream timeout, {e}")
                        self.advance_position()
                should_wait = False
                logger.debug("previous segment has been finished")
            async with self.lock:
                if self.is_finished():
                    break

                segment = self.get_current_segment()
                if not self.should_render():
                    yield segment
                    self.advance_position()
                    continue

                value = get_value_by_nested_path(segment, inputs)
                if value is None:
                    logger.debug(f"current segment [{segment}] should wait for other method")
                    should_wait = True
                    continue

                if isinstance(value, AsyncGenerator):
                    logger.debug(f"current segment generator [{segment}] is generator")
                    async for frame in value:
                        logger.debug(f"rendering generator frame: {frame}")
                        yield frame
                else:
                    yield value
                self.advance_position()
                async with self.condition:
                    self.condition.notify_all()
                logger.debug(f"current segment [{segment}] has been finished")

    def is_finished(self) -> bool:
        return self.current_position >= len(self.segments)
