#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
import asyncio
from typing import AsyncIterator, TypedDict, Union

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.common.logging import logger
from jiuwen.core.common.utils.dict import safe_get
from jiuwen.core.common.utils.utils import TemplateUtils
from jiuwen.core.component.base import WorkflowComponent
from jiuwen.core.context_engine.base import Context
from jiuwen.core.graph.executable import Input, Output
from jiuwen.core.runtime.base import ComponentExecutable
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.utils.config.user_config import UserConfig

STREAM_CACHE_KEY = "_stream_cache_key"


class EndConfig(TypedDict):
    responseTemplate: str


class End(ComponentExecutable, WorkflowComponent):
    def __init__(self, conf: Union[EndConfig, dict] = None):
        super().__init__()
        self.conf = conf
        if conf and "responseTemplate" in conf:
            self.template = conf["responseTemplate"]
        else:
            self.template = None

        if self.template and not isinstance(self.template, str):
            raise JiuWenBaseException(StatusCode.WORKFLOW_END_CREATE_VALUE.code,
                                      message=StatusCode.WORKFLOW_END_CREATE_VALUE.errmsg.format(
                                          reason="`responseTemplate` type error, is not str"))
        if self.template and len(self.template) == 0:
            self.template = None
        self.use_cache = False
        self.cache_all = asyncio.Condition()
        self.cache = dict()

    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        if self.template:
            answer = TemplateUtils.render_template(self.template, inputs)
            output = {}
        else:
            answer = ""
            output = {k: v for k, v in inputs.items() if v is not None} if isinstance(inputs, dict) else inputs
        return {
            "responseContent": answer,
            "output": output
        }

    async def stream(self, inputs: Input, runtime: Runtime, context: Context) -> AsyncIterator[Output]:
        if self.use_cache:
            async with self.cache_all:
                await self.cache_all.wait()
        try:
            if self.template:
                response_list = TemplateUtils.render_template_to_list(self.template)
                for res in response_list:
                    if res.startswith("{{") and res.endswith("}}"):
                        param_name = res[2:-2]
                        value = safe_get(inputs, param_name)
                        if value is not None:
                            param_value = value
                        else:
                            param_value = safe_get(self.cache, f'{STREAM_CACHE_KEY}.{param_name}')
                        if param_value is None:
                            continue
                        yield dict(answer=param_value)
                    else:
                        yield dict(answer=res)
            else:
                for key, value in inputs.items():
                    yield dict(output={key: value})

        except Exception as e:
            if UserConfig.is_sensitive():
                logger.info("stream output error")
            else:
                logger.error("stream output error: {}".format(e))
        finally:
            self.cache = dict()

    async def transform(self, inputs: AsyncIterator[Input], runtime: Runtime, context: Context) -> AsyncIterator[
        Output]:
        self.use_cache = True
        if self.template is None:
            async for input_item in inputs:
                yield dict(output=input_item)
        else:
            stream_cache_value = {}
            async for input_item in inputs:
                if isinstance(input_item, dict):
                    for key, value in input_item.items():
                        if value is None:
                            continue
                        stream_cache_value[key] = stream_cache_value.get(key, "") + str(value)
            self.cache.update({STREAM_CACHE_KEY: stream_cache_value})
        async with self.cache_all:
            self.use_cache = False
            self.cache_all.notify_all()
