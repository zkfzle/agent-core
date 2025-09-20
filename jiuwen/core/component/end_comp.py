#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from typing import AsyncIterator, TypedDict, Union

from jiuwen.core.common.logging import logger
from jiuwen.core.common.utils.utils import TemplateUtils
from jiuwen.core.component.base import WorkflowComponent
from jiuwen.core.context_engine.base import Context
from jiuwen.core.graph.executable import Input, Output
from jiuwen.core.runtime.base import ComponentExecutable
from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.stream.base import StreamCode

STREAM_CACHE_KEY = "_stream_cache_key"

class EndConfig(TypedDict):
    responseTemplate: str

class End(ComponentExecutable, WorkflowComponent):
    def __init__(self, conf: Union[EndConfig, dict] = None):
        super().__init__()
        self.conf = conf
        self.template = conf["responseTemplate"] if ( conf and
                "responseTemplate" in conf and len(conf["responseTemplate"]) > 0) else None

    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        if self.template:
            answer = TemplateUtils.render_template(self.template, inputs)
            output = {}
        else:
            answer = ""
            # 只输出inputs中值不为None的键值对
            output = {k: v for k, v in inputs.items() if v is not None} if isinstance(inputs, dict) else inputs
        return {
            "responseContent": answer,
            "output": output
        }

    async def stream(self, inputs: Input, runtime: Runtime, context: Context) -> AsyncIterator[Output]:
        try:
            if self.template:
                response_list = TemplateUtils.render_template_to_list(self.template)
                index = 0
                for res in response_list:
                    if res.startswith("{{") and res.endswith("}}"):
                        param_name = res[2:-2]
                        if inputs:
                            # 参数从当前input获取
                            param_value = inputs.get(param_name)
                        else:
                            # 参数transform存入的runtime中获取
                            content = runtime.get_state(STREAM_CACHE_KEY)
                            if content:
                                param_value = content.get(param_name)
                            else:
                                param_value = None
                        if param_value is None:
                            continue
                        yield dict(type=StreamCode.PARTIAL_CONTENT.name, index=index, payload=dict(answer=param_value))
                    else:
                        yield dict(type=StreamCode.PARTIAL_CONTENT.name, index=index, payload=dict(answer=res))
                    index += 1
                final_output = TemplateUtils.render_template(self.template, inputs)
            else:
                index = 0
                for res in inputs:
                    yield dict(type=StreamCode.PARTIAL_CONTENT.name, index=index,
                               payload=dict(outputs=res))
                    index += 1
                final_output = dict(outputs=inputs)
            final_index = 0

            yield dict(type=StreamCode.MESSAGE_END.name, index=final_index, payload=dict(outputs=final_output))
            yield dict(type=StreamCode.WORKFLOW_END.name, index=final_index, payload=dict(outputs=final_output))
            yield dict(type=StreamCode.FINISH.name, index=final_index, payload=dict(outputs=final_output))
        except Exception as e:
            logger.info("stream output error: {}".format(e))

    async def transform(self, inputs: AsyncIterator[Input], runtime: Runtime, context: Context) -> AsyncIterator[Output]:
        # 异步遍历输入迭代器
        index = 0
        stream_cache_value = {}
        async for input_item in inputs:
            # 将当前输入项存入runtime
            if isinstance(input_item, dict):
                for key, value in input_item.items():
                    stream_cache_value[key] = stream_cache_value.get(key, "") + str(value)
            yield dict(type=StreamCode.PARTIAL_CONTENT.name, index=index, payload=dict(answer=input_item))
            index += 1
        runtime.update_state({STREAM_CACHE_KEY: stream_cache_value})
