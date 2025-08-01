#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from typing import AsyncIterator


from jiuwen.core.common.logging.base import logger


from jiuwen.core.common.constants.constant import USER_FIELDS
from jiuwen.core.common.utils.utils import TemplateUtils
from jiuwen.core.component.base import WorkflowComponent
from jiuwen.core.context.context import Context
from jiuwen.core.graph.executable import Executable, Input, Output
from jiuwen.core.stream.base import StreamCode

STREAM_CACHE_KEY = "_stream_cache_key"


class End(Executable,WorkflowComponent):
    def __init__(self, node_id: str, node_name: str, conf: dict):
        super().__init__()
        self.node_id = node_id
        self.node_name = node_name
        self.conf = conf
        self.template = conf["responseTemplate"] if "responseTemplate" in conf and len(conf["responseTemplate"])>0 else None

    async def invoke(self, inputs: Input, context: Context) -> Output:
        user_fields = inputs.get(USER_FIELDS)
        if self.template:
            answer = TemplateUtils.render_template(self.template, user_fields)
            output = {}
        else:
            answer = ""
            output = user_fields
        return {
            "responseContent": answer,
            "output": output
        }


    async def stream(self, inputs: Input, context: Context) -> AsyncIterator[Output]:
        try:
            if self.template:
              response_list = TemplateUtils.render_template_to_list(self.template)
              index = 0
              for res in response_list:
                 if res.startswith("{{") and res.endswith("}}"):
                    param_name = res[2:-2]
                    if inputs.get(USER_FIELDS):
                        # 参数从当前input获取
                        param_value = inputs.get(USER_FIELDS).get(param_name)
                    else:
                        # 参数transform存入的context中获取
                        stream_cache_key = self.node_id + STREAM_CACHE_KEY
                        content = context.state().get(stream_cache_key)
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
              final_output = TemplateUtils.render_template(self.template, inputs.get(USER_FIELDS))
            else:
              index = 0
              for res in inputs.get(USER_FIELDS):
                   yield dict(type=StreamCode.PARTIAL_CONTENT.name, index=index, payload=dict(outputs={USER_FIELDS: res}))
                   index += 1
              final_output = dict(outputs={USER_FIELDS: inputs.get(USER_FIELDS)})
            final_index = 0

            yield dict(type=StreamCode.MESSAGE_END.name, index=final_index, payload=dict(outputs={USER_FIELDS: final_output}))
            yield dict(type=StreamCode.WORKFLOW_END.name, index=final_index, payload=dict(outputs={USER_FIELDS: final_output}))
            yield dict(type=StreamCode.FINISH.name, index=final_index, payload=dict(outputs={USER_FIELDS: final_output}))
        except Exception as e:
            logger.info("stream output error: {}".format(e))

    async def collect(self, inputs: AsyncIterator[Input], contex: Context) -> Output:
        pass

    async def transform(self, inputs: AsyncIterator[Input], context: Context) -> AsyncIterator[Output]:
        # 异步遍历输入迭代器
        index = 0
        stream_cache_key = self.node_id + STREAM_CACHE_KEY
        stream_cache_value = {}
        async for input_item in inputs:
            # 将当前输入项存入context
            if isinstance(input_item, dict):
                for key, value in input_item.items():
                    stream_cache_value[key] = stream_cache_value.get(key,"")+ str(value)
            yield dict(type=StreamCode.PARTIAL_CONTENT.name, index=index, payload=dict(answer=input_item))
            index += 1
        context.state().update({stream_cache_key: stream_cache_value})

    async def interrupt(self, message: dict):
        pass

    def to_executable(self) -> Executable:
        return self

