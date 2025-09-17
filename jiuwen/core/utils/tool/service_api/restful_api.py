#!/usr/bin/python3.11
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

import json
from typing import List

import requests
import aiohttp

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.utils.llm.messages import ToolInfo, Function
from jiuwen.core.utils.tool import constant
from jiuwen.core.utils.tool.base import Tool
from jiuwen.core.utils.tool.constant import Input, Output
from jiuwen.core.utils.tool.service_api.param import Param
from jiuwen.core.utils.tool.service_api.types import ValueTypeEnum


timeout_aiohttp = aiohttp.ClientTimeout(total=constant.REQUEST_TIMEOUT)

class RestfulApi(Tool):
    def __init__(self, name: str, description: str, params: List[Param], path: str, headers: dict, method: str,
                 response: List[Param]):
        super().__init__()
        self.name = name
        self.description = description
        self.params: List[Param] = params
        self.path = path
        self.headers = headers
        self.method = method
        self.response: List[Param] = response

    @staticmethod
    def _basic_type_inputs(param: Param, inputs: dict):
        if ValueTypeEnum.is_nested_array(param.type):
            if not inputs.get(param.name) and inputs.get(param.name) is not False and param.default_value:
                inputs[param.name] = param.default_value
        else:
            if not inputs.get(param.name) and inputs.get(param.name) is not False:
                if param.default_value or param.default_value is False:
                    inputs[param.name] = param.default_value
        return inputs

    @staticmethod
    def _gen_new_inputs(param, inputs):
        if not ValueTypeEnum.is_nested_array(param.type):
            param_value = inputs.get(param.name)
            param_value = {} if not param_value else param_value
            new_inputs = [param_value]
        else:
            new_inputs = inputs.get(param.name, [{}])
            new_inputs = [{}] if not new_inputs else new_inputs
        return new_inputs

    @staticmethod
    def _assign_format_default_value(params: List[Param], inputs: dict):
        for param in params:
            if not param.required:
                continue
            if ValueTypeEnum.is_object(param.type):
                if not inputs.get(param.name) and param.default_value:
                    inputs[param.name] = param.default_value
                    continue
                if inputs.get(param.name) and param.default_value:
                    continue
                new_inputs = RestfulApi._gen_new_inputs(param, inputs)
                temp_inputs = [RestfulApi._assign_format_default_value(param.schema, item) for item in new_inputs]
                if not ValueTypeEnum.is_nested_array(param.type):
                    if not temp_inputs[0]:
                        continue
                    if not inputs:
                        inputs = {}
                    inputs[param.name] = temp_inputs[0]
                else:
                    if any(temp_inputs):
                        inputs[param.name] = temp_inputs
            else:
                inputs = RestfulApi._basic_type_inputs(param, inputs)
        return inputs

    def get_tool_info(self) -> ToolInfo:
        tool_info_dict = Param.format_functions(self)
        func = Function(**tool_info_dict)
        tool_info = ToolInfo(function=func)
        return tool_info


    def format_input_with_default_when_required(self, inputs: dict):
        """format input with default"""
        inputs = RestfulApi._assign_format_default_value(self.params, inputs)
        return inputs

    def get_header_params_from_input(self, inputs: dict):
        """get header params from input"""
        header_params = {}
        for param in self.params:
            if param.method == "Headers" and (inputs.get(param.name) or inputs.get(param.name) is False):
                header_params[param.name] = str(inputs.get(param.name))
                inputs.pop(param.name, None)
        return header_params

    def get_query_params_from_input(self, inputs: dict):
        """get query params from input"""
        query_params = {}
        for param in self.params:
            if inputs.get(param.name) or inputs.get(param.name) is False:
                query_params[param.name] = str(inputs.get(param.name))
                inputs.pop(param.name, None)
        return query_params

    def parse_retrieval_inputs(self, inputs: dict):
        """parse retrieval inputs"""
        if 'retrieval' in self.name:
            if 'multi_queries' not in inputs.keys():
                inputs['query'] = str(inputs.get('query'))
            else:
                for simple_input in inputs['multi_queries']:
                    simple_input['query'] = str(simple_input['query'])
        return inputs

    def invoke(self, inputs: Input, **kwargs) -> Output:
        """invoke api"""
        request_params = RequestParams(self, inputs, **kwargs)
        try:
            request_params.prepare_params()
            response = requests.request(
                self.method, request_params.ip_address_url, headers=request_params.headers,
                verify=False, stream=False, params=request_params.query_params_in_inputs,
                timeout=constant.REQUEST_TIMEOUT,
                **request_params.request_arg
            )
            response_data = _data_of(response)
            return response_data
        except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout):
            return {
                constant.ERR_CODE: StatusCode.PLUGIN_REQUEST_TIMEOUT_ERROR.code,
                constant.ERR_MESSAGE: "plugin request time out",
                constant.RESTFUL_DATA: ""
            }
        except requests.exceptions.ProxyError:
            return {
                constant.ERR_CODE: StatusCode.PLUGIN_PROXY_CONNECT_ERROR.code,
                constant.ERR_MESSAGE: StatusCode.PLUGIN_PROXY_CONNECT_ERROR.errmsg,
                constant.RESTFUL_DATA: ""
            }
        except JiuWenBaseException as error:
            return {
                constant.ERR_CODE: error.error_code,
                constant.ERR_MESSAGE: error.message,
                constant.RESTFUL_DATA: ""
            }
        except Exception:
            return {
                constant.ERR_CODE: StatusCode.PLUGIN_UNEXPECTED_ERROR.code,
                constant.ERR_MESSAGE: "plugin request unknown error",
                constant.RESTFUL_DATA: ""
            }

    async def ainvoke(self, inputs: Input, **kwargs) -> Output:
        """async invoke api"""
        request_params = RequestParams(self, inputs, **kwargs)
        try:
            request_params.prepare_params()
            return await self._async_request(
                dict(ip_address_url=request_params.ip_address_url,
                     headers=request_params.headers,
                     request_arg=request_params.request_arg,
                     query_params_in_inputs=request_params.query_params_in_inputs)
            )
        except Exception:
            return {
                constant.ERR_CODE: StatusCode.PLUGIN_UNEXPECTED_ERROR.code,
                constant.ERR_MESSAGE: "plugin request unknown error",
                constant.RESTFUL_DATA: ""
            }

    async def _async_request(self, request_args: dict):
        ip_address_url = request_args.get('ip_address_url')
        query_params_in_inputs = request_args.get('query_params_in_inputs')
        request_arg = request_args.get('request_arg')
        async with aiohttp.ClientSession() as session:
            async with session.request(
                self.method, ip_address_url, headers=request_args.get("headers"),
                allow_redirects=False, timeout=timeout_aiohttp,
                params=query_params_in_inputs, **request_arg
            ) as response:
                response_data = await _data_of_async_request(response)
        return response_data

class RequestParams:
    """Restful API request parameters"""

    def __init__(self, restful_api: RestfulApi, inputs: Input, **kwargs):
        self.restful_api = restful_api
        self.inputs = inputs
        self.kwargs = kwargs

        inputs = restful_api.format_input_with_default_when_required(inputs)
        self.header_params_in_inputs = restful_api.get_header_params_from_input(inputs)
        self.query_params_in_inputs = restful_api.get_query_params_from_input(inputs)
        self.inputs = restful_api.parse_retrieval_inputs(inputs)

        self.method = restful_api.method.upper()
        restful_api.method = self.method

        self.ip_address_url = None
        self.headers = None
        self.request_arg = None

    def prepare_params(self):
        """prepare params"""
        restful_api = self.restful_api
        if restful_api.method not in constant.HTTP_METHOD:
            raise JiuWenBaseException(
                error_code=StatusCode.PLUGIN_UNEXPECTED_ERROR.code,
                message="the http method is not supported"
            )
        url = restful_api.path
        headers = restful_api.headers if isinstance(restful_api.headers, dict) else {}
        headers.update(self.header_params_in_inputs)
        request_arg = dict(json=self.inputs)
        self.ip_address_url = url
        self.headers = headers
        self.request_arg = request_arg


def _data_of(response):
    if response.status_code == 200:
        content = b""
        try:
            for chunk in response.iter_content(chunk_size=1024):
                content += chunk
                if len(content) > constant.MAX_RESULT_SIZE:
                    raise JiuWenBaseException(
                        error_code=StatusCode.PLUGIN_UNEXPECTED_ERROR.code,
                        message=StatusCode.PLUGIN_UNEXPECTED_ERROR.errmsg
                    )
            res = json.loads(content.decode('utf-8'))
        except json.JSONDecodeError:
            return {constant.ERR_CODE: 0, constant.ERR_MESSAGE: 'success', constant.RESTFUL_DATA: response.text}
        if constant.ERR_CODE not in res or constant.ERR_MESSAGE not in res or constant.RESTFUL_DATA not in res:
            return {constant.ERR_CODE: 0, constant.ERR_MESSAGE: 'success', constant.RESTFUL_DATA: res}
        return res
    raise JiuWenBaseException(
        error_code=StatusCode.PLUGIN_RESPONSE_HTTP_CODE_ERROR.code,
        message=f"Plugin response code: {response.status_code} error."
    )


async def _data_of_async_request(response: aiohttp.ClientResponse):
    if response.status == 200:
        content = b""
        try:
            async for chunk in response.content.iter_chunked(1024):
                content += chunk
                if len(content) > constant.MAX_RESULT_SIZE:
                    raise JiuWenBaseException(
                        error_code=StatusCode.PLUGIN_RESPONSE_TOO_BIG_ERROR.code,
                        message=StatusCode.PLUGIN_RESPONSE_TOO_BIG_ERROR.errmsg
                    )
            res = json.loads(content.decode('utf-8'))
        except json.JSONDecodeError:
            return {
                constant.ERR_CODE: 0,
                constant.ERR_MESSAGE: 'success',
                constant.RESTFUL_DATA: content.decode('utf-8')
            }
        if constant.ERR_CODE not in res or constant.ERR_MESSAGE not in res or constant.RESTFUL_DATA not in res:
            return {constant.ERR_CODE: 0, constant.ERR_MESSAGE: 'success', constant.RESTFUL_DATA: res}
        return res
    raise JiuWenBaseException(
        error_code=StatusCode.PLUGIN_RESPONSE_HTTP_CODE_ERROR.code,
        message=f"Plugin response code: {response.status} error."
    )
