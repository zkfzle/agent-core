#!/usr/bin/python3.11
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

import aiohttp
import json
from typing import List, Dict, Any, Iterator, AsyncIterator, Optional

from aiohttp import ClientSession
from pydantic import ConfigDict
from requests import Session

from jiuwen.core.utils.llm.base import BaseChatModel, BaseModelInfo
from jiuwen.core.utils.llm.messages import AIMessage, UsageMetadata, ToolInfo, FunctionInfo, ToolCall
from jiuwen.core.utils.llm.messages_chunk import AIMessageChunk


class RequestChatModel(BaseChatModel, BaseModelInfo):
    model_info: BaseModelInfo
    model_config = ConfigDict(arbitrary_types_allowed=True)
    sync_client: Session = Session()
    aiohttp_session: Optional[ClientSession] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stream_state = {
            'current_tool_call_id': '',
            'current_tool_name': '',
            'current_tool_args': '',
            'tool_calls': []
        }
        self._usage = dict()

    async def ensure_session(self):
        if self.aiohttp_session is None or self.aiohttp_session.closed:
            self.aiohttp_session = aiohttp.ClientSession()

    async def close_session(self):
        if self.aiohttp_session is not None and not self.aiohttp_session.closed:
            await self.aiohttp_session.close()
            self.aiohttp_session = None

    def model_provider(self) -> str:
        return "generic_http_api"

    def _invoke(self, messages: List[Dict], tools: List[Dict] = None, **kwargs: Any) -> AIMessage:
        params = self._request_params(messages, tools, **kwargs)

        response = self.sync_client.post(
            verify=False,
            url=self.model_info.api_base,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.model_info.api_key}"
            },
            json=params,
            timeout=self.model_info.timeout
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    async def _ainvoke(self, messages: List[Dict], tools: List[Dict] = None, **kwargs: Any) -> AIMessage:
        await self.ensure_session()
        params = self._request_params(messages, tools, **kwargs)
        async with self.aiohttp_session.post(
                url=self.model_info.api_base,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.model_info.api_key}"
                },
                json=params,
                timeout=self.model_info.timeout
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return self._parse_response(data)

    def _stream(self, messages: List[Dict], tools: List[Dict] = None, **kwargs: Any) -> Iterator[AIMessageChunk]:
        # 重置流状态
        self._reset_stream_state()

        params = self._request_params(messages, tools, **kwargs)
        params["stream"] = True

        with self.sync_client.post(
                verify=False,
                url=self.model_info.api_base,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.model_info.api_key}"
                },
                json=params,
                stream=True,
                timeout=self.model_info.timeout
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = self._parse_stream_line(line)
                    if chunk:
                        yield chunk

    async def _astream(self, messages: List[Dict], tools: List[Dict] = None, **kwargs: Any) -> AsyncIterator[
        AIMessageChunk]:
        # 重置流状态
        self._reset_stream_state()

        await self.ensure_session()
        params = self._request_params(messages, tools, **kwargs)
        params["stream"] = True

        async with self.aiohttp_session.post(
                url=self.model_info.api_base,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.model_info.api_key}"
                },
                json=params,
                timeout=aiohttp.ClientTimeout(total=self.model_info.timeout)
        ) as response:
            response.raise_for_status()
            async for line in response.content:
                if line:
                    chunk = self._parse_stream_line(line)
                    if chunk:
                        yield chunk
        await self.close_session()

    def _request_params(self, messages: List[Dict], tools: List[Dict] = None, **kwargs: Any) -> Dict:
        params = {
            "model": self.model_info.model_name,
            "messages": messages,
            "temperature": self.model_info.temperature,
            "top_p": self.model_info.top_p,
            **kwargs
        }

        if tools:
            params["tools"] = tools

        return params

    def _parse_response(self, response_data: Dict) -> AIMessage:
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = "" if message.get("content") is None else message.get("content")
        return AIMessage(
            content=content,
            tool_calls=message.get("tool_calls", []),
            usage_metadata=UsageMetadata(
                model_name=self.model_name,
                finish_reason=choice.get("finish_reason", ""),
                total_latency=response_data.get('usage', {}).get('total_tokens', 0)
            )
        )

    def _reset_stream_state(self):
        """重置流处理状态"""
        self._stream_state = {
            'current_tool_call_id': '',
            'current_tool_name': '',
            'current_tool_args': '',
            'tool_calls': []
        }

    def _parse_stream_line(self, line: bytes) -> Optional[AIMessageChunk]:
        if line.startswith(b"data: "):
            line = line[6:]

        if line.strip() == b"[DONE]":
            # 处理流结束，返回最终的工具调用信息
            tool_calls = []
            if (self._stream_state['current_tool_name'] and
                    self._stream_state['current_tool_args']):
                function = FunctionInfo(
                    name=self._stream_state['current_tool_name'],
                    arguments=self._stream_state['current_tool_args']
                )
                tool_call = ToolCall(
                    args={"name": self._stream_state['current_tool_name'], "arguments": self._stream_state['current_tool_args']},
                    id=self._stream_state['current_tool_call_id'],
                    function=function,
                    type="function_call"
                )
                tool_calls.append(tool_call)

            # 添加之前完成的工具调用
            tool_calls.extend(self._stream_state['tool_calls'])

            chunk = AIMessageChunk(
                content="",
                reason_content="",
                tool_calls=tool_calls,
                usage_metadata=UsageMetadata(**self._usage)
            )
            return chunk

        try:
            data = json.loads(line.decode("utf-8"))
            choice = data.get("choices", [{}])[0]
            finish_reason = choice.get("finish_reason")
            usage = data.get("usage", dict())
            usage.update(dict(finish_reason=finish_reason or ""))
            self._usage = usage
            delta = choice.get("delta", {})
            content = delta.get("content", "") or ""
            reasoning_content = delta.get("reasoning_content", "") or ""

            # 处理工具调用
            tool_calls_delta = delta.get("tool_calls")
            tool_calls = []

            if tool_calls_delta:
                for tool_call_delta in tool_calls_delta:
                    index = tool_call_delta.get("index", 0)
                    tool_call_id = tool_call_delta.get("id", "")
                    function_delta = tool_call_delta.get("function", {})

                    if index == 0:  # 假设我们只处理第一个工具调用
                        if tool_call_id:
                            self._stream_state['current_tool_call_id'] = tool_call_id

                        name_delta = function_delta.get("name", "")
                        if name_delta:
                            self._stream_state['current_tool_name'] += name_delta

                        args_delta = function_delta.get("arguments", "")
                        if args_delta:
                            self._stream_state['current_tool_args'] += args_delta

            return AIMessageChunk(
                content=content,
                reason_content=reasoning_content,
                tool_calls=tool_calls,
                usage_metadata=UsageMetadata(**usage)
            )
        except json.JSONDecodeError:
            return None

    async def close(self):
        if self.aiohttp_session:
            await self.aiohttp_session.close()
