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
from jiuwen.core.utils.llm.messages import AIMessage, UsageMetadata
from jiuwen.core.utils.llm.messages_chunk import AIMessageChunk


class RequestChatModel(BaseChatModel, BaseModelInfo):
    model_info: BaseModelInfo
    model_config = ConfigDict(arbitrary_types_allowed=True)
    sync_client: Session = Session()
    aiohttp_session: Optional[ClientSession] = None

    async def ensure_session(self):
        if self.aiohttp_session is None or self.aiohttp_session.closed:
            self.aiohttp_session = aiohttp.ClientSession()

    def model_provider(self) -> str:
        return "generic_http_api"

    def _invoke(self, messages: List[Dict], tools: List[Dict] = None, **kwargs: Any) -> AIMessage:
        params = self._request_params(messages, tools, *kwargs)

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
        params = self._request_params(messages, tools, *kwargs)
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
        params = self._request_params(messages, tools, *kwargs)

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
        await self.ensure_session()
        params = self._request_params(messages, tools, *kwargs)
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

    def _parse_stream_line(self, line: bytes) -> Optional[AIMessageChunk]:
        if line.startswith(b"data: "):
            line = line[6:]

        if line.strip() == b"[DONE]":
            return None

        try:
            data = json.loads(line.decode("utf-8"))
            choice = data.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            if content is None:
                content = ""

            return AIMessageChunk(
                content=content,
                tool_calls=delta.get("tool_calls", [])
            )
        except json.JSONDecodeError:
            return None

    async def close(self):
        if self.aiohttp_session:
            await self.aiohttp_session.close()
