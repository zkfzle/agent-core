#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import aiohttp
import json
from typing import List, Dict, Any, Iterator, AsyncIterator, Optional

from pydantic import ConfigDict
from requests import Session
import openai

from openjiuwen.core.common.logging import logger
from openjiuwen.core.common.security.ssl_utils import SslUtils
from openjiuwen.core.common.security.url_utils import UrlUtils
from openjiuwen.core.common.security.user_config import UserConfig
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.llm.messages import AIMessage, UsageMetadata
from openjiuwen.core.utils.tool.schema import ToolCall
from openjiuwen.core.utils.llm.messages_chunk import AIMessageChunk


class RequestChatModel(BaseModelClient):

    model_config = ConfigDict(arbitrary_types_allowed=True)
    sync_client: Session = Session()

    def __init__(self, api_key: str, api_base: str, max_retries: int = 3, timeout: int = 60, **kwargs):
        super().__init__(api_key=api_key, api_base=api_base, max_retries=max_retries, timeout=timeout, **kwargs)
        self._stream_state = {
            'current_tool_call_id': '',
            'current_tool_name': '',
            'current_tool_args': '',
            'tool_calls': []
        }
        self._usage = dict()
        self._setup_ssl_adapter()

    def close_session(self):
        if self.sync_client is not None:
            self.sync_client.close()

    def model_provider(self) -> str:
        return "generic_http_api"

    def _setup_ssl_adapter(self):
        """Setup SSL adapter, mount only when SSL verification enabled"""
        adapter = SslUtils.create_ssl_adapter("LLM_SSL_VERIFY", "LLM_SSL_CERT", ["false"])
        if adapter is not None:
            self.sync_client.mount("https://", adapter)

    def _invoke(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                temperature: Optional[float] = None, top_p: Optional[float] = None, **kwargs) -> AIMessage:
        UrlUtils.check_url_is_valid(self.api_base)
        messages = self.sanitize_tool_calls(messages)
        model_params = self._update_model_params(temperature=temperature, top_p=top_p, **kwargs)
        params = self._request_params(model_name=model_name, messages=messages, tools=tools, **model_params)

        ssl_verify, ssl_cert = SslUtils.get_ssl_config("LLM_SSL_VERIFY", "LLM_SSL_CERT", ["false"])
        verify = ssl_cert if ssl_verify else False

        response = self.sync_client.post(
                verify=verify,
                url=self.api_base,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=params,
                proxies=UrlUtils.get_global_proxies(self.api_base),
                allow_redirects=False,
                timeout=self.timeout
            )

        response.raise_for_status()
        self.close_session()
        return self._parse_response(model_name, response.json())

    async def _ainvoke(self, model_name:str, messages: List[Dict], tools: List[Dict] = None,
                       temperature: Optional[float] = None, top_p: Optional[float] = None, **kwargs: Any) -> AIMessage:
        UrlUtils.check_url_is_valid(self.api_base)
        messages = self.sanitize_tool_calls(messages)
        model_params = self._update_model_params(temperature=temperature, top_p=top_p, **kwargs)
        params = self._request_params(model_name=model_name, messages=messages, tools=tools, **model_params)
        ssl_verify, ssl_cert = SslUtils.get_ssl_config("LLM_SSL_VERIFY", "LLM_SSL_CERT", ["false"])
        
        if ssl_verify:
            ssl_context = SslUtils.create_strict_ssl_context(ssl_cert)
            connector = aiohttp.TCPConnector(ssl=ssl_context)
        else:
            connector = aiohttp.TCPConnector(ssl=False)
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                    url=self.api_base,
                    proxy=UrlUtils.get_global_proxy_url(self.api_base),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    json=params,
                    allow_redirects=False,
                    timeout=timeout
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return self._parse_response(model_name, data)

    def _stream(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                temperature: Optional[float] = None, top_p: Optional[float] = None, **kwargs: Any) -> Iterator[
        AIMessageChunk]:
        UrlUtils.check_url_is_valid(self.api_base)
        self._reset_stream_state()

        messages = self.sanitize_tool_calls(messages)
        model_params = self._update_model_params(temperature=temperature, top_p=top_p, **kwargs)
        params = self._request_params(model_name=model_name, messages=messages, tools=tools, **model_params)
        params["stream"] = True
        ssl_verify, ssl_cert = SslUtils.get_ssl_config("LLM_SSL_VERIFY", "LLM_SSL_CERT", ["false"])
        verify = ssl_cert if ssl_verify else False

        with self.sync_client.post(
                verify=verify,
                url=self.api_base,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=params,
                proxies=UrlUtils.get_global_proxies(self.api_base),
                stream=True,
                allow_redirects=False,
                timeout=self.timeout
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = self._parse_stream_line(line)
                    if chunk:
                        yield chunk
        self.close_session()


    async def _astream(self, model_name:str, messages: List[Dict], tools: List[Dict] = None,
                       temperature: Optional[float] = None, top_p: Optional[float] = None,
                       **kwargs: Any) -> AsyncIterator[AIMessageChunk]:
        UrlUtils.check_url_is_valid(self.api_base)
        self._reset_stream_state()

        messages = self.sanitize_tool_calls(messages)
        model_params = self._update_model_params(temperature=temperature, top_p=top_p, **kwargs)
        params = self._request_params(model_name=model_name, messages=messages, tools=tools, **model_params)
        params["stream"] = True

        ssl_verify, ssl_cert = SslUtils.get_ssl_config("LLM_SSL_VERIFY", "LLM_SSL_CERT", ["false"])
        
        if ssl_verify:
            ssl_context = SslUtils.create_strict_ssl_context(ssl_cert)
            connector = aiohttp.TCPConnector(ssl=ssl_context)
        else:
            connector = aiohttp.TCPConnector(ssl=False)

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                    url=self.api_base,
                    proxy=UrlUtils.get_global_proxy_url(self.api_base),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    json=params,
                    allow_redirects=False,
                    timeout=timeout
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line:
                        chunk = self._parse_stream_line(line)
                        if chunk:
                            yield chunk

    def sanitize_tool_calls(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sanitize tool_calls in messages, keep OpenAI standard fields:
        id, type, function.name, function.arguments
        Force type to "function"
        """
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue

            cleaned = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                # Extract only valid fields
                func = tc.get("function", {})
                cleaned.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", "")
                    }
                })
            msg["tool_calls"] = cleaned
        return messages

    def _request_params(self, model_name: str, messages: List[Dict], tools: List[Dict] = None, **kwargs: Any) -> Dict:
        params = {
            "model": model_name,
            "messages": messages,
            **kwargs
        }

        if tools:
            params["tools"] = tools

        if UserConfig.is_sensitive():
            logger.info("Before request chat model, request params is ready.")
        else:
            logger.info(f"Before request chat model, request params is ready. "
                        f"params: {params}, timeout: {self.timeout}")

        return params

    def _parse_response(self, model_name: str, response_data: Dict) -> AIMessage:
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = "" if message.get("content") is None else message.get("content")
        return AIMessage(
            content=content,
            tool_calls=self._convert_tool_call_format(message.get("tool_calls", [])),
            usage_metadata=UsageMetadata(
                model_name=model_name,
                finish_reason=choice.get("finish_reason", ""),
                total_latency=response_data.get('usage', {}).get('total_tokens', 0)
            )
        )

    def _reset_stream_state(self):
        """Reset stream processing state"""
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
            # Handle stream end, return final tool call info
            tool_calls = []
            if (self._stream_state['current_tool_name'] and
                    self._stream_state['current_tool_args']):
                tool_call = ToolCall(
                    id=self._stream_state['current_tool_call_id'],
                    type="function_call",
                    name=self._stream_state['current_tool_name'],
                    arguments=self._stream_state['current_tool_args']
                )
                tool_calls.append(tool_call)

            # Add previously completed tool calls
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
            usage = data.get("usage", {})
            usage.update(dict(finish_reason=finish_reason or ""))
            self._usage = usage
            delta = choice.get("delta", {})
            content = delta.get("content", "") or ""
            reasoning_content = delta.get("reasoning_content", "") or ""

            # Handle tool calls
            tool_calls_delta = delta.get("tool_calls")
            tool_calls = []

            if tool_calls_delta:
                for tool_call_delta in tool_calls_delta:
                    index = tool_call_delta.get("index", 0)
                    tool_call_id = tool_call_delta.get("id", "")
                    function_delta = tool_call_delta.get("function", {})

                    if index == 0:
                        if tool_call_id:
                            self._stream_state['current_tool_call_id'] = tool_call_id

                        name_delta = function_delta.get("name", "")
                        if name_delta:
                            self._stream_state['current_tool_name'] += name_delta

                        args_delta = function_delta.get("arguments", "")
                        if args_delta:
                            self._stream_state['current_tool_args'] += args_delta

            if not content and not reasoning_content and not tool_calls:
                return None

            return AIMessageChunk(
                content=content,
                reason_content=reasoning_content,
                tool_calls=tool_calls,
                usage_metadata=UsageMetadata(**usage)
            )
        except json.JSONDecodeError:
            return None

    async def close(self):
        pass

    @staticmethod
    def _convert_tool_call_format(tool_calls: List[Dict]):
        if not tool_calls:
            return []
        result = []
        for tool_call in tool_calls:
            result.append(ToolCall(
                id=tool_call.get("id", ""),
                type=tool_call.get("type", ""),
                name=tool_call.get("function", {}).get("name", ""),
                arguments=tool_call.get("function", {}).get("arguments", "")
            ))
        return result


class OpenAIChatModel(BaseModelClient):
    """OpenAI-specific chat model implementation using official openai library"""

    def __init__(self,
                 api_key: str, api_base: str, max_retries: int = 3, timeout: int = 60, **kwargs):
        super().__init__(api_key=api_key, api_base=api_base, max_retries=max_retries, timeout=timeout, **kwargs)

    def model_provider(self) -> str:
        return "openai"

    def _invoke(self, model_name:str, messages: List[Dict], tools: List[Dict] = None,
                temperature: Optional[float] = None, top_p: Optional[float] = None, **kwargs: Any) -> AIMessage:
        model_params = self._update_model_params(temperature=temperature, top_p=top_p, **kwargs)
        params = self._build_request_params(model_name=model_name, messages=messages, tools=tools, **model_params)
        sync_client = None
        try:
            sync_client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
            response = sync_client.chat.completions.create(**params)
            return self._parse_openai_response(model_name, response)
        except Exception as e:
            if UserConfig.is_sensitive():
                raise Exception("OpenAI API error")
            else:
                raise Exception(f"OpenAI API error: {str(e)}")
        finally:
            if sync_client is not None:
                sync_client.close()

    async def _ainvoke(self, model_name:str, messages: List[Dict], tools: List[Dict] = None,
                       temperature: Optional[float] = None, top_p: Optional[float] = None, **kwargs: Any) -> AIMessage:
        """Async call OpenAI API"""
        model_params = self._update_model_params(temperature=temperature, top_p=top_p, **kwargs)
        params = self._build_request_params(model_name=model_name, messages=messages, tools=tools, **model_params)
        async_client = None
        try:
            async_client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)
            response = await async_client.chat.completions.create(**params)
            return self._parse_openai_response(model_name, response)
        except Exception as e:
            if UserConfig.is_sensitive():
                raise Exception("OpenAI API async error")
            else:
                raise Exception(f"OpenAI API async error: {str(e)}")
        finally:
            if async_client is not None:
                await async_client.close()

    def _stream(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                temperature: Optional[float] = None, top_p: Optional[float] = None, **kwargs: Any) -> Iterator[
        AIMessageChunk]:
        model_params = self._update_model_params(temperature=temperature, top_p=top_p, **kwargs)
        params = self._build_request_params(model_name=model_name, messages=messages, tools=tools, stream=True,
                                            **model_params)
        sync_client = None
        try:
            sync_client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
            stream = sync_client.chat.completions.create(**params)
            for chunk in stream:
                parsed_chunk = self._parse_openai_stream_chunk(model_name, chunk)
                if parsed_chunk:
                    yield parsed_chunk
        except Exception as e:
            if UserConfig.is_sensitive():
                raise Exception("OpenAI API stream error")
            else:
                raise Exception(f"OpenAI API stream error: {str(e)}")
        finally:
            if sync_client is not None:
                sync_client.close()

    async def _astream(self, model_name:str, messages: List[Dict], tools: List[Dict] = None,
                       temperature: Optional[float] = None, top_p: Optional[float] = None,
                       **kwargs: Any) -> AsyncIterator[AIMessageChunk]:
        """Async stream call OpenAI API"""
        model_params = self._update_model_params(temperature=temperature, top_p=top_p, **kwargs)
        params = self._build_request_params(model_name=model_name, messages=messages, tools=tools, stream=True,
                                            **model_params)
        async_client = None
        try:
            async_client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)
            stream = await async_client.chat.completions.create(**params)
            async for chunk in stream:
                parsed_chunk = self._parse_openai_stream_chunk(model_name, chunk)
                if parsed_chunk:
                    yield parsed_chunk
        except Exception as e:
            if UserConfig.is_sensitive():
                raise Exception("OpenAI API async stream error")
            else:
                raise Exception(f"OpenAI API async stream error: {str(e)}")
        finally:
            if async_client is not None:
                await async_client.close()


    def _build_request_params(self, model_name:str, messages: List[Dict],
                              tools: List[Dict] = None, stream: bool = False,
                              **kwargs) -> Dict:
        """Build OpenAI API request parameters"""
        params = {
            "model": model_name,
            "messages": messages,
            "stream": stream,
            "timeout": self.timeout,
            **kwargs
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        if UserConfig.is_sensitive():
            logger.info("Before request openai chat model, request params is ready.")
        else:
            logger.info(f"Before request openai chat model, request params is ready. params: {params}")

        return params


    def _parse_openai_response(self, model_name, response) -> AIMessage:
        """Parse OpenAI API response"""
        choice = response.choices[0]
        message = choice.message

        # Parse tool calls
        tool_calls = []
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tc in message.tool_calls:
                function_name = getattr(getattr(tc, 'function', None), 'name', None) or ""
                function_arguments = getattr(getattr(tc, 'function', None), 'arguments', None) or ""
                tool_call = ToolCall(
                    id=getattr(tc, 'id', '') or "",
                    type="function",
                    name=function_name,
                    arguments=function_arguments
                )
                tool_calls.append(tool_call)

        return AIMessage(
            content=message.content or "",
            tool_calls=tool_calls,
            usage_metadata=UsageMetadata(
                model_name=model_name,
                finish_reason=choice.finish_reason or "",
                total_latency=response.usage.total_tokens if response.usage else 0
            )
        )


    def _parse_openai_stream_chunk(self, model_name, chunk) -> Optional[AIMessageChunk]:
        """Parse OpenAI stream response chunk"""
        if not chunk.choices:
            return None

        choice = chunk.choices[0]
        delta = choice.delta

        content = getattr(delta, 'content', None) or ""
        tool_calls = []

        # Handle tool call delta
        if hasattr(delta, 'tool_calls') and delta.tool_calls:
            for tc_delta in delta.tool_calls:
                if hasattr(tc_delta, 'function') and tc_delta.function:
                    function_name = getattr(tc_delta.function, 'name', None) or ""
                    function_arguments = getattr(tc_delta.function, 'arguments', None) or ""
                    tool_call = ToolCall(
                        id=getattr(tc_delta, 'id', '') or "",
                        type="function",
                        name=function_name,
                        arguments=function_arguments
                    )
                    tool_calls.append(tool_call)

        usage_metadata = None
        if hasattr(chunk, 'usage') and chunk.usage:
            usage_metadata = UsageMetadata(
                model_name=model_name,
                finish_reason=choice.finish_reason or "",
                total_latency=chunk.usage.total_tokens if chunk.usage else 0
            )

        return AIMessageChunk(
            content=content,
            tool_calls=tool_calls,
            usage_metadata=usage_metadata
        )
