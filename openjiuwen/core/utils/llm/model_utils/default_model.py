# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import json
import asyncio
import time
from typing import List, Dict, Any, Iterator, AsyncIterator, Optional

import httpx
import aiohttp
import openai
from pydantic import ConfigDict
from requests import Session, Timeout, ConnectionError as RequestsConnectionError, HTTPError

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.logging import logger
from openjiuwen.core.common.security.ssl_utils import SslUtils
from openjiuwen.core.common.security.url_utils import UrlUtils
from openjiuwen.core.common.security.user_config import UserConfig
from openjiuwen.core.utils.llm.base import BaseModelClient
from openjiuwen.core.utils.llm.messages import AIMessage, UsageMetadata
from openjiuwen.core.utils.tool.schema import ToolCall
from openjiuwen.core.utils.llm.messages_chunk import AIMessageChunk


class RateLimiter:
    """全局请求速率限制器，用于控制 API 请求频率，避免触发并发限制"""

    _instance = None
    _lock = asyncio.Lock() if asyncio.get_event_loop_policy() else None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._last_request_time = 0.0
        self._min_interval = 1.0  # 最小请求间隔（秒）
        self._sync_lock = None
        self._async_lock = None

    def _get_sync_lock(self):
        """懒加载同步锁"""
        if self._sync_lock is None:
            import threading
            self._sync_lock = threading.Lock()
        return self._sync_lock

    def _get_async_lock(self):
        """懒加载异步锁"""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def wait_sync(self):
        """同步等待，确保请求间隔"""
        with self._get_sync_lock():
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                wait_time = self._min_interval - elapsed
                logger.debug(f"RateLimiter: waiting {wait_time:.2f}s before next request")
                time.sleep(wait_time)
            self._last_request_time = time.time()

    async def wait_async(self):
        """异步等待，确保请求间隔"""
        async with self._get_async_lock():
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                wait_time = self._min_interval - elapsed
                logger.debug(f"RateLimiter: waiting {wait_time:.2f}s before next request")
                await asyncio.sleep(wait_time)
            self._last_request_time = time.time()

    def set_min_interval(self, interval: float):
        """设置最小请求间隔（秒）"""
        self._min_interval = max(0.1, interval)


# 全局速率限制器实例
_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """获取全局速率限制器"""
    return _rate_limiter


class RequestChatModel(BaseModelClient):

    model_config = ConfigDict(arbitrary_types_allowed=True)
    sync_client: Session = Session()

    def __init__(self, api_key: str, api_base: str, max_retries: int = 3, timeout: int = 60, **kwargs):
        api_base = self._normalize_api_base(api_base)
        super().__init__(api_key=api_key, api_base=api_base, max_retries=max_retries, timeout=timeout, **kwargs)
        self._usage = dict()
        self._setup_ssl_adapter()

    @staticmethod
    def _normalize_api_base(api_base: str) -> str:
        """
        Normalize the api_base URL for Silicon Flow.
        Ensures the URL ends with /chat/completions.

        Args:
            api_base: The original API base URL

        Returns:
            The normalized API base URL with /chat/completions suffix
        """
        if not api_base:
            return api_base

        # Remove trailing slashes
        api_base = api_base.rstrip('/')

        # Check if it already ends with /chat/completions
        if not api_base.endswith('/chat/completions'):
            api_base = f"{api_base}/chat/completions"

        return api_base

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
        url_is_https = self.api_base.startswith("https://")
        messages = self.sanitize_tool_calls(messages)
        model_params = self._update_model_params(temperature=temperature, top_p=top_p, **kwargs)
        params = self._request_params(model_name=model_name, messages=messages, tools=tools, **model_params)

        ssl_verify, ssl_cert = SslUtils.get_ssl_config("LLM_SSL_VERIFY", "LLM_SSL_CERT",
                                                       ["false"], url_is_https)
        verify = ssl_cert if ssl_verify else False
        try:
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
            return self._parse_response(model_name, response.json())
        except (Timeout, TimeoutError) as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg="Generic API call timeout")
            ) from e
        except RequestsConnectionError as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg="Generic API connection failed")
            ) from e
        except HTTPError as e:
            status_code = e.response.status_code if hasattr(e, "response") else "unknown"
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg=f"Generic API error, status code is {status_code}")
            ) from e
        except Exception as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(error_msg="Generic API error")
            ) from e
        finally:
            self.close_session()

    async def _ainvoke(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                       temperature: Optional[float] = None, top_p: Optional[float] = None, **kwargs: Any) -> AIMessage:
        UrlUtils.check_url_is_valid(self.api_base)
        url_is_https = self.api_base.startswith("https://")
        messages = self.sanitize_tool_calls(messages)
        model_params = self._update_model_params(temperature=temperature, top_p=top_p, **kwargs)
        params = self._request_params(model_name=model_name, messages=messages, tools=tools, **model_params)
        ssl_verify, ssl_cert = SslUtils.get_ssl_config("LLM_SSL_VERIFY", "LLM_SSL_CERT",
                                                       ["false"], url_is_https)
        if ssl_verify:
            ssl_context = SslUtils.create_strict_ssl_context(ssl_cert)
            connector = aiohttp.TCPConnector(ssl=ssl_context)
        else:
            connector = aiohttp.TCPConnector(ssl=False)
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
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
        except (aiohttp.ConnectionTimeoutError, TimeoutError) as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg="Generic API async call timeout")
            ) from e
        except aiohttp.ClientConnectionError as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg="Generic API async connection failed")
            ) from e
        except aiohttp.ClientResponseError as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg=f"Generic API async error, status code is {e.status}")
            ) from e
        except Exception as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(error_msg="Generic API async error")
            ) from e

    def _stream(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                temperature: Optional[float] = None, top_p: Optional[float] = None, **kwargs: Any) -> Iterator[
        AIMessageChunk]:
        UrlUtils.check_url_is_valid(self.api_base)
        url_is_https = self.api_base.startswith("https://")
        messages = self.sanitize_tool_calls(messages)
        model_params = self._update_model_params(temperature=temperature, top_p=top_p, **kwargs)
        params = self._request_params(model_name=model_name, messages=messages, tools=tools, **model_params)
        params["stream"] = True
        ssl_verify, ssl_cert = SslUtils.get_ssl_config("LLM_SSL_VERIFY", "LLM_SSL_CERT",
                                                       ["false"], url_is_https)
        verify = ssl_cert if ssl_verify else False
        try:
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
        except (Timeout, TimeoutError) as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg="Generic API stream call timeout")
            ) from e
        except RequestsConnectionError as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg="Generic API stream connection failed")
            ) from e
        except HTTPError as e:
            status_code = e.response.status_code if hasattr(e, "response") else "unknown"
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg=f"Generic API stream error, status code is {status_code}")
            ) from e
        except Exception as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(error_msg="Generic API stream error")
            ) from e
        finally:
            self.close_session()


    async def _astream(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                       temperature: Optional[float] = None, top_p: Optional[float] = None,
                       **kwargs: Any) -> AsyncIterator[AIMessageChunk]:
        UrlUtils.check_url_is_valid(self.api_base)
        url_is_https = self.api_base.startswith("https://")

        messages = self.sanitize_tool_calls(messages)
        model_params = self._update_model_params(temperature=temperature, top_p=top_p, **kwargs)
        params = self._request_params(model_name=model_name, messages=messages, tools=tools, **model_params)
        params["stream"] = True

        ssl_verify, ssl_cert = SslUtils.get_ssl_config("LLM_SSL_VERIFY", "LLM_SSL_CERT",
                                                       ["false"], url_is_https)

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        # 429 重试配置
        max_429_retries = 3
        base_429_delay = 2.0  # 基础 429 重试延迟（秒）

        # 连接重试配置
        max_conn_retries = 2
        conn_retry_delay = 1.0

        for retry_429 in range(max_429_retries + 1):
            # 使用速率限制器控制请求频率
            await get_rate_limiter().wait_async()

            if ssl_verify:
                ssl_context = SslUtils.create_strict_ssl_context(ssl_cert)
                connector = aiohttp.TCPConnector(ssl=ssl_context)
            else:
                connector = aiohttp.TCPConnector(ssl=False)

            for attempt in range(max_conn_retries + 1):
                try:
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
                            # Check for HTTP errors before streaming
                            if response.status >= 400:
                                error_body = ""
                                try:
                                    error_body = await response.text()
                                except Exception:
                                    pass

                                # 检查是否是 429 错误（并发限制）
                                if response.status == 429 and retry_429 < max_429_retries:
                                    retry_delay = base_429_delay * (2 ** retry_429)
                                    logger.warning(f"=== 429 Rate Limit Hit (Generic API) ===")
                                    logger.warning(f"Retry attempt {retry_429 + 1}/{max_429_retries}")
                                    logger.warning(f"Error body: {error_body[:500] if error_body else 'N/A'}")
                                    logger.warning(f"Waiting {retry_delay:.1f}s before retry...")
                                    await asyncio.sleep(retry_delay)
                                    break  # 跳出连接重试循环，进入 429 重试循环

                                error_detail = f"status code is {response.status}"
                                if error_body:
                                    error_body_preview = error_body[:500] if len(error_body) > 500 else error_body
                                    error_detail += f", response: {error_body_preview}"
                                if not UserConfig.is_sensitive():
                                    logger.error(f"API request failed. Messages count: {len(messages)}, "
                                                f"Tools count: {len(tools) if tools else 0}, "
                                                f"Error: {error_detail}")
                                raise JiuWenBaseException(
                                    error_code=StatusCode.MODEL_CALL_FAILED.code,
                                    message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                                        error_msg=f"Generic API async stream error, {error_detail}")
                                )
                            # 使用正确的异步迭代方法读取流式响应
                            async for line_bytes, _ in response.content.iter_chunks():
                                if line_bytes:
                                    chunk = self._parse_stream_line(line_bytes)
                                    if chunk:
                                        yield chunk
                            # Successfully completed, exit all retry loops
                            return
                except (aiohttp.ClientConnectionError, aiohttp.ConnectionTimeoutError, TimeoutError) as e:
                    if attempt < max_conn_retries:
                        logger.warning(f"Connection failed (attempt {attempt + 1}/{max_conn_retries + 1}), "
                                      f"retrying in {conn_retry_delay}s... Error: {str(e)}")
                        await asyncio.sleep(conn_retry_delay)
                        conn_retry_delay *= 2
                        if ssl_verify:
                            connector = aiohttp.TCPConnector(ssl=ssl_context)
                        else:
                            connector = aiohttp.TCPConnector(ssl=False)
                        continue
                    if isinstance(e, (aiohttp.ConnectionTimeoutError, TimeoutError)):
                        raise JiuWenBaseException(
                            error_code=StatusCode.MODEL_CALL_FAILED.code,
                            message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                                error_msg="Generic API async stream call timeout")
                        ) from e
                    else:
                        raise JiuWenBaseException(
                            error_code=StatusCode.MODEL_CALL_FAILED.code,
                            message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                                error_msg="Generic API async stream connection failed")
                        ) from e
                except aiohttp.ClientResponseError as e:
                    error_detail = f"status code is {e.status}"
                    if hasattr(e, 'message') and e.message:
                        error_detail += f", message: {e.message}"
                    if not UserConfig.is_sensitive():
                        logger.error(f"API request failed. Messages count: {len(messages)}, "
                                    f"Tools count: {len(tools) if tools else 0}, "
                                    f"Error: {error_detail}")
                    raise JiuWenBaseException(
                        error_code=StatusCode.MODEL_CALL_FAILED.code,
                        message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                            error_msg=f"Generic API async stream error, {error_detail}")
                    ) from e
                except JiuWenBaseException:
                    raise
                except Exception as e:
                    raise JiuWenBaseException(
                        error_code=StatusCode.MODEL_CALL_FAILED.code,
                        message=StatusCode.MODEL_CALL_FAILED.errmsg.format(error_msg="Generic API async stream error")
                    ) from e
            else:
                # 连接重试循环正常结束（没有 break），说明请求成功或抛出了异常
                # 如果到这里说明是 429 重试，继续外层循环
                continue

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
                tool_call_dict = {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", "")
                    }
                }
                # Only add index if it has a value (avoid null/None in JSON)
                if tc.get("index") is not None:
                    tool_call_dict["index"] = tc.get("index")
                cleaned.append(tool_call_dict)
            msg["tool_calls"] = cleaned
        return messages

    def _request_params(self, model_name: str, messages: List[Dict], tools: List[Dict] = None, **kwargs: Any) -> Dict:
        # Truncate overly large tool results to prevent API errors
        MAX_TOOL_RESULT_LENGTH = 50000  # 50KB limit per tool result
        for msg in messages:
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
                if len(msg["content"]) > MAX_TOOL_RESULT_LENGTH:
                    msg["content"] = msg["content"][:MAX_TOOL_RESULT_LENGTH] + "\n...[内容已截断/Content truncated]"

        params = {
            "model": model_name,
            "messages": messages,
            **kwargs
        }

        if tools:
            params["tools"] = tools

        # Log request body size for debugging large requests
        try:
            request_size = len(json.dumps(params, ensure_ascii=False))
            if request_size > 50000:  # 50KB threshold
                logger.warning(f"Large request body detected: {request_size} bytes, "
                              f"messages: {len(messages)}, tools: {len(tools) if tools else 0}")
        except Exception:
            pass  # Ignore serialization errors in logging

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

    def _parse_stream_line(self, line: bytes) -> Optional[AIMessageChunk]:
        if line.startswith(b"data: "):
            line = line[6:]

        if line.strip() == b"[DONE]":
            chunk = AIMessageChunk(
                content="",
                reason_content="",
                tool_calls=[],
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
                    name_delta = function_delta.get("name", "")
                    args_delta = function_delta.get("arguments", "")

                    tool_calls.append(ToolCall(
                        id=tool_call_id or "",
                        type="function",
                        name=name_delta or "",
                        arguments=args_delta,
                        index=index
                    ))

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
                arguments=tool_call.get("function", {}).get("arguments", ""),
                index=tool_call.get("index"),
            ))
        return result


class OpenAIChatModel(BaseModelClient):
    """OpenAI-specific chat model implementation using official openai library"""

    def __init__(self,
                 api_key: str, api_base: str, max_retries: int = 3, timeout: int = 60, **kwargs):
        super().__init__(api_key=api_key, api_base=api_base, max_retries=max_retries, timeout=timeout, **kwargs)

    def model_provider(self) -> str:
        return "openai"

    def _invoke(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                temperature: Optional[float] = None, top_p: Optional[float] = None, **kwargs: Any) -> AIMessage:
        model_params = self._update_model_params(temperature=temperature, top_p=top_p, **kwargs)
        params = self._build_request_params(model_name=model_name, messages=messages, tools=tools, **model_params)
        sync_client = None
        try:
            url_is_https = self.api_base.startswith("https://")
            ssl_verify, ssl_cert = SslUtils.get_ssl_config("LLM_SSL_VERIFY", "LLM_SSL_CERT",
                                                           ["false"], url_is_https)

            if ssl_verify:
                ssl_context = SslUtils.create_strict_ssl_context(ssl_cert)
                http_client = httpx.Client(proxy=UrlUtils.get_global_proxy_url(self.api_base), verify=ssl_context)
            else:
                http_client = httpx.Client(proxy=UrlUtils.get_global_proxy_url(self.api_base), verify=None)
            sync_client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base, http_client=http_client,
                                        timeout=self.timeout, max_retries=0)
            response = sync_client.chat.completions.create(**params)
            return self._parse_openai_response(model_name, response)
        except (httpx.TimeoutException, openai.APITimeoutError) as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg="OpenAI API call timeout")
            ) from e
        except (httpx.ConnectError, openai.APIConnectionError) as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg="OpenAI API connection failed")
            ) from e
        except (httpx.HTTPStatusError, openai.APIStatusError) as e:
            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
            else:
                status_code = e.status_code
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg=f"OpenAI API error, status code is {status_code}")
            ) from e
        except Exception as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(error_msg="OpenAI API error")
            ) from e
        finally:
            if sync_client is not None:
                sync_client.close()

    async def _ainvoke(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                       temperature: Optional[float] = None, top_p: Optional[float] = None, **kwargs: Any) -> AIMessage:
        """Async call OpenAI API"""
        model_params = self._update_model_params(temperature=temperature, top_p=top_p, **kwargs)
        params = self._build_request_params(model_name=model_name, messages=messages, tools=tools, **model_params)
        async_client = None
        try:
            url_is_https = self.api_base.startswith("https://")
            ssl_verify, ssl_cert = SslUtils.get_ssl_config("LLM_SSL_VERIFY", "LLM_SSL_CERT",
                                                           ["false"], url_is_https)

            if ssl_verify:
                ssl_context = SslUtils.create_strict_ssl_context(ssl_cert)
                http_client = httpx.AsyncClient(proxy=UrlUtils.get_global_proxy_url(self.api_base), verify=ssl_context)
            else:
                http_client = httpx.AsyncClient(proxy=UrlUtils.get_global_proxy_url(self.api_base), verify=None)
            async_client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.api_base, http_client=http_client,
                                              timeout=self.timeout, max_retries=0)
            response = await async_client.chat.completions.create(**params)
            return self._parse_openai_response(model_name, response)
        except (httpx.TimeoutException, openai.APITimeoutError) as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg="OpenAI API async call timeout")
            ) from e
        except (httpx.ConnectError, openai.APIConnectionError) as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg="OpenAI API async connection failed")
            ) from e
        except (httpx.HTTPStatusError, openai.APIStatusError) as e:
            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
            else:
                status_code = e.status_code
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg=f"OpenAI API async error, status code is {status_code}")
            ) from e
        except Exception as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(error_msg="OpenAI API async error")
            ) from e
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
            url_is_https = self.api_base.startswith("https://")
            ssl_verify, ssl_cert = SslUtils.get_ssl_config("LLM_SSL_VERIFY", "LLM_SSL_CERT",
                                                           ["false"], url_is_https)

            if ssl_verify:
                ssl_context = SslUtils.create_strict_ssl_context(ssl_cert)
                http_client = httpx.Client(proxy=UrlUtils.get_global_proxy_url(self.api_base), verify=ssl_context)
            else:
                http_client = httpx.Client(proxy=UrlUtils.get_global_proxy_url(self.api_base), verify=None)
            sync_client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base, http_client=http_client,
                                        timeout=self.timeout, max_retries=0)
            stream = sync_client.chat.completions.create(**params)
            for chunk in stream:
                parsed_chunk = self._parse_openai_stream_chunk(model_name, chunk)
                if parsed_chunk:
                    yield parsed_chunk
        except (httpx.TimeoutException, openai.APITimeoutError) as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg="OpenAI API stream call timeout")
            ) from e
        except (httpx.ConnectError, openai.APIConnectionError) as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg="OpenAI API stream connection failed")
            ) from e
        except (httpx.HTTPStatusError, openai.APIStatusError) as e:
            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
            else:
                status_code = e.status_code
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                    error_msg=f"OpenAI API stream error, status code is {status_code}")
            ) from e
        except Exception as e:
            raise JiuWenBaseException(
                error_code=StatusCode.MODEL_CALL_FAILED.code,
                message=StatusCode.MODEL_CALL_FAILED.errmsg.format(error_msg="OpenAI API stream error")
            ) from e
        finally:
            if sync_client is not None:
                sync_client.close()

    async def _astream(self, model_name: str, messages: List[Dict], tools: List[Dict] = None,
                       temperature: Optional[float] = None, top_p: Optional[float] = None,
                       **kwargs: Any) -> AsyncIterator[AIMessageChunk]:
        """Async stream call OpenAI API with rate limiting and retry"""
        model_params = self._update_model_params(temperature=temperature, top_p=top_p, **kwargs)
        params = self._build_request_params(model_name=model_name, messages=messages, tools=tools, stream=True,
                                            **model_params)

        # 重试配置
        max_retries = 3
        base_retry_delay = 2.0  # 基础重试延迟（秒）

        for retry_attempt in range(max_retries + 1):
            async_client = None
            try:
                # 使用速率限制器控制请求频率
                await get_rate_limiter().wait_async()

                url_is_https = self.api_base.startswith("https://")
                ssl_verify, ssl_cert = SslUtils.get_ssl_config("LLM_SSL_VERIFY", "LLM_SSL_CERT",
                                                               ["false"], url_is_https)

                if ssl_verify:
                    ssl_context = SslUtils.create_strict_ssl_context(ssl_cert)
                    http_client = httpx.AsyncClient(proxy=UrlUtils.get_global_proxy_url(self.api_base), verify=ssl_context)
                else:
                    http_client = httpx.AsyncClient(proxy=UrlUtils.get_global_proxy_url(self.api_base), verify=None)
                async_client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.api_base, http_client=http_client,
                                                  timeout=self.timeout, max_retries=0)
                stream = await async_client.chat.completions.create(**params)
                async for chunk in stream:
                    parsed_chunk = self._parse_openai_stream_chunk(model_name, chunk)
                    if parsed_chunk:
                        yield parsed_chunk
                # 成功完成，退出重试循环
                return
            except (httpx.TimeoutException, openai.APITimeoutError) as e:
                # 超时错误也重试
                if retry_attempt < max_retries:
                    retry_delay = base_retry_delay * (2 ** retry_attempt)
                    logger.warning(f"=== API Timeout, retrying ===")
                    logger.warning(f"Retry attempt {retry_attempt + 1}/{max_retries}")
                    logger.warning(f"Waiting {retry_delay:.1f}s before retry...")
                    if async_client is not None:
                        await async_client.close()
                        async_client = None
                    await asyncio.sleep(retry_delay)
                    continue
                raise JiuWenBaseException(
                    error_code=StatusCode.MODEL_CALL_FAILED.code,
                    message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                        error_msg="OpenAI API async stream call timeout")
                ) from e
            except (httpx.ConnectError, openai.APIConnectionError) as e:
                # 连接错误重试
                if retry_attempt < max_retries:
                    retry_delay = base_retry_delay * (2 ** retry_attempt)
                    logger.warning(f"=== API Connection Failed, retrying ===")
                    logger.warning(f"Retry attempt {retry_attempt + 1}/{max_retries}")
                    logger.warning(f"Error: {str(e)[:200]}")
                    logger.warning(f"Waiting {retry_delay:.1f}s before retry...")
                    if async_client is not None:
                        await async_client.close()
                        async_client = None
                    await asyncio.sleep(retry_delay)
                    continue
                raise JiuWenBaseException(
                    error_code=StatusCode.MODEL_CALL_FAILED.code,
                    message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                        error_msg="OpenAI API async stream connection failed")
                ) from e
            except (httpx.HTTPStatusError, openai.APIStatusError) as e:
                if isinstance(e, httpx.HTTPStatusError):
                    status_code = e.response.status_code
                    error_body = ""
                    try:
                        error_body = e.response.text
                    except Exception:
                        pass
                else:
                    status_code = e.status_code
                    error_body = str(e.body) if hasattr(e, 'body') else ""

                # 检查是否是 429 错误（并发限制）或 5xx 服务器错误
                if (status_code == 429 or status_code >= 500) and retry_attempt < max_retries:
                    # 指数退避重试
                    retry_delay = base_retry_delay * (2 ** retry_attempt)
                    logger.warning(f"=== API Error {status_code}, retrying ===")
                    logger.warning(f"Retry attempt {retry_attempt + 1}/{max_retries}")
                    logger.warning(f"Error body: {error_body[:500] if error_body else 'N/A'}")
                    logger.warning(f"Waiting {retry_delay:.1f}s before retry...")

                    # 关闭当前客户端
                    if async_client is not None:
                        await async_client.close()
                        async_client = None

                    # 等待后重试
                    await asyncio.sleep(retry_delay)
                    continue

                # 非可重试错误或重试次数用尽，记录错误并抛出异常
                logger.error(f"=== API Error Debug ===")
                logger.error(f"Status code: {status_code}")
                logger.error(f"Error body: {error_body[:2000] if error_body else 'N/A'}")
                logger.error(f"Request had {len(messages)} messages, {len(tools) if tools else 0} tools")

                # 保存错误详情到文件
                try:
                    timestamp = int(time.time())
                    error_file = f"/tmp/jiuwen_api_error_{timestamp}.json"
                    error_data = {
                        "status_code": status_code,
                        "error_body": error_body,
                        "messages_count": len(messages),
                        "tools_count": len(tools) if tools else 0,
                        "messages": messages,
                    }
                    with open(error_file, 'w', encoding='utf-8') as f:
                        json.dump(error_data, f, ensure_ascii=False, indent=2)
                    logger.error(f"Error details saved to: {error_file}")
                except Exception as save_err:
                    logger.error(f"Failed to save error details: {save_err}")

                raise JiuWenBaseException(
                    error_code=StatusCode.MODEL_CALL_FAILED.code,
                    message=StatusCode.MODEL_CALL_FAILED.errmsg.format(
                        error_msg=f"OpenAI API async stream error, status code is {status_code}")
                ) from e
            except Exception as e:
                raise JiuWenBaseException(
                    error_code=StatusCode.MODEL_CALL_FAILED.code,
                    message=StatusCode.MODEL_CALL_FAILED.errmsg.format(error_msg="OpenAI API async stream error")
                ) from e
            finally:
                if async_client is not None:
                    await async_client.close()


    def _build_request_params(self, model_name: str, messages: List[Dict],
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

        # 添加详细调试日志
        try:
            import time
            request_json = json.dumps(params, ensure_ascii=False)
            request_size = len(request_json.encode('utf-8'))

            # 打印每条消息的大小
            msg_sizes = []
            for i, msg in enumerate(messages):
                msg_json = json.dumps(msg, ensure_ascii=False)
                msg_size = len(msg_json.encode('utf-8'))
                role = msg.get('role', 'unknown')
                # 对于 tool 消息，显示 tool_call_id
                if role == 'tool':
                    tool_call_id = msg.get('tool_call_id', 'N/A')[:8]
                    msg_sizes.append(f"msg[{i}]({role}, id={tool_call_id}): {msg_size} bytes")
                # 对于 assistant 消息，显示是否有 tool_calls
                elif role == 'assistant':
                    has_tool_calls = 'tool_calls' in msg and msg['tool_calls']
                    tc_count = len(msg.get('tool_calls', [])) if has_tool_calls else 0
                    msg_sizes.append(f"msg[{i}]({role}, tc={tc_count}): {msg_size} bytes")
                else:
                    msg_sizes.append(f"msg[{i}]({role}): {msg_size} bytes")

            logger.warning(f"=== API Request Debug ===")
            logger.warning(f"Total request size: {request_size} bytes")
            logger.warning(f"Messages count: {len(messages)}")
            logger.warning(f"Tools count: {len(tools) if tools else 0}")
            logger.warning(f"Message sizes: {msg_sizes}")

            # 如果请求过大，保存完整内容到文件
            if request_size > 50000:  # 50KB
                timestamp = int(time.time())
                debug_file = f"/tmp/jiuwen_api_request_{timestamp}.json"
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(request_json)
                logger.warning(f"Large request ({request_size} bytes) saved to: {debug_file}")
        except Exception as e:
            logger.error(f"Debug logging failed: {e}")

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
            for idx, tc in enumerate(message.tool_calls):
                function_name = getattr(getattr(tc, 'function', None), 'name', None) or ""
                function_arguments = getattr(getattr(tc, 'function', None), 'arguments', None) or ""
                tool_call = ToolCall(
                    id=getattr(tc, 'id', '') or "",
                    type="function",
                    name=function_name,
                    arguments=function_arguments,
                    index=getattr(tc, 'index', idx)
                )
                tool_calls.append(tool_call)

        reasoning_content = getattr(message, 'reasoning_content', "")
        
        return AIMessage(
            content=message.content or "",
            tool_calls=tool_calls,
            usage_metadata=UsageMetadata(
                model_name=model_name,
                finish_reason=choice.finish_reason or "",
                total_latency=response.usage.total_tokens if response.usage else 0
            ),
            reason_content=reasoning_content
        )


    def _parse_openai_stream_chunk(self, model_name, chunk) -> Optional[AIMessageChunk]:
        """Parse OpenAI stream response chunk"""
        if not chunk.choices:
            return None

        choice = chunk.choices[0]
        delta = choice.delta

        content = getattr(delta, 'content', None) or ""
        reasoning_content = getattr(delta, 'reasoning_content', "")
        tool_calls = []

        # Handle tool call delta
        if hasattr(delta, 'tool_calls') and delta.tool_calls:
            for tc_delta in delta.tool_calls:
                if hasattr(tc_delta, 'function') and tc_delta.function:
                    index = getattr(tc_delta, 'index', None)
                    
                    function_name = getattr(tc_delta.function, 'name', None) or ""
                    function_arguments = getattr(tc_delta.function, 'arguments', None) or ""
                    tool_call = ToolCall(
                        id=getattr(tc_delta, 'id', '') or "",
                        type="function",
                        name=function_name,
                        arguments=function_arguments,
                        index=index
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
            reason_content=reasoning_content,
            tool_calls=tool_calls,
            usage_metadata=usage_metadata
        )
