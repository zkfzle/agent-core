# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import threading
from typing import Any, Dict, List, Optional, Self, Tuple

import httpx
from pydantic import BaseModel

from openjiuwen.core.utils.llm.messages import AIMessage, BaseMessage, UsageMetadata
from openjiuwen.core.utils.prompt.template.template import Template

from .api_requests import async_request_with_retry, sync_request_with_retry


def get_usage_data(response_data: Dict[str, Any]) -> Tuple[int, int]:
    """Get model usage statistics

    Args:
        response_data (Dict[str, Any]): _description_

    Returns:
        Dict[str, int]: _description_
    """
    usage_data = response_data.get("usage", response_data)
    tokens_in = usage_data.get("input_tokens", 0) + usage_data.get("prompt_tokens", 0)
    tokens_out = usage_data.get("output_tokens", 0) + usage_data.get("completion_tokens", 0)
    return tokens_in, tokens_out


class GraphLLMClient:
    """LLM Client with support for token-tracking, structured output, API request retry"""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_base: str,
        temperature: float = 0.6,
        top_p: float = 1.0,
        timeout: float = 60.0,
        structured_output: bool = True,
        max_retries: int = 30,
        retry_wait: float = 0.1,
        **kwargs,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.timeout = timeout
        self.top_p = top_p
        self.structured_output = structured_output
        self.extra_args = kwargs

    @classmethod
    def from_config(cls, llm_config: BaseModel) -> Self:
        """Create an LLM service instance via config"""
        config_fields = ["model_name", "api_key", "api_base", "temperature", "top_p", "timeout", "structured_output"]
        if "JIUWEN_GRAPH_MEM_RERANK_MODE" in llm_config.extras:
            del llm_config.extras["JIUWEN_GRAPH_MEM_RERANK_MODE"]
        return cls(**llm_config.model_dump(include=config_fields), **llm_config.extras)

    def assemble_invoke_params(self, kwargs: Dict, template: Template, output_model: Optional[Dict] = None) -> Dict:
        """Assemble invoke parameters"""
        params = dict(messages=template.format(kwargs).content)
        if self.structured_output and output_model:
            params["response_format"] = output_model
        return params

    async def ainvoke(self, messages: List[Dict], tools: Optional[List[Dict]] = None, **kwargs: Dict) -> BaseMessage:
        """Asynchronous invoke LLM service"""
        async_client: httpx.AsyncClient = kwargs.pop("async_client", None) or httpx.AsyncClient(
            verify=False, timeout=self.timeout
        )
        max_retry = kwargs.pop("max_retry", 30)
        retry_wait = kwargs.pop("retry_wait", 0.1)
        token_record: Dict = kwargs.pop("token_record", {})
        record_lock: threading.Lock = kwargs.pop("record_lock", threading.Lock())

        params = self._request_params(messages=messages, tools=tools, **self.extra_args, **kwargs)
        headers = self._request_headers(**kwargs)
        response = await async_request_with_retry(
            client=async_client,
            max_retry=max_retry,
            retry_wait=retry_wait,
            task="LLM Chat",
            timeout=self.timeout,
            url=self.api_base,
            json=params,
            headers=headers,
        )

        if token_record:
            tokens_in, tokens_out = get_usage_data(response.get("usage", {}))
            with record_lock:
                token_record["input_tokens"] += tokens_in
                token_record["output_tokens"] += tokens_out
        return self._parse_response(response)

    def invoke(self, messages: List[Dict], tools: Optional[List[Dict]] = None, **kwargs: Dict) -> BaseMessage:
        """Synchronous invoke LLM service"""
        client: httpx.Client = kwargs.pop("client", None) or httpx.Client(verify=False, timeout=self.timeout)
        max_retry = kwargs.pop("max_retry", 30)
        retry_wait = kwargs.pop("retry_wait", 0.1)
        token_record: Dict = kwargs.pop("token_record", {})
        record_lock: threading.Lock = kwargs.pop("record_lock", threading.Lock())

        params = self._request_params(messages=messages, tools=tools, **self.extra_args, **kwargs)
        headers = self._request_headers(**kwargs)
        response = sync_request_with_retry(
            client=client,
            max_retry=max_retry,
            retry_wait=retry_wait,
            task="LLM Chat",
            timeout=self.timeout,
            url=self.api_base,
            json=params,
            headers=headers,
        )

        if token_record:
            tokens_in, tokens_out = get_usage_data(response.get("usage", {}))
            with record_lock:
                token_record["input_tokens"] += tokens_in
                token_record["output_tokens"] += tokens_out
        return self._parse_response(response)

    def _request_headers(self, **kwargs: Dict):
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    def _request_params(self, **kwargs: Dict):
        return dict(temperature=self.temperature, top_p=self.top_p, model=self.model_name, **kwargs)

    def _parse_response(self, response_data: Dict) -> AIMessage:
        choice: Dict[str, Any] = response_data.get("choices", [{}])[0]
        message: Dict = choice.get("message", {})
        content = message.get("content") or ""
        tokens_in, tokens_out = get_usage_data(response_data)
        usage_meta = UsageMetadata(
            model_name=self.model_name,
            finish_reason=choice.get("finish_reason", ""),
            model_stats=dict(input_tokens=tokens_in, output_tokens=tokens_out),
        )
        return AIMessage(
            content=content,
            tool_calls=message.get("tool_calls", []),
            usage_metadata=usage_meta,
        )
