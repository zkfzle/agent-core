# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import asyncio
import random
import time
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

import httpx

from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.generation.graph.custom_types import JSONLike


def default_error_handling(resp: Union[httpx.Response, Any], attempt: int, should_retry: bool) -> Tuple[int, bool, Any]:
    """Default Error Handling"""
    should_retry = True
    return attempt, should_retry, None


CallbackFunction = Callable[[Union[httpx.Response, Any], int, bool], Tuple[int, bool, Optional[JSONLike]]]
HANDLE_ERR_CODE: MappingProxyType[int, CallbackFunction] = MappingProxyType(
    {
        429: default_error_handling,
        500: default_error_handling,
        503: default_error_handling,
    }
)
ERROR_TEMPLATE = "%s Service Error%s: %s"


def sync_request_with_retry(
    client: Union[httpx.Client, Any],
    max_retry: int = 30,
    retry_wait: float = 0.1,
    custom_callback: Mapping[int, CallbackFunction] = HANDLE_ERR_CODE,
    task: str = "LLM Chat",
    **kwargs,
) -> Optional[JSONLike]:
    """Send api requests with retries (sync)"""
    attempt = 0
    should_retry = False

    resp_str = "Unknown error"
    last_exception = Exception("Unknown Error")
    response = None
    for backoff in range(1, max_retry + 1):
        if should_retry:
            time.sleep(random.random() * retry_wait * backoff)
            should_retry = False
        try:
            response = client.post(**kwargs)
            resp_json, resp_str = _handle_response(response)
        except Exception as e:
            resp_str = str(e)
            last_exception = e
            should_retry = True
            continue
        attempt, should_retry, result = _handle_response_by_status(
            response, attempt, should_retry, resp_json, resp_str, task, custom_callback
        )
        if result is not None:
            return result
    logger.error(ERROR_TEMPLATE, task, "", resp_str)
    if response:
        response.raise_for_status()
    else:
        raise last_exception
    return result


async def async_request_with_retry(
    client: Union[httpx.AsyncClient, Any],
    max_retry: int = 30,
    retry_wait: float = 0.1,
    custom_callback: Mapping[int, CallbackFunction] = HANDLE_ERR_CODE,
    task: str = "LLM Chat",
    **kwargs,
) -> Optional[JSONLike]:
    """Send api requests with retries (async)"""
    attempt = 0
    should_retry = False

    resp_str = "Unknown error"
    last_exception = Exception("Unknown Error")
    response = None
    for backoff in range(1, max_retry + 1):
        if should_retry:
            await asyncio.sleep(random.random() * retry_wait * backoff)
            should_retry = False
        try:
            response = await client.post(**kwargs)
            resp_json, resp_str = _handle_response(response)
        except Exception as e:
            resp_str = str(e)
            last_exception = e
            should_retry = True
            continue
        attempt, should_retry, result = _handle_response_by_status(
            response, attempt, should_retry, resp_json, resp_str, task, custom_callback
        )
        if result is not None:
            return result
    logger.error(ERROR_TEMPLATE, task, "", resp_str)
    if response:
        response.raise_for_status()
    else:
        raise last_exception
    return result


def _handle_response(response: Union[httpx.Response, Any]) -> Tuple[JSONLike, str]:
    if not response.text.startswith("{") and response.text.endswith("}"):
        raise ValueError("Empty response")
    resp_json = response.json()
    resp_str = f"{response.text=}"
    return resp_json, resp_str


def _handle_response_by_status(
    response: Union[httpx.Response, Any],
    attempt: int,
    should_retry: bool,
    resp_json: Dict,
    resp_str: str,
    task: str,
    custom_callback: Mapping[int, CallbackFunction],
) -> Tuple[int, bool, Optional[JSONLike]]:
    match response.status_code:
        case 200:
            return attempt, should_retry, resp_json
        case 400:
            attempt += 1
            attempt_str = f" ({attempt=})"
            logger.error(ERROR_TEMPLATE, task, attempt_str, resp_str)
            try:
                resp_json = resp_json.get("error", resp_json)
                magic_words = ["safety", "violation", "policy", "inspection", "appropriate"]
                error_code = resp_json.get("code", "") + resp_json.get("message", "") + resp_json.get("content", "")
                error_code = error_code.casefold()
                is_censored = any(k in error_code for k in magic_words)
                if is_censored:
                    raise ValueError("Censor word detected")
                response.raise_for_status()
            except Exception:
                """Ignore exception"""
        case _:
            callback = custom_callback.get(response.status_code, default_error_handling)
            attempt, should_retry, result = callback(response, attempt, should_retry)
            if result is not None:
                return attempt, should_retry, result
    return attempt, should_retry, None
