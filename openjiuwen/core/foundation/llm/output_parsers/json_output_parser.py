# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import json
import re
from typing import Any, AsyncIterator, Optional, Union, Dict

from openjiuwen.core.common.security.user_config import UserConfig
from openjiuwen.core.foundation.llm.schema.message import AssistantMessage
from openjiuwen.core.foundation.llm.output_parsers.output_parser import BaseOutputParser
from openjiuwen.core.foundation.llm.schema.message_chunk import AssistantMessageChunk
from openjiuwen.core.common.logging import llm_logger, LogEventType


class JsonOutputParser(BaseOutputParser):
    """
    JsonOutputParser
    """

    async def parse(self, llm_output: Union[str, AssistantMessage]) -> Any:
        """
        parse
        """
        if isinstance(llm_output, AssistantMessage):
            text = llm_output.content
        elif isinstance(llm_output, str):
            text = llm_output
        else:
            if UserConfig.is_sensitive():
                llm_logger.warning(
                    "Unsupported llm_output type for parse.",
                    event_type=LogEventType.LLM_PARSE_ERROR
                )
            else:
                llm_logger.warning(
                    f"Unsupported llm_output type for parse: {type(llm_output)}",
                    event_type=LogEventType.LLM_PARSE_ERROR
                )
            return None

        if not text:
            return None

        match = re.search(r"```json\n(.*?)```", text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            json_str = text.strip()

        try:
            parsed_data = json.loads(json_str)
            return parsed_data
        except json.JSONDecodeError as e:
            if UserConfig.is_sensitive():
                llm_logger.error(
                    "Failed to decode JSON from LLM output",
                    event_type=LogEventType.LLM_PARSE_ERROR
                )
            else:
                llm_logger.error(
                    f"Failed to decode JSON from LLM output: {e}\nContent: {json_str}",
                    event_type=LogEventType.LLM_PARSE_ERROR
                )
            return None
        except Exception as e:
            if UserConfig.is_sensitive():
                llm_logger.error(
                    "An unexpected error occurred during JSON parsing",
                    event_type=LogEventType.LLM_PARSE_ERROR
                )
            else:
                llm_logger.error(
                    f"An unexpected error occurred during JSON parsing: {e}\nContent: {json_str}",
                    event_type=LogEventType.LLM_PARSE_ERROR
                )
            return None

    async def stream_parse(self, streaming_inputs: AsyncIterator[Union[str, AssistantMessageChunk]]) -> AsyncIterator[
        Optional[Dict[str, Any]]]:
        """
        stream_parse json
        """
        buffer = ""
        async for chunk in streaming_inputs:
            if isinstance(chunk, AssistantMessageChunk):
                if chunk.content:
                    buffer += chunk.content
            elif isinstance(chunk, str):
                buffer += chunk
            else:
                if UserConfig.is_sensitive():
                    llm_logger.warning(
                        "Unsupported chunk type for stream_parse.",
                        event_type=LogEventType.LLM_STREAM_PARSE_ERROR
                    )
                else:
                    llm_logger.warning(
                        f"Unsupported chunk type for stream_parse: {type(chunk)}",
                        event_type=LogEventType.LLM_STREAM_PARSE_ERROR
                    )
                continue

            match = re.search(r"```json\n(.*?)```", buffer, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                try:
                    parsed_data = json.loads(json_str)
                    yield parsed_data
                    buffer = buffer[match.end():].strip()
                except json.JSONDecodeError as e:
                    if UserConfig.is_sensitive():
                        llm_logger.error(
                            "An unexpected error occurred during streaming JSON parsing",
                            event_type=LogEventType.LLM_STREAM_PARSE_ERROR
                        )
                    else:
                        llm_logger.error(
                            f"An unexpected error occurred during streaming JSON parsing: {e}\nContent: {json_str}",
                            event_type=LogEventType.LLM_STREAM_PARSE_ERROR
                        )

                except Exception as e:
                    if UserConfig.is_sensitive():
                        llm_logger.error(
                            "An unexpected error occurred during streaming JSON parsing",
                            event_type=LogEventType.LLM_STREAM_PARSE_ERROR
                        )
                    else:
                        llm_logger.error(
                            f"An unexpected error occurred during streaming JSON parsing: {e}\nContent: {json_str}",
                            event_type=LogEventType.LLM_STREAM_PARSE_ERROR
                        )
                    buffer = ""
            elif buffer.strip().startswith("{") and buffer.strip().endswith("}"):
                try:
                    parsed_data = json.loads(buffer.strip())
                    yield parsed_data
                    buffer = ""
                except json.JSONDecodeError as e:
                    if UserConfig.is_sensitive():
                        llm_logger.error(
                            "An unexpected error occurred during streaming JSON parsing",
                            event_type=LogEventType.LLM_STREAM_PARSE_ERROR
                        )
                    else:
                        llm_logger.error(
                            f"An unexpected error occurred during streaming JSON parsing: {e}\nContent: {json_str}",
                            event_type=LogEventType.LLM_STREAM_PARSE_ERROR
                        )

                except Exception as e:
                    if UserConfig.is_sensitive():
                        llm_logger.error(
                            "An unexpected error occurred during streaming JSON parsing (direct)",
                            event_type=LogEventType.LLM_STREAM_PARSE_ERROR
                        )
                    else:
                        llm_logger.error(
                            f"An unexpected error occurred during streaming JSON parsing (direct): {e}\n"
                            f"Content: {buffer}",
                            event_type=LogEventType.LLM_STREAM_PARSE_ERROR
                        )
                    buffer = ""

        if buffer.strip():
            match = re.search(r"```json\n(.*?)```", buffer, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                json_str = buffer.strip()

            try:
                parsed_data = json.loads(json_str)
                yield parsed_data
            except json.JSONDecodeError as e:
                if UserConfig.is_sensitive():
                    llm_logger.warning(
                        "Remaining buffer could not be fully parsed as JSON",
                        event_type=LogEventType.LLM_STREAM_PARSE_ERROR
                    )
                else:
                    llm_logger.warning(
                        f"Remaining buffer could not be fully parsed as JSON: {e}\nContent: {json_str}",
                        event_type=LogEventType.LLM_STREAM_PARSE_ERROR
                    )
            except Exception as e:
                if UserConfig.is_sensitive():
                    llm_logger.error(
                        "An unexpected error occurred during final streaming JSON parsing",
                        event_type=LogEventType.LLM_STREAM_PARSE_ERROR
                    )
                else:
                    llm_logger.error(
                        f"An unexpected error occurred during final streaming JSON parsing: {e}\nContent: {json_str}",
                        event_type=LogEventType.LLM_STREAM_PARSE_ERROR
                    )
