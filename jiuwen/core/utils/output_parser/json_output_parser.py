import json
import re
import asyncio
from typing import Any, Iterator, Optional, Union, Dict

from jiuwen.core.utils.output_parser.base import BaseOutputParser
from jiuwen.core.utils.llm.messages import AIMessage
from jiuwen.core.utils.llm.messages_chunk import AIMessageChunk
from jiuwen.core.common.logging import logger


class JsonOutputParser(BaseOutputParser):
    """
    JsonOutputParser
    """

    async def parse(self, llm_output: Union[str, AIMessage]) -> Any:
        """
        parse
        """
        if isinstance(llm_output, AIMessage):
            text = llm_output.content
        elif isinstance(llm_output, str):
            text = llm_output
        else:
            logger.warning(f"Unsupported llm_output type for parse: {type(llm_output)}")
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
            logger.error(f"Failed to decode JSON from LLM output: {e}\nContent: {json_str}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during JSON parsing: {e}\nContent: {json_str}")
            return None

    async def stream_parse(self, streaming_inputs: Iterator[Union[str, AIMessageChunk]]) -> Iterator[Optional[Dict[str, Any]]]:
        """
        stream_parse json
        """
        buffer = ""
        for chunk in streaming_inputs:
            await asyncio.sleep(0.001)

            if isinstance(chunk, AIMessageChunk):
                if chunk.content:
                    buffer += chunk.content
            elif isinstance(chunk, str):
                buffer += chunk
            else:
                logger.warning(f"Unsupported chunk type for stream_parse: {type(chunk)}")
                continue

            match = re.search(r"```json\n(.*?)```", buffer, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                try:
                    parsed_data = json.loads(json_str)
                    yield parsed_data
                    buffer = buffer[match.end():].strip()
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.error(
                        f"An unexpected error occurred during streaming JSON parsing: {e}\nContent: {json_str}")
                    buffer = ""
            elif buffer.strip().startswith("{") and buffer.strip().endswith("}"):
                try:
                    parsed_data = json.loads(buffer.strip())
                    yield parsed_data
                    buffer = ""
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.error(
                        f"An unexpected error occurred during streaming JSON parsing (direct): {e}\nContent: {buffer}")
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
                logger.warning(f"Remaining buffer could not be fully parsed as JSON: {e}\nContent: {json_str}")
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred during final streaming JSON parsing: {e}\nContent: {json_str}")
