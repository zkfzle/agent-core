from typing import Any, Iterator, Union

from jiuwen.core.utils.llm.messages import AIMessage
from jiuwen.core.utils.llm.messages_chunk import AIMessageChunk


class BaseOutputParser:
    """Base class for output parsers."""

    async def parse(self, inputs: Union[str, AIMessage]) -> Any:
        """convert content into its expected format"""
        raise NotImplementedError()

    async def stream_parse(self, streaming_inputs: Union[Iterator[str], Iterator[AIMessageChunk]]) -> Iterator[Any]:
        """parse in the streaming manner"""
        raise NotImplementedError()
