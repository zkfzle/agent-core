from typing import Any, Dict

from jiuwen.core.context_engine.engine import EngineInput, EngineOutput
from jiuwen.core.context_engine.processor.factory import ProcessorFactory
from jiuwen.core.context_engine.processor.preprocess.base import (
    CompressorConfig,
    PreprocessStage,
)
from jiuwen.core.utils.llm.base import BaseChatModel, BaseModelInfo
from jiuwen.core.utils.llm.model_utils.model_factory import ModelFactory
from jiuwen.core.utils.llm.messages import BaseMessage
from jiuwen.core.common.logging import logger


class LLMCompressorConfig(CompressorConfig):
    """Configuration for LLMCompressor"""

    processor_type: str = "llm_compressor"
    model_type: str = "default"
    model_config: Dict[str, Any] = {}
    compression_prompt: str = (
        "Please compress the following content, retaining key information:"
    )
    max_length: int = 500
    content_format: str = "{label}: {content}"
    labels: Dict[str, str] = {
        "user_input": "User Input",
        "chat_history": "Chat History",
        "system_input": "System Input",
    }


@ProcessorFactory.register("llm_compressor", CompressorConfig)
class LLMCompressor(PreprocessStage):
    """Processor that uses LLM for content compression"""

    def __init__(self, config: LLMCompressorConfig):
        super().__init__(config)
        self.llm_client = self._initialize_llm_client()
        # Access config attributes directly
        self.compression_prompt = config.compression_prompt
        self.max_length = config.max_length
        self.content_format = config.content_format
        self.labels = config.labels

    def _initialize_llm_client(self) -> BaseChatModel:
        model_factory = ModelFactory()
        # Get config from parent class - it should be LLMCompressorConfig
        config = self._BaseProcessor__config
        model_type = config.model_type
        model_config = config.model_config

        # Convert model_config dict to BaseModelInfo
        model_info = BaseModelInfo(**model_config)

        return model_factory.get_model(model_type, model_info)

    def _should_compress(self, content: str) -> bool:
        """Determine if content needs compression"""
        return len(content) > self.max_length

    def run(self, input_data: EngineInput) -> EngineOutput:
        """Synchronous version of process for pipeline execution"""
        content_to_compress = self._extract_content(input_data)
        if not self._should_compress(content_to_compress):
            return EngineOutput(full_output=content_to_compress)
        
        # Run compression and handle any errors
        try:
            compressed_content = self._compress_with_llm(content_to_compress)
            return EngineOutput(full_output=compressed_content)
        except Exception:
            # If compression fails, return original content
            return EngineOutput(full_output=content_to_compress)

    def _compress_with_llm(self, content: str) -> str:
        """Compress content using LLM"""
        try:
            messages = [
                BaseMessage(role="system", content=self.compression_prompt),
                BaseMessage(role="user", content=content),
            ]
            response = self.llm_client.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(
                f"LLM compression failed with exception {e}, truncating content."
            )
            # If the model compression fails, truncate the data
            return content[: self.max_length] + "..."

    def _extract_content(self, engine_input: EngineInput) -> str:
        """Extract content to be compressed from EngineInput"""
        contents = []

        if engine_input.user_input:
            contents.append(
                self.content_format.format(
                    label=self.labels["user_input"], content=engine_input.user_input
                )
            )

        if engine_input.chat_history:
            contents.append(
                self.content_format.format(
                    label=self.labels["chat_history"], content=engine_input.chat_history
                )
            )

        if engine_input.system_prompt:
            contents.append(
                self.content_format.format(
                    label=self.labels["system_input"],
                    content=engine_input.system_prompt,
                )
            )

        return "\n".join(contents)
