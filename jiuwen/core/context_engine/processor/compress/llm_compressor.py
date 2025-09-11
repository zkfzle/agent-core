import re
import json
from typing import Optional, Dict, List
from pydantic import Field

from jiuwen.core.context_engine.base import ContextWindow, ContextType
from jiuwen.core.context_engine.processor.factory import ProcessorFactory
from jiuwen.core.context_engine.processor.base import BaseContextProcessor
from jiuwen.core.utils.llm.messages import HumanMessage
from jiuwen.core.common.logging import logger
from jiuwen.core.context_engine.config import BaseProcessorConfig


DEFAULT_COMPRESS_TEMPLATE = \
"""
You are a context compression assistant. Please compress the given content as required.
Context types to be compressed:
{{context_types}}

Content to be compressed:
{{context_contents}}

1. The compressed content must maintain the same meaning as the original context, be concise and fluent, and should not include any additional content.
2. If the compressed content contains special characters such as double quotes that are not compatible with JSON, please convert them into a JSON parsable format.

Output the compressed content in JSON format, target format:
{{output_format}}

Ensure that the output format strictly adheres to JSON format and always includes the ```json``` tag.

"""

OUTPUT_JSON_TEMPLATE = \
"""
```json
{
{{output_format}}
}
```
"""

DEFAULT_COMPRESS_MAX_LENGTH = 2048


class LLMCompressorConfig(BaseProcessorConfig):
    """Configuration for LLMCompressor"""
    processor_type: str = "llm_compressor"
    compression_prompt: Optional[str] = Field(default=DEFAULT_COMPRESS_TEMPLATE)
    max_length: int = Field(default=DEFAULT_COMPRESS_MAX_LENGTH, ge=0)
    compress_targets: List[ContextType] = Field(default=[])


@ProcessorFactory.register("llm_compressor", LLMCompressorConfig)
class LLMCompressor(BaseContextProcessor):
    """Processor that uses LLM for content compression"""

    def __init__(self, config: LLMCompressorConfig):
        super().__init__(config)
        # Access config attributes directly
        self.compression_prompt = config.compression_prompt
        self.max_length = config.max_length
        self.compress_targets = config.compress_targets

    def run(self, input_data: ContextWindow) -> ContextWindow:
        """Synchronous version of process for pipeline execution"""
        input_format = self._generate_input_content(input_data)
        output_format = self._generate_output_format()
        prompt = self.compression_prompt.replace("{{context_types}}", str(self.compress_targets)) \
                                        .replace("{{context_contents}}", input_format) \
                                        .replace("{{output_format}}", output_format)
        
        # Run compression and handle any errors
        compressed_contents = self._compress_with_llm(prompt)
        return self._fill_compressed_result(compressed_contents, input_data)


    def _compress_with_llm(self, prompt: str) -> Dict[str, str]:
        """Compress content using LLM"""
        try:
            messages = [
                HumanMessage(content=prompt),
            ]
            response = self._llm.invoke(messages)
            return self._parse_compressed_result(response.content)
        except Exception as e:
            logger.error(
                f"LLM compression failed with exception {e}, truncating content."
            )
            # If the model compression fails, truncate the data
            return dict()

    def _generate_input_content(self,
                                input_data: ContextWindow) -> Optional[str]:
        input_data_dict = input_data.model_dump()
        format_data = ""
        for context_type in self.compress_targets:
            content = str(input_data_dict.get(context_type.value, ""))
            format_data += f"{context_type}:\n{content}\n"
        return format_data

    def _generate_output_format(self) -> str:
        output_json_format = dict()
        for context_type in self.compress_targets:
            output_json_format[context_type.value] = f"compressed {context_type.value} content"
        return OUTPUT_JSON_TEMPLATE.replace("{{output_format}}",
            "\n".join(f"    \"{target}\": \"{content}\"" for target, content in output_json_format.items())
        )

    def _fill_compressed_result(self,
                                compressed_content: Dict[str, str],
                                input_data: ContextWindow) -> ContextWindow:
        output_data = input_data.model_dump()
        for context_type in self.compress_targets:
            if context_type.value in compressed_content:
                output_data[context_type.value] = compressed_content[context_type.value]
        return ContextWindow(**output_data)

    def _parse_compressed_result(self, json_str: str) -> Optional[Dict[str, str]]:
        """Parse json string"""
        if not json_str:
            return None
        pattern = r"```json(.*?)```"
        match = re.search(pattern, json_str, re.DOTALL)

        if match:
            json_string = match.group(1).strip()
            try:
                parsed_data = json.loads(json_string)
            except json.decoder.JSONDecodeError:
                logger.warning("Failed to decode json string")
                return None
            return parsed_data

        logger.warning("No valid json string found")
        return None