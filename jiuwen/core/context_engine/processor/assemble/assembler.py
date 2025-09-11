#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import Any, Dict, List, Union
from pydantic import Field

from jiuwen.core.context_engine.base import EngineInput, EngineOutput
from jiuwen.core.context_engine.processor.assemble.base import AssembleStage
from jiuwen.core.context_engine.config import BaseProcessorConfig
from jiuwen.core.context_engine.processor.factory import ProcessorFactory
from jiuwen.core.utils.prompt.assemble.assembler import Assembler
from jiuwen.core.utils.llm.messages import BaseMessage
from jiuwen.core.common.logging import logger


class AssemblerConfig(BaseProcessorConfig):
    """Configuration for AssemblerProcessor"""

    processor_type: str = "assembler"
    template_content: Union[str, List[Dict], List[BaseMessage]] = Field(
        default="",
        description="Template content for prompt assembler. Can be string, message list, or dict list",
    )
    return_format: str = Field(
        default="message",
        description="Output format: 'message' for message list, 'text' for string",
    )
    variable_mappings: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from context field names to template variable names",
    )
    default_values: Dict[str, Any] = Field(
        default_factory=dict, description="Default values for template variables"
    )


@ProcessorFactory.register("assembler", AssemblerConfig)
class AssemblerProcessor(AssembleStage):
    """Processor that uses the existing Assembler class for context assembly"""

    def __init__(self, config: AssemblerConfig):
        super().__init__(config)
        self.assembler = self._initialize_assembler()
        self.variable_mappings = config.variable_mappings
        self.default_values = config.default_values

    def _initialize_assembler(self) -> Assembler:
        """Initialize the Assembler instance with template configuration"""
        try:
            return Assembler(
                template_content=self.config.template_content,
                return_format=self.config.return_format,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Assembler: {str(e)}")
            raise

    def _extract_template_variables(self, engine_input: EngineInput) -> Dict[str, Any]:
        """Extract variables from EngineInput based on mappings"""
        variables = {}
        variables.update(self.default_values)
        input_dict = (
            engine_input.model_dump() if hasattr(engine_input, "model_dump") else {}
        )

        for context_field, template_var in self.variable_mappings.items():
            if context_field in input_dict and input_dict[context_field] is not None:
                variables[template_var] = input_dict[context_field]

        if (
            engine_input.user_input
            and "user_input" not in self.variable_mappings.values()
            and "user_input" not in self.variable_mappings.keys()
        ):
            variables.setdefault("user_input", engine_input.user_input)

        if (
            engine_input.chat_history
            and "chat_history" not in self.variable_mappings.values()
            and "chat_history" not in self.variable_mappings.keys()
        ):
            variables.setdefault("chat_history", engine_input.chat_history)

        if (
            engine_input.system_prompt
            and "system_prompt" not in self.variable_mappings.values()
            and "system_prompt" not in self.variable_mappings.keys()
        ):
            variables.setdefault("system_prompt", engine_input.system_prompt)

        return variables

    def _assemble_fallback(self, template_variables: Dict[str, Any]) -> str:
        """Fallback assembly method when main assembler fails"""
        if isinstance(self.config.template_content, str):
            template = self.config.template_content
            for key, value in template_variables.items():
                if value is not None:
                    template = template.replace(f"{{{key}}}", str(value))
            return template
        return ""

    def run(self, input_data: EngineInput) -> EngineOutput:
        """Assemble context using the configured template and variables"""
        try:
            # Extract variables from input data
            template_variables = self._extract_template_variables(input_data)

            # Assemble the prompt using the existing Assembler with validation
            try:
                assembled_content = self.assembler.assemble(**template_variables)
            except Exception as e:
                logger.warning(
                    f"Assembler failed with variables {template_variables}: {str(e)}"
                )
                # Fallback to basic template assembly
                assembled_content = self._assemble_fallback(template_variables)

            # Include all context information
            if isinstance(input_data.chat_history, str) and input_data.chat_history:
                history_str = input_data.chat_history
            elif isinstance(input_data.chat_history, list) and input_data.chat_history:
                history_str = "\n".join(
                    [f"[{msg.role}]:{msg.content}" for msg in input_data.chat_history]
                )
            else:
                history_str = ""
            # Handle different return formats consistently
            if isinstance(assembled_content, list):
                # Convert message list to string format
                assembled_content = "\n".join(
                    [
                        f"[{msg.get('role', 'unknown')}]: {msg.get('content', '')}"
                        for msg in assembled_content
                    ]
                )

            # Append history and user input to assembled content
            if assembled_content:
                if history_str:
                    assembled_content += f"\nhistory:\n{history_str}\n"
                if input_data.user_input:
                    assembled_content += f"query: {input_data.user_input}\n"

            output = EngineOutput.from_input(input_data)
            output.full_output = assembled_content
            return output

        except Exception as e:
            logger.error(f"AssemblerProcessor failed: {str(e)}")
            # Return original input on failure
            return EngineOutput.from_input(input_data)
