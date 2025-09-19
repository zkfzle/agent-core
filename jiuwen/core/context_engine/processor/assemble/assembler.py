#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import Any, Dict, List, Union
from pydantic import Field

from jiuwen.core.context_engine.utils import ContextUtils
from jiuwen.core.context_engine.base import ContextWindow, ContextType
from jiuwen.core.context_engine.processor.base import BaseContextProcessor
from jiuwen.core.context_engine.config import BaseProcessorConfig
from jiuwen.core.context_engine.processor.factory import ProcessorFactory
from jiuwen.core.utils.prompt.assemble.assembler import Assembler
from jiuwen.core.utils.llm.messages import BaseMessage
from jiuwen.core.common.logging import logger
from jiuwen.core.utils.prompt.template.template import Template


class AssemblerConfig(BaseProcessorConfig):
    """Configuration for AssemblerProcessor"""

    processor_type: str = "assembler"
    template_content: Union[str, List[Dict], List[BaseMessage]] = Field(
        default="",
        description="Template content for prompt assembler. Can be string, message list, or dict list",
    )
    variable_mappings: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from context field names to template variable names",
    )
    default_values: Dict[str, Any] = Field(
        default_factory=dict, description="Default values for template variables"
    )


@ProcessorFactory.register("assembler", AssemblerConfig)
class AssemblerProcessor(BaseContextProcessor):
    """Processor that uses the existing Assembler class for context assembly"""

    def __init__(self, config: AssemblerConfig):
        super().__init__(config)
        self.assembler = self._initialize_assembler()
        self.variable_mappings = config.variable_mappings
        self.default_values = config.default_values

    def run(self, context_window: ContextWindow) -> ContextWindow:
        """Assemble context using the configured template and variables"""
        try:
            # Extract variables from input data
            prompt_content = context_window.prompt.content if isinstance(context_window.prompt, Template) \
                else context_window.prompt
            if context_window.prompt:
                self.assembler = Assembler(
                    template_content=prompt_content
                )

            template_variables = self._extract_template_variables(context_window)

            # Assemble the prompt using the existing Assembler with validation
            try:
                assembled_content = self.assembler.assemble(**template_variables)
            except Exception as e:
                logger.warning(
                    f"Assembler failed with variables {template_variables}: {str(e)}"
                )
                # Fallback to basic template assembly
                assembled_content = self._assemble_fallback(template_variables)

            output = context_window
            output.full_prompt = self._process_assemble_output(assembled_content)
            return output

        except Exception as e:
            logger.error(f"AssemblerProcessor failed: {str(e)}")
            # Return original input on failure
            return context_window

    def _initialize_assembler(self) -> Assembler:
        """Initialize the Assembler instance with template configuration"""
        try:
            return Assembler(
                template_content=self.config.template_content
            )
        except Exception as e:
            logger.error(f"Failed to initialize Assembler: {str(e)}")
            raise

    def _extract_template_variables(self, context_window: ContextWindow) -> Dict[str, Any]:
        """Extract variables from EngineInput based on mappings"""
        variables = {}
        variables.update(self.default_values)

        if (
            context_window.variables
            and ContextType.VARIABLES.value not in self.variable_mappings.values()
            and ContextType.VARIABLES.value not in self.variable_mappings.keys()
        ):
            variables.update(ContextUtils.convert_variables_to_dict(context_window.variables))

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

    def _process_assemble_output(self, output: Any) -> Union[str, BaseMessage, List[BaseMessage]]:
        if isinstance(output, str):
            return output
        assemble_output = []
        for message in output:
            if isinstance(message, BaseMessage):
                assemble_output.append(message)
            else:
                assemble_output.append(ContextUtils.convert_dict_to_message(message))
        return assemble_output
