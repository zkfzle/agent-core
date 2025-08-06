import re
from typing import Union, List, Dict

from pydantic import BaseModel, Field


from jiuwen.core.common.logging import logger
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.utils.llm.messages import BaseMessage
from jiuwen.core.utils.prompt.assemble.assembler import Assembler
from jiuwen.core.utils.prompt.assemble.message_handler import template_to_messages
from jiuwen.core.utils.prompt.assemble.variables.textable import TEMPLATE_VARIABLE_PLACEHOLDER_PATTERN


class Template(BaseModel):
    """
    template data

    """
    name: str = Field(default='')
    content: Union[List[Dict], List[BaseMessage], str]
    filters: dict = Field(default=None)

    def to_messages(self) -> List[BaseMessage]:
        """Return Template as a list of Messages."""
        messages = []
        if self.content is None or len(self.content) == 0:
            self.content = []
            return messages
        if isinstance(self.content, str):
            self.content = template_to_messages(self.content)

        for msg in self.content:
            if isinstance(msg, BaseMessage):
                messages.append(msg)
            elif isinstance(msg, dict):
                messages.append(BaseMessage(**msg))
            else:
                raise JiuWenBaseException(
                    error_code=StatusCode.PROMPT_TEMPLATE_INCORRECT_ERROR.code,
                    message=f"Template type must be in str, list[dict] or list[BaseMessage]."
                )
        self.content = messages
        self._validate_template_content_assembled()
        return self.content

    def format(self, keywords: dict = None):
        """format prompt"""
        assembler = Assembler(self.content)
        input_keys = assembler.input_keys
        format_dict = {}
        for key in input_keys:
            if keywords and keywords.get(key):
                format_dict[key] = keywords.get(key)
        self.content = assembler.assemble(**format_dict)
        return self

    def _validate_template_content_assembled(self):
        if isinstance(self.content, str):
            placeholder_matches = re.findall(TEMPLATE_VARIABLE_PLACEHOLDER_PATTERN, self.content)
            if placeholder_matches:
                logger.warning(f"template content has not assembled "
                            f"with variable placeholders: {', '.join(placeholder_matches)}")
            return
        for message in self.content:
            content = message.content if isinstance(message, BaseMessage) else message.get('content', '')
            placeholder_matches = re.findall(TEMPLATE_VARIABLE_PLACEHOLDER_PATTERN, content)
            if placeholder_matches:
                logger.warning(f"template content has not assembled "
                            f"with variable placeholders: {', '.join(placeholder_matches)}")
        return
