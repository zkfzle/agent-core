import json
import re
from typing import List

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.utils.llm.messages import BaseMessage

MESSAGE_VALIDATION_SCHEMA = {
    "system": {
        "role": str,
        "content": str
    },
    "assistant": {
        "role": str,
        "content": (type(None), str),
        "function_call": (type(None), dict)
    },
    "user": {
        "role": str,
        "content": str
    },
    "function": {
        "role": str,
        "content": str,
        "name": str
    }
}

EXTRA_VALIDATION_SCHEMA = {
    "function_call": {
        "name": str,
        "arguments": str
    }
}


def messages_to_template(messages: List[dict]) -> str:
    """messages to template"""
    template = ""
    for message in messages:
        if isinstance(message, BaseMessage):
            message = message.__dict__
            message.pop("name", None)
        if not isinstance(message, dict):
            raise JiuWenBaseException(
                error_code=StatusCode.PROMPT_ASSEMBLER_TEMPLATE_FORMAT_ERROR.code,
                message="Each message in the template must be a dict"
            )
        role = message.get("role")
        validate_schema = MESSAGE_VALIDATION_SCHEMA.get(role)
        if not validate_schema:
            raise JiuWenBaseException(
                error_code=StatusCode.PROMPT_ASSEMBLER_TEMPLATE_FORMAT_ERROR.code,
                message=f"No validation schema found for message role `{role}`."
            )
        validate(message, validate_schema)
        content = message.get("content")
        if not content:
            content = ""
        template += f"`#{role}#`\n{content}\n"
        for extra_key in set(message.keys()) - {"role", "content"}:
            if isinstance(message[extra_key], str):
                extra_content = message[extra_key]
            elif isinstance(message[extra_key], dict):
                extra_content = json.dumps(message['function_call'], ensure_ascii=False)
            else:
                raise JiuWenBaseException(
                    error_code=StatusCode.PROMPT_ASSEMBLER_TEMPLATE_FORMAT_ERROR.code,
                    message="Cannot parse data into string"
                )
            template += f"`*{extra_key}*`\n{extra_content}\n"

    return template


def template_to_messages(template: str) -> List[dict]:
    """template to messages"""
    messages = []
    message_prefix_matches = list(re.finditer(r'`#(system|assistant|user|tool|function)#`', template))
    for message_index, message_match in enumerate(message_prefix_matches):
        message_content, message_prefix, validation_schema = get_message(
            message_index, message_match, message_prefix_matches, template
        )
        message = padding_message(message_prefix, message_content, validation_schema)
        validate(message, validation_schema)
        messages.append(message)
    return messages


def get_message(message_index, message_match, message_prefix_matches, template):
    """get message"""
    message_prefix = message_match.group(1)
    message_start = message_match.end()
    if message_index < len(message_prefix_matches) - 1:
        message_end = message_prefix_matches[message_index + 1].start()
    else:
        message_end = len(template)
    message_content = template[message_start:message_end].strip()
    validation_schema = MESSAGE_VALIDATION_SCHEMA.get(message_prefix)
    if not validation_schema:
        raise JiuWenBaseException(
            error_code=StatusCode.PROMPT_ASSEMBLER_TEMPLATE_FORMAT_ERROR.code,
            message=f"No validation schema found for message role `{message_prefix}`."
        )
    return message_content, message_prefix, validation_schema


def validate(data: dict, schema: dict):
    """validate data"""
    if len(set(data.keys()) - set(schema.keys())) > 0:
        raise JiuWenBaseException(
            error_code=StatusCode.PROMPT_ASSEMBLER_TEMPLATE_FORMAT_ERROR.code,
            message="Failed validate the data against the schema."
        )
    for name, data_type in schema.items():
        if not isinstance(data.get(name), data_type):
            raise JiuWenBaseException(
                error_code=StatusCode.PROMPT_ASSEMBLER_TEMPLATE_FORMAT_ERROR.code,
                message=f"Failed validate the data against the schema `{name}`."
            )
        if name in EXTRA_VALIDATION_SCHEMA and data.get(name) is not None:
            validate(data.get(name), EXTRA_VALIDATION_SCHEMA.get(name))


def padding_message(message_prefix, message_content, validation_schema):
    """message padding"""
    key_role = "role"
    key_content = "content"
    message = {
        key_role: message_prefix,
        key_content: message_content
    }
    extra_fields_matches = list(re.finditer(r'`\*(name|function_call)\*`', message_content))
    for field_index, field_match in enumerate(extra_fields_matches):
        field_name = field_match.group(1)
        field_start = field_match.end()
        if field_index < len(extra_fields_matches) - 1:
            field_end = extra_fields_matches[field_index + 1].start()
        else:
            field_end = len(message_content)
        field_content = message_content[field_start:field_end].strip()
        try:
            data_type = validation_schema.get(field_name)
            if (isinstance(data_type, tuple) and dict in data_type) or dict == data_type:
                field_content = json.loads(field_content)
        except Exception as e:
            raise JiuWenBaseException(
                error_code=StatusCode.PROMPT_ASSEMBLER_TEMPLATE_FORMAT_ERROR.code,
                message=f"Errors occur when parsing field `{field_name}` into dict."
            ) from e
        message[field_name] = field_content
        validate(message, MESSAGE_VALIDATION_SCHEMA.get(message_prefix))
        if field_index == 0:
            message[key_content] = message_content[:field_match.start()].strip()
            if len(message[key_content]) == 0:
                message[key_content] = None
    return message

