import json
import re
from typing import Dict, Any

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode


class WorkflowLLMUtils:
    """工作流LLM组件相关工具类"""

    @staticmethod
    def extract_content(response) -> str:
        return response.content if hasattr(response, "content") else str(response)


class ValidationUtils:
    """验证工具类"""

    @staticmethod
    def raise_invalid_params_error(error_msg: str = "") -> None:
        """抛出参数无效错误"""
        raise JiuWenBaseException(
            StatusCode.PROMPT_JSON_SCHEMA_ERROR.code,
            StatusCode.PROMPT_JSON_SCHEMA_ERROR.errmsg.format(error_msg=error_msg),
        )

    @staticmethod
    def validate_type(instance: Any, expected_type: str) -> None:
        """校验schema json实例类型：object,array,string,integer,boolean, and number types"""
        type_validators = {
            "object": lambda x: isinstance(x, dict),
            "array": lambda value: isinstance(value, list),
            "string": lambda value: isinstance(value, str),
            "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
            "boolean": lambda value: isinstance(value, bool),
            "number": lambda value: isinstance(value, int) and not isinstance(value, bool),
        }

        validator = type_validators.get(expected_type)
        if not validator:
            ValidationUtils.raise_invalid_params_error(error_msg=f"{expected_type} is not a valid type")

        if not validator(instance):
            ValidationUtils.raise_invalid_params_error(
                error_msg=f"expected type {expected_type} but got {type(instance)}")

    @staticmethod
    def validate_json_schema(instance: Any, schema: Dict[str, Any]) -> None:
        """对json schema实例进行schema校验type类型和缺失字段"""
        if "type" not in schema:
            ValidationUtils.raise_invalid_params_error("schema must have 'type' key")
        # 验证类型
        ValidationUtils.validate_type(instance=instance, expected_type=schema["type"])

        # 验证对象属性
        if schema["type"] == "object":
            ValidationUtils._validate_object_properties(instance, schema)

        # 验证数组元素
        elif schema["type"] == "array":
            ValidationUtils._validate_array_items(instance, schema)

    @staticmethod
    def _validate_object_properties(instance: Any, schema: Dict[str, Any]) -> None:
        if "properties" not in schema:
            return
        # 检查必填字段
        required_fields = schema.get("required", [])
        missing_fields = [field for field in required_fields if field not in instance]
        if missing_fields:
            ValidationUtils.raise_invalid_params_error(f"missing required properties {missing_fields}")
        # 递归验证每个属性
        for prop_name, prop_schema in schema["properties"].items():
            if prop_name in instance:
                ValidationUtils.validate_json_schema(instance=instance[prop_name], schema=prop_schema)

    @staticmethod
    def _validate_array_items(instance: Any, schema: Dict[str, Any]) -> None:
        if "items" not in schema:
            return

        # 递归验证每个数组元素
        for i, item in enumerate(instance):
            try:
                ValidationUtils.validate_json_schema(instance=item, schema=schema["items"])
            except JiuWenBaseException as e:
                original_msg = str(e)
                ValidationUtils.raise_invalid_params_error(f"invalid array item {i}: {original_msg}")

    @staticmethod
    def validate_outputs_config(outputs_config: Any) -> None:
        """验证输出配置参数"""
        if not outputs_config:
            ValidationUtils.raise_invalid_params_error("outputs config must not be empty")
        if not isinstance(outputs_config, dict):
            ValidationUtils.raise_invalid_params_error("outputs config must be a dict")


class SchemaGenerator:
    @staticmethod
    def generate_json_schema(outputs_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        properties = {}
        required = []

        for field_name, field_config in outputs_config.items():
            properties[field_name] = {
                "type": field_config.get("type", "string"),
                "description": field_config.get("description", "")
            }

            if field_config.get("type") == "array" and "items" in field_config:
                properties[field_name]["items"] = field_config["items"]

            if field_config.get("required", True):
                required.append(field_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }


class JsonParser:
    @staticmethod
    def parse_json_content(response_content: str) -> Dict[str, Any]:
        content = JsonParser._clean_markdown_blocks(response_content)

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            ValidationUtils.raise_invalid_params_error(f"JSON解析失败: {response_content} 不是有效的JSON格式")

    @staticmethod
    def _clean_markdown_blocks(content: str):
        content = content.strip()

        if not (content.startswith("```") and content.endswith("```")):
            return content

        lines = content.split("\n")

        if lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1] == "```":
            lines = lines[:-1]

        return '\n'.join(lines).strip()


class OutputFormatter:
    @staticmethod
    def format_response(response_content: str, response_format: dict, outputs_config: dict) -> dict:
        response_type = response_format.get("type")
        ValidationUtils.validate_outputs_config(outputs_config)

        formatters = {
            "text": OutputFormatter._format_text_response,
            "markdown": OutputFormatter._format_text_response,
            "json": OutputFormatter._format_json_response
        }

        formatter = formatters.get(response_type)
        if not formatter:
            ValidationUtils.raise_invalid_params_error(f"不支持的响应类型: '{response_type}'")

        return formatter(response_content, outputs_config)

    @staticmethod
    def _format_text_response(response_content: str, outputs_config: dict) -> dict:
        if len(outputs_config) != 1:
            ValidationUtils.raise_invalid_params_error(f"文本/markdown响应类型的outputs_config必须只包含一个字段")
        field_name = next(iter(outputs_config))
        return {field_name: response_content}

    @staticmethod
    def _format_json_response(response_content: str, outputs_config: dict) -> dict:
        if not outputs_config:
            ValidationUtils.raise_invalid_params_error(f"文本/markdown响应类型的outputs_config必须至少包含一个字段")

        parsed_json = JsonParser.parse_json_content(response_content)
        json_schema = SchemaGenerator.generate_json_schema(outputs_config)
        OutputFormatter._validate_json_schema(parsed_json, json_schema, response_content)

        return OutputFormatter._extract_configured_fields(parsed_json, outputs_config)

    @staticmethod
    def _validate_json_schema(parsed_json: dict, json_schema: dict, original_content: str) -> None:
        try:
            ValidationUtils.validate_json_schema(parsed_json, json_schema)
        except JiuWenBaseException:
            raise
        except Exception as e:
            ValidationUtils.raise_invalid_params_error(f"JSON schema验证失败: {original_content}")

    @staticmethod
    def _extract_configured_fields(parsed_json: dict, outputs_config: dict) -> dict:
        output = {}
        missing_keys = []

        for field_name, field_config in outputs_config.items():
            if field_name not in parsed_json:
                if field_config.get("required", True):
                    missing_keys.append(field_name)
            else:
                output[field_name] = parsed_json[field_name]

        if missing_keys:
            ValidationUtils.raise_invalid_params_error(f"响应中缺少必填字段: {', '.join(missing_keys)}")

        return output

class TemplateUtils:

    @staticmethod
    def render_template(template: str, inputs: dict) -> str:
        pattern = re.compile(r'\{\{(\w+)}}')

        # 替换所有匹配的变量
        return pattern.sub(lambda match: str(inputs.get(match.group(1), match.group(0))), template)

    @staticmethod
    def render_template_to_list(template: str) -> list[str | Any]:

        # 替换所有匹配的变量
        return re.split(r'(\{\{[^}]+\}\})', template)
