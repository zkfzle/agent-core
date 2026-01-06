# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import Dict, Any, Optional, Union
from jsonschema import validate
from pydantic import BaseModel, Field
from openjiuwen.core.common.schema.card import BaseCard


class ToolCard(BaseCard):
    parameters: Union[Dict[str, Any], BaseModel] = Field(default_factory=dict)

    def remove_nulls_recursive(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(inputs, dict):
            return {k: self.remove_nulls_recursive(v) for k, v in inputs.items() if v is not None}
        elif isinstance(inputs, list):
            return [self.remove_nulls_recursive(v) for v in inputs if v is not None]
        else:
            return inputs

    def validate_inputs(self, origin_inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs = self.remove_nulls_recursive(origin_inputs)
        if not inputs:
            return {}

        if isinstance(self.parameters, dict):
            schema = self.parameters
            props = schema.get("properties", {})
            required = schema.get("required", [])
            try:
                validate(inputs, schema)
            except Exception as e:
                raise ValueError(f"Invalid inputs for tool {self.name}: {e}")

            result = {}
            for key, prop_schema in props.items():
                if key in inputs:
                    result[key] = inputs[key]
                else:
                    default_value = prop_schema.get("default", None)
                    if default_value is not None:
                        try:
                            validate(default_value, prop_schema)
                            result[key] = default_value
                        except Exception as e:
                            raise ValueError(f"Invalid default value for tool {self.name} parameter {key}: {e}")
                    elif key in required:
                        raise ValueError(f"Missing required parameter {key} for tool {self.name}")
            return result
        elif isinstance(self.parameters, BaseModel):
            model_class = self.parameters
            try:
                model_class.model_validate(inputs)
            except Exception as e:
                raise ValueError(f"Invalid inputs for tool {self.name}: {e}")

            result = {}
            fields = model_class.model_fields
            for field_name, field_info in fields.items():
                if field_name in inputs:
                    result[field_name] = inputs[field_name]
                else:
                    default_value = field_info.get("default", None)
                    default_factory = field_info.get("default_factory", None)
                    if default_factory is not None:
                        default_value = default_factory()
                    if default_value is not None:
                        result[field_name] = default_value
                    elif field_info.get("required", False):
                        raise ValueError(f"Field {field_name} is required")
            return result
        else:
            raise TypeError(f"Unsupported type: {type(self.parameters)}")


class ToolInfo(BaseModel):
    type: str = Field(default="function")
    name: str = Field(default="")
    description: str = Field(default="")
    parameters: Union[Dict[str, Any], BaseModel] = Field(default_factory=dict)


class ToolCall(BaseModel):
    id: Optional[str]
    type: str
    name: str
    arguments: str
    index: Optional[int] = None
