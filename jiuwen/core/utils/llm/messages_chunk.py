#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import Any

from jiuwen.core.utils.llm.messages import BaseMessage, AIMessage, ToolMessage


class BaseMessageChunk(BaseMessage):
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {type(None): lambda _: None}

    def __add__(self, other: "BaseMessageChunk") -> "BaseMessageChunk":
        if not isinstance(other, BaseMessageChunk):
            raise TypeError(f"Cannot add {self.__class__.__name__} to {type(other)}")

        if isinstance(self.content, str) and isinstance(other.content, str):
            combined_content = self.content + other.content
        elif isinstance(self.content, list) and isinstance(other.content, list):
            combined_content = self.content + other.content
        else:
            combined_content = other.content


        return self.__class__(role=self.role, content=combined_content, name=self.name or other.name)


class AIMessageChunk(AIMessage, BaseMessageChunk):
    def __add__(self, other: Any) -> "AIMessageChunk":
        if not isinstance(other, AIMessageChunk):
            raise TypeError(f"Cannot add AIMessageChunk to {type(other)}")

        return AIMessageChunk(
            role=self.role,
            content=(self.content or "") + (other.content or ""),
            name=other.name or self.name,
            tool_calls=(self.tool_calls or []) + (other.tool_calls or []),
            usage_metadata=other.usage_metadata or self.usage_metadata,
            raw_content=other.raw_content or self.raw_content,
            reason_content=other.reason_content or self.reason_content
        )


class ToolMessageChunk(ToolMessage, BaseMessageChunk):
    def __add__(self, other: Any) -> "ToolMessageChunk":
        if not isinstance(other, ToolMessageChunk):
            raise TypeError(f"Cannot add ToolMessageChunk to {type(other)}")

        return ToolMessageChunk(
            role="tool",
            content=(self.content or "") + (other.content or ""),
            name=other.name or self.name,
            tool_call_id=other.tool_call_id or self.tool_call_id
        )