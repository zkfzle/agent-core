#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from abc import abstractmethod
from typing import AsyncIterator

from openjiuwen.core.foundation.tool.schema import ToolInfo
from openjiuwen.core.foundation.tool.constant import Input, Output


class Tool:
    """tool class that defined the data types and content for LLM modules"""
    def __init__(self):
        pass

    @abstractmethod
    async def invoke(self, inputs: Input, **kwargs) -> Output:
        """Execute the tool with provided inputs and return final result.

        This method performs complete tool execution in a single call,
        processing all inputs and returning the final output when the
        operation is fully completed.

        Args:
            inputs: Structured input data conforming to the tool's input schema
            **kwargs: Additional execution parameters such as timeout,
                     retry policies, or tool-specific options

        Returns:
            Output: The complete result of tool execution

        """
        pass

    @abstractmethod
    async def stream(self, inputs: Input, **kwargs) -> AsyncIterator[Output]:
        """Execute the tool and stream incremental results.

        This method supports long-running operations by yielding partial
        results as they become available, enabling real-time processing
        and progress tracking.

        Args:
            inputs: Structured input data conforming to the tool's input schema
            **kwargs: Additional execution parameters for streaming behavior

        Yields:
            Output: Incremental results during tool execution

        """
        pass

    @abstractmethod
    def get_tool_info(self) -> ToolInfo:
        """get tool info"""
        pass
