# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import Callable, Optional, Union


from openjiuwen.core.foundation.tool.function.function import LocalFunction, ToolCard
from openjiuwen.core.foundation.tool.utils.callable_schema_extractor import CallableSchemaExtractor


def tool(func: Optional[Callable] = None, *, card: Optional[ToolCard] = None) -> Union[Callable, LocalFunction]:
    """
    Decorator to convert a regular function into a LocalFunction with a tool card.

    Usage:
        1. @tool
        2. @tool(card=ToolCard(...))
        3. @tool(card=ToolCard(name="custom_name", description="...", input_params={...}))

    Args:
        func: Function to be decorated
        card: Optional tool card, if not provided will be automatically extracted from the function

    Returns:
        LocalFunction object or decorator function
    """

    def decorator(func_: Callable) -> LocalFunction:
        # Get function name
        func_name = func_.__name__

        # If a card is provided, use it
        if card:
            # Ensure card name matches function name (unless specifically overridden)
            if card.name != func_name:
                # Could issue a warning here or keep the card name as is
                pass
            return LocalFunction(card=card, func=func_)

        # Otherwise, automatically extract information to create a card
        # Generate input parameter schema
        input_params = CallableSchemaExtractor.generate_schema(func_)

        # Extract function description
        description = CallableSchemaExtractor.extract_function_description(func_)

        # Create tool card
        new_card = ToolCard(
            name=func_name,
            description=description,
            input_params=input_params
        )

        return LocalFunction(card=new_card, func=func_)

    # If called directly as @tool
    if func is not None:
        return decorator(func)

    # If called as @tool() or @tool(card=...)
    return decorator
