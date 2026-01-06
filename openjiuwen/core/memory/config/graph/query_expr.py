# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""Query Expression Classes, provide high-level abstraction for combining queries together"""

from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Callable, List, Literal, Optional, Sequence, Set, Union

from pydantic import BaseModel, Field

from .milvus_query_func import (
    milvus_arithmetic_filter,
    milvus_array_filter,
    milvus_comparison_filter,
    milvus_json_filter,
    milvus_logical_filter,
    milvus_null_filter,
    milvus_range_filter,
    milvus_text_match_filter,
)

__query_language_register_lock = Lock()


class QueryLanguageDefinition(BaseModel):
    comparison: Callable[["BaseFilter"], str]
    range: Callable[["BaseFilter"], str]
    arithmetic: Callable[["BaseFilter"], str]
    null: Callable[["BaseFilter"], str]
    json_filter: Callable[["BaseFilter"], str]
    array: Callable[["BaseFilter"], str]
    logical: Callable[["BaseFilter"], str]
    text_match: Callable[["BaseFilter"], str]


def register_database_query_language(name: str, definition: QueryLanguageDefinition, force: bool = False):
    """Register query language definition for a database"""
    with __query_language_register_lock:
        if name in QUERY_EXPR_FUNCTIONS and not force:
            raise ValueError(f"Database query language for {name=} already registered")
        QUERY_EXPR_FUNCTIONS[name] = definition


def validate_language_registered(name: str):
    if name not in QUERY_EXPR_FUNCTIONS:
        raise NotImplementedError(
            f"Database query language {name} not registered via register_database_query_language method"
        )


class BaseFilter(BaseModel, ABC):
    """Base class for all query filters."""

    def __and__(self, other: "BaseFilter") -> "LogicalFilter":
        """Combine filters with and operator."""
        return LogicalFilter(operator="and", left=self, right=other)

    def __or__(self, other: "BaseFilter") -> "LogicalFilter":
        """Combine filters with or operator."""
        return LogicalFilter(operator="or", left=self, right=other)

    def __xor__(self, other: "BaseFilter") -> "LogicalFilter":
        """Combine filters with xor operator (implemented as or with negation)."""
        return LogicalFilter(operator="or", left=self, right=other)

    def __invert__(self) -> "LogicalFilter":
        """Negate the filter with not operator."""
        return LogicalFilter(operator="not", left=self, right=None)

    @staticmethod
    def sanitize_str(value: Any) -> str:
        """Sanitize string values"""
        value = str(value)
        if '"' in value:
            value = value.replace('"', '\\"')
            return f'"{value}"'
        return f'"{value}"'

    @abstractmethod
    def to_str(self, database: str) -> str:
        """Convert the filter to database query string format."""
        raise NotImplementedError("BaseFilter should not be used directly")


class CustomFilter(BaseFilter):
    """Filter for custom expressions."""

    expr: str = Field(..., description="Custom expression")

    def to_str(self, database: str) -> str:
        """Convert the filter to database query string format."""
        return self.expr


class ComparisonFilter(BaseFilter):
    """Filter for comparison operations (==, !=, >, <, >=, <=)."""

    field: str = Field(..., description="Field name to filter on")
    operator: str = Field(..., description="Comparison operator")
    value: Any = Field(..., description="Value to compare against")

    def to_str(self, database: str) -> str:
        """Convert the filter to database query string format."""
        validate_language_registered(database)
        return QUERY_EXPR_FUNCTIONS[database].comparison(self)


class RangeFilter(BaseFilter):
    """Filter for range operations (in, like)."""

    field: str = Field(..., description="Field name to filter on")
    operator: str = Field(..., description="Range operator (in or like)")
    value: Union[Sequence, Set, str] = Field(..., description="Value(s) for range operation")

    def to_str(self, database: str) -> str:
        """Convert the filter to database query string format."""
        validate_language_registered(database)
        return QUERY_EXPR_FUNCTIONS[database].range(self)


class ArithmeticFilter(BaseFilter):
    """Filter for arithmetic operations with field values."""

    field: str = Field(..., description="Field name to filter on")
    arithmetic_operator: str = Field(..., description="Arithmetic operator (+, -, *, /, %, **)")
    arithmetic_value: Union[int, float] = Field(..., description="Value for arithmetic operation")
    comparison_operator: str = Field(..., description="Comparison operator after arithmetic")
    comparison_value: Union[int, float] = Field(..., description="Value to compare against")

    def to_str(self, database: str) -> str:
        """Convert the filter to database query string format."""
        validate_language_registered(database)
        return QUERY_EXPR_FUNCTIONS[database].arithmetic(self)


class NullFilter(BaseFilter):
    """Filter for null value checks (is null, is not null)."""

    field: str = Field(..., description="Field name to filter on")
    is_null: bool = Field(..., description="True for is null, False for is not null")

    def to_str(self, database: str) -> str:
        """Convert the filter to database query string format."""
        validate_language_registered(database)
        return QUERY_EXPR_FUNCTIONS[database].null(self)


class JSONFilter(BaseFilter):
    """Filter for JSON field operations."""

    field: str = Field(..., description="JSON field name")
    key: str = Field(..., description="JSON key to filter on")
    operator: str = Field(..., description="Comparison operator")
    value: Any = Field(..., description="Value to compare against")

    def to_str(self, database: str) -> str:
        """Convert the filter to database query string format."""
        validate_language_registered(database)
        return QUERY_EXPR_FUNCTIONS[database].json_filter(self)


class ArrayFilter(BaseFilter):
    """Filter for array field operations."""

    field: str = Field(..., description="Array field name")
    index: Optional[int] = Field(None, description="Array index to filter on")
    operator: str = Field(..., description="Comparison operator")
    value: Any = Field(..., description="Value to compare against")

    def to_str(self, database: str) -> str:
        """Convert the filter to database query string format."""
        validate_language_registered(database)
        return QUERY_EXPR_FUNCTIONS[database].array(self)


class LogicalFilter(BaseFilter):
    """Filter for logical operations (and, or, not)."""

    operator: str = Field(..., description="Logical operator (and, or, not)")
    left: BaseFilter = Field(..., description="Left operand filter")
    right: Optional[BaseFilter] = Field(None, description="Right operand filter (not needed for not)")

    def to_str(self, database: str) -> str:
        """Convert the filter to database query string format."""
        validate_language_registered(database)
        return QUERY_EXPR_FUNCTIONS[database].logical(self)


class MatchFilter(BaseFilter):
    """Filter for text match operations."""

    field: str = Field(..., description="Field name")
    value: str = Field(..., description="Text value")
    match_mode: Literal["prefix", "suffix", "infix", "exact"] = Field(default="exact", description="Matching mode")

    def to_str(self, database: str) -> str:
        """Convert the filter to database query string format."""
        validate_language_registered(database)
        return QUERY_EXPR_FUNCTIONS[database].text_match(self)


# Convenience factory functions for creating filters
def eq(field: str, value: Any) -> ComparisonFilter:
    """Create an equality filter."""
    return ComparisonFilter(field=field, operator="==", value=value)


def ne(field: str, value: Any) -> ComparisonFilter:
    """Create a not-equal filter."""
    return ComparisonFilter(field=field, operator="!=", value=value)


def gt(field: str, value: Union[int, float]) -> ComparisonFilter:
    """Create a greater-than filter."""
    return ComparisonFilter(field=field, operator=">", value=value)


def lt(field: str, value: Union[int, float]) -> ComparisonFilter:
    """Create a less-than filter."""
    return ComparisonFilter(field=field, operator="<", value=value)


def gte(field: str, value: Union[int, float]) -> ComparisonFilter:
    """Create a greater-than-or-equal filter."""
    return ComparisonFilter(field=field, operator=">=", value=value)


def lte(field: str, value: Union[int, float]) -> ComparisonFilter:
    """Create a less-than-or-equal filter."""
    return ComparisonFilter(field=field, operator="<=", value=value)


def in_list(field: str, values: Union[Sequence, Set]) -> Union[RangeFilter, ComparisonFilter]:
    """Create an in filter for a list of values."""
    if len(values) == 1:
        return ComparisonFilter(field=field, operator="==", value=next(iter(values)))
    return RangeFilter(field=field, operator="in", value=values)


def wildcard_match(field: str, pattern: str, operator: str = "wildcard") -> RangeFilter:
    """Create a filter for wildcard matching (database must support this), put * in pattern string."""
    return RangeFilter(field=field, operator=operator, value=pattern)


def is_null(field: str) -> NullFilter:
    """Create an IS NULL filter."""
    return NullFilter(field=field, is_null=True)


def is_not_null(field: str) -> NullFilter:
    """Create an IS NOT NULL filter."""
    return NullFilter(field=field, is_null=False)


def json_key(field: str, key: str, operator: str, value: Any) -> JSONFilter:
    """Create a JSON key filter."""
    return JSONFilter(field=field, key=key, operator=operator, value=value)


def array_index(field: str, index: int, operator: str, value: Any) -> ArrayFilter:
    """Create an array index filter."""
    return ArrayFilter(field=field, index=index, operator=operator, value=value)


def filter_user(users: Union[str, List[str]], user_id_field: str = "user_id") -> Union[RangeFilter, ComparisonFilter]:
    """Create an in filter for one or multiple user id."""
    if isinstance(users, str):
        users = [users]
    return in_list(user_id_field, users)


def chain_filters(filters: List[BaseFilter]) -> Optional[LogicalFilter]:
    """Chain filters with AND operator (&), returns None for empty input."""
    if filters:
        final_expr = filters.pop(0)
        for expr in filters:
            final_expr = final_expr & expr
        return final_expr
    return None


QUERY_EXPR_FUNCTIONS = dict(
    milvus=QueryLanguageDefinition(
        comparison=milvus_comparison_filter,
        range=milvus_range_filter,
        arithmetic=milvus_arithmetic_filter,
        null=milvus_null_filter,
        json_filter=milvus_json_filter,
        array=milvus_array_filter,
        logical=milvus_logical_filter,
        text_match=milvus_text_match_filter,
    )
)
