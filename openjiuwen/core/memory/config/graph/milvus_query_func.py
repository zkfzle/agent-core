# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""Query Expression support for Milvus"""

from typing import Any, Literal, Optional, Protocol, Sequence, Set


class QueryExprInterface(Protocol):
    is_null: bool
    left: "QueryExprInterface"
    right: "QueryExprInterface"
    match_mode: Literal["prefix", "suffix", "infix", "exact"]
    index: Optional[int]
    key: str
    field: str
    operator: str
    arithmetic_operator: str
    comparison_operator: str
    value: Any
    comparison_value: Any
    arithmetic_value: Any

    @staticmethod
    def sanitize_str(value: Any) -> str: ...

    def to_str(self, database: str) -> str: ...


def milvus_comparison_filter(self: QueryExprInterface) -> str:
    """Convert to Milvus comparison filter string."""
    if isinstance(self.value, str):
        return f"{self.field} {self.operator} {self.sanitize_str(self.value)}"
    else:
        return f"{self.field} {self.operator} {self.value}"


def milvus_range_filter(self: QueryExprInterface) -> str:
    """Convert to Milvus range filter string."""
    if self.operator.lower() == "in":
        if isinstance(self.value, (Sequence, Set)):
            # Handle list values for in operator
            if all(isinstance(v, str) for v in self.value):
                values_str = ",".join(self.sanitize_str(v) for v in self.value)
            else:
                values_str = ",".join(str(v) for v in self.value)
            return f"{self.field} in [{values_str}]"
        else:
            raise ValueError("in operator requires a sequence or set value")
    elif self.operator.lower() == "like":
        if isinstance(self.value, str):
            if "%" not in self.value:
                raise ValueError(f"Milvus's like operator uses % for wildcard matching")
            return f"{self.field} like {self.sanitize_str(self.value)}"
        else:
            raise ValueError("like operator requires a string value")
    else:
        raise ValueError(f"Unsupported range operator: {self.operator}")


def milvus_arithmetic_filter(self: QueryExprInterface) -> str:
    """Convert to Milvus arithmetic filter string."""
    return (
        f"{self.field} {self.arithmetic_operator} {self.arithmetic_value}"
        + f"{self.comparison_operator} {self.comparison_value}"
    )


def milvus_null_filter(self: QueryExprInterface) -> str:
    """Convert to Milvus null filter string."""
    if self.is_null:
        return f"{self.field} is null"
    else:
        return f"{self.field} is not null"


def milvus_json_filter(self: QueryExprInterface) -> str:
    """Convert to Milvus JSON filter string."""
    if isinstance(self.value, str):
        return f"{self.field}[{self.sanitize_str(self.key)}] {self.operator} {self.sanitize_str(self.value)}"
    else:
        return f"{self.field}[{self.sanitize_str(self.key)}] {self.operator} {self.value}"


def milvus_array_filter(self: QueryExprInterface) -> str:
    """Convert to Milvus array filter string."""
    if self.index is not None:
        if isinstance(self.value, str):
            return f"{self.field}[{self.index}] {self.operator} {self.sanitize_str(self.value)}"
        else:
            return f"{self.field}[{self.index}] {self.operator} {self.value}"
    else:
        # Filter on the entire array field
        if isinstance(self.value, str):
            return f"{self.field} {self.operator} {self.sanitize_str(self.value)}"
        else:
            return f"{self.field} {self.operator} {self.value}"


def milvus_logical_filter(self: QueryExprInterface) -> str:
    """Convert to Milvus logical filter string."""
    if self.operator.lower() == "not":
        if self.right is not None:
            raise ValueError("not operator should not have a right operand")
        return f"not ({self.left.to_str('milvus')})"
    elif self.operator.lower() in ["and", "or"]:
        if self.right is None:
            raise ValueError(f"{self.operator} operator requires both left and right operands")
        return f"({self.left.to_str('milvus')}) {self.operator} ({self.right.to_str('milvus')})"
    else:
        raise ValueError(f"Unsupported logical operator: {self.operator}")


def milvus_text_match_filter(self: QueryExprInterface) -> str:
    """Convert to Milvus text match filter string."""
    pattern = self.value
    match self.match_mode:
        case "exact":
            return f"TEXT_MATCH({self.field}, {self.sanitize_str(pattern)})"
        case "prefix":
            pattern = "%" + pattern
        case "suffix":
            pattern = pattern + "%"
        case "infix":
            pattern = "%" + pattern + "%"
        case _:
            raise ValueError(f"Unknown match mode: {self.match_mode}")
    return f"{self.field} like {self.sanitize_str(pattern)}"
