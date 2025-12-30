#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ConflictType(Enum):
    ADD = "ADD"
    DELETE = "DELETE"
    UPDATE = "UPDATE"
    NONE = "NONE"


class MemoryType(Enum):
    USER_PROFILE = "user_profile"
    VARIABLE = "variable"
    IMPLICIT_USER_PROFILE = "implicit_user_profile"
    SEMANTIC_MEMORY = "semantic_memory"
    SUMMARY = "summary"
    UNKNOWN = "unknown"


@dataclass
class BaseMemoryUnit:
    """a single memory data item"""
    mem_type: MemoryType
    user_id: str
    group_id: str


@dataclass
class UserProfileUnit(BaseMemoryUnit):
    profile_type: str
    profile_mem: str
    score: Optional[float] = None  # Relevance Scoring
    message_mem_id: Optional[str] = None  # Corresponding Message ID
    mem_id: str = ""
    is_implicit: bool = False
    reasoning: str = ""
    context_summary: str = ""


@dataclass
class VariableUnit(BaseMemoryUnit):
    variable_name: str
    variable_mem: str
    mem_id: str = ""


@dataclass
class SemanticMemoryUnit(BaseMemoryUnit):
    mem_id: str = ""
    semantic_mem: str = ""
    message_mem_id: str = ""


@dataclass
class SummaryUnit(BaseMemoryUnit):
    summary: str
    mem_id: str = ""
    message_mem_id: Optional[str] = None  # Corresponding Message ID
