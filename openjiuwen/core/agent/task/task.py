#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Set, List
from pydantic import BaseModel, Field
from openjiuwen.agent.common.enum import TaskStatus, TaskType


class DependencyType(Enum):
    """依赖类型"""
    SEQUENTIAL = "sequential"  # 顺序依赖（前置任务完成后才能执行）
    PARALLEL = "parallel"  # 并行依赖（可以同时执行，但需要等待依赖完成）
    CONDITIONAL = "conditional"  # 条件依赖（根据条件决定是否执行）
    DATA = "data"  # 数据依赖（需要前置任务的输出数据）


@dataclass
class TaskDependency:
    """任务依赖关系"""
    dependency_id: str  # 依赖的任务ID
    dependency_type: DependencyType = DependencyType.SEQUENTIAL
    condition: Optional[str] = Field(default=None)  # 条件表达式（用于条件依赖）
    data_mapping: Dict[str, str] = field(default_factory=dict)  # 数据映射：{源字段: 目标字段}
    required: bool = True  # 是否为必需依赖

    def __post_init__(self):
        if self.data_mapping is None:
            self.data_mapping = {}


class TaskInput(BaseModel):
    """任务调用输入 - 统一处理工具、工作流、MCP等调用"""
    target_id: str = Field(default="")
    target_name: str = Field(default="")
    arguments: Any = Field(default_factory=dict)


class TaskResult(BaseModel):
    """任务执行结果 - 极简设计，消除重复字段"""
    status: TaskStatus
    output: Any = Field(default=None)  # 成功时的输出数据（WorkflowOutput等）
    error: Optional[str] = Field(default=None)  # 失败时的错误信息
    metadata: Dict[str, Any] = Field(default_factory=dict)  # execution_time等扩展信息


class Task(BaseModel):
    """统一的任务类 - 支持依赖关系"""
    agent_id: Optional[str] = Field(default=None)
    task_id: str = Field(default="")
    task_type: TaskType = Field(default=TaskType.UNDEFINED)

    description: Optional[str] = Field(default=None)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    input: TaskInput = Field(default_factory=TaskInput)
    result: Optional[TaskResult] = Field(default=None)  # 明确类型，不再是 Any

    # 依赖关系管理
    dependencies: List[TaskDependency] = Field(default_factory=list)  # 此任务依赖的其他任务
    dependents: Set[str] = Field(default_factory=set)  # 依赖此任务的任务ID集合

    # DAG相关属性
    parent_task_id: Optional[str] = Field(default=None)  # 父任务ID（用于子任务）
    child_task_ids: Set[str] = Field(default_factory=set)  # 子任务ID集合
    group_id: Optional[str] = Field(default=None)  # 任务组ID
    level: int = Field(default=0)  # 在依赖图中的层级（0为根任务）

    def set_agent_id(self, agent_id: str) -> None:
        """设置Agent ID"""
        self.agent_id = agent_id
