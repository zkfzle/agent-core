"""任务管理器模块

该模块实现了任务管理的核心功能，包括：
- TaskManagerState: 任务管理器状态，用于序列化和恢复
- TaskManager: 任务管理器，负责任务的CRUD、状态管理、优先级管理和层级关系管理

主要功能：
- 任务的增删改查（CRUD）
- 任务执行状态管理（submitted、working、paused、completed等）
- 任务优先级管理（支持按优先级查询和排序）
- 任务层级关系管理（父子任务关系）

索引结构：
- _priority_index: 优先级索引，用于快速按优先级查找任务
- _parent_to_children: 父子关系索引，用于快速查找子任务
- _child_to_parent: 子父关系索引，用于快速查找父任务
- _root_tasks: 根任务集合，用于快速查找根任务
"""
from typing import Dict, Any, List, Union, Set, Optional
from collections import defaultdict

from pydantic.v1 import BaseModel
from typing_extensions import Literal

from openjiuwen.core.controller.schema.task import Task, TaskStatus


class TaskManagerState(BaseModel):
    """任务管理器状态
    
    用于序列化和恢复任务管理器的状态。
    """
    tasks: Dict[str, Task]
    priority_index: Dict[int, List[str]]
    parent_to_children: Dict[str, Set[str]]
    children_to_parent: Dict[str, str]
    root_tasks: Set[str]


class TaskQuery(BaseModel):
    task_id: Optional[Union[str, List[str]]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    priority: int = None,
    status: Optional[TaskStatus] = None,
    with_children: bool = False,
    is_recursive: bool = False,
    is_root: bool = False


class TaskManager:
    """任务管理器
    
    负责任务的CRUD操作、状态管理、优先级管理和层级关系管理。
    提供高效的索引结构用于快速查询任务。
    """
    def __init__(self, config):
        """初始化任务管理器
        
        Args:
            config: Agent配置字典
        """
        self._config: Dict[str, Any] = config
        self.tasks: Dict[str, Task] = {}  # task_id -> Task
        
        # ==================== 优先级索引 ====================
        # 优先级索引：priority -> List[task_id]
        # 用于快速按优先级查找和排序任务
        self._priority_index: Dict[int, List[str]] = defaultdict(list)  # priority_index -> List[Task]
        
        # ==================== 层级关系索引 ====================
        # 父子关系索引：parent_task_id -> Set[child_task_id]
        # 用于快速查找某个任务的所有直接子任务
        self._parent_to_children: Dict[str, Set[str]] = defaultdict(set)
        
        # 子父关系索引：child_task_id -> parent_task_id
        # 用于快速查找某个任务的父任务
        self._child_to_parent: Dict[str, str] = {}
        
        # 根任务集合：存储所有没有父任务的任务ID
        # 用于快速查找根任务（is_root=True的查询）
        self._root_tasks: Set[str] = set()

    def get_state(self) -> TaskManagerState:
        """获取任务管理器状态
        
        Returns:
            TaskManagerState: 任务管理器状态对象
        """
        return TaskManagerState(
            tasks=self.tasks,
            priority_index=self._priority_index,
            parent_to_children=self._parent_to_children,
            children_to_parent=self._child_to_parent,
            root_tasks=self._root_tasks
        )

    def load_state(self, state: TaskManagerState) -> None:
        """加载任务管理器状态
        
        Args:
            state: 任务管理器状态对象
        """
        self.tasks = state.tasks
        self._priority_index = state.priority_index
        self._parent_to_children = state.parent_to_children
        self._child_to_parent = state.child_to_parent
        self.root_tasks = state.root_tasks

    # ==================== 任务 CRUD 操作 ====================
    def add_task(self, task: Union[Task, List[Task]]):
        """添加任务到任务队列
        
        Args:
            task: 单个任务或任务列表
        """
        self.tasks[task.task_id] = task
        ...

    def get_task(
            self,
            task_query: Optional[TaskQuery] = None
    ) -> List["Task"]:
        """查询任务
        
        根据多个条件查询任务，支持多种查询方式。
        
        Args:
            task_query: 任务查询条件
            
        Returns:
            List[Task]: 匹配的任务列表
        """
        # 按 task_id 查询
        tasks = []
        if task_query.task_id is not None and isinstance(task_query.task_id, str) and task_query.task_id in self.tasks:
            tasks.append(self.tasks[task_query.task_idtask_id])
        # 其他查询逻辑
        return tasks

    def pop_task(
            self,
            task_query: Optional[TaskQuery] = None
    ) -> List["Task"]:
        """弹出任务（查询并移除）
        
        查询任务并从任务管理器中移除，参数同get_task。
        
        Args:
            task_query: 任务查询条件
            
        Returns:
            List[Task]: 匹配的任务列表（已从管理器中移除）
        """
        # 按 task_id 查询
        tasks = []
        if task_query.task_id is not None and isinstance(task_query.task_id, str) and task_query.task_id in self.tasks:
            tasks.append(self.tasks[task_query.task_id])
            # 删除任务
            del self.tasks[task_query.task_id]
            # 在其他数据结构中删除 task_id
        # 其他查询逻辑
        return tasks

    def update_task(self, task: Union[Task, List[Task]]):
        """更新任务
        
        更新任务信息，如果任务不存在则不会创建新任务。
        
        Args:
            task: 要更新的任务，可以是单个任务或任务列表
            
        Returns:
            bool: 是否成功更新
        """
        if isinstance(task, Task) and task.task_id in self.tasks:
            del self.tasks[task.task_id]
            self.tasks[task.task_id] = task
            # 处理其他添加逻辑
        ...

    def remove_task(
            self,
            task_query: Optional[TaskQuery] = None
    ):
        """删除任务
        
        根据条件删除任务，支持删除子任务。
        
        Args:
            task_query: 任务查询条件
        """
        # 按 task_id 删除
        tasks = []
        if task_query.task_id is not None and isinstance(task_query.task_id, str) and task_query.task_id in self.tasks:
            # 删除任务
            del self.tasks[task_query.task_id]
            # 在其他数据结构中删除 task_id
        # 其他删除逻辑
        return tasks

    def get_child_task(
            self,
            task_id: Union[str, List[str]],
            is_recursive: bool = False,
    ) -> List[Task]:
        """获取子任务
        
        获取指定任务的所有子任务。
        
        Args:
            task_id: 任务ID，可以是单个ID或ID列表
            is_recursive: 是否递归获取所有子任务（包括子任务的子任务）
            
        Returns:
            List[Task]: 子任务列表
        """
        tasks = []
        if isinstance(task_id, str) and task_id in self._parent_to_children:
            for child in self._parent_to_children[task_id]:
                tasks.append(self.tasks.get(child))
        # 其他获取子任务逻辑
        return tasks

    # ==================== 任务执行状态管理 ====================
    def update_task_status(
            self,
            task_id: Union[str, List[str]],
            new_status: "TaskStatus",
            with_children: bool = False,
            is_recursive: bool = False,
    ):
        """更新任务状态
        
        更新任务的状态，并同步更新优先级索引。
        
        Args:
            task_id: 任务ID，可以是单个ID或ID列表
            new_status: 新状态
            with_children: 是否同时更新子任务状态
            is_recursive: 是否递归更新子任务状态
        """
        ...

    # ==================== 任务优先级管理 ====================
    def set_priority(
            self,
            task_id: Union[str, List[str]],
            new_priority: str,
            with_children: bool = False,
            is_recursive: bool = False,
    ):
        """设置任务优先级
        
        更新任务的优先级，并同步更新优先级索引。
        
        Args:
            task_id: 任务ID，可以是单个ID或ID列表
            new_priority: 新优先级
            with_children: 是否同时更新子任务优先级
            is_recursive: 是否递归更新子任务优先级
        """
        ...


