"""
Task Manager Module

This module implements the core functionality of task management, including:
- TaskManagerState: Task manager state for serialization and restoration
- TaskManager: Task manager responsible for CRUD operations, status management,
priority management, and hierarchical relationship management

Main Features:
- Task CRUD operations (Create, Read, Update, Delete)
- Task execution status management (submitted, working, paused, completed, etc.)
- Task priority management (supports querying and sorting by priority)
- Task hierarchical relationship management (parent-child task relationships)

Index Structure:
- _priority_index: Priority index for fast task lookup by priority
- _parent_to_children: Parent-to-children relationship index for fast child task lookup
- _child_to_parent: Child-to-parent relationship index for fast parent task lookup
- _root_tasks: Root task set for fast root task lookup
"""

# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.


from typing import Dict, Any, List, Union, Set, Optional
from collections import defaultdict

from pydantic import BaseModel
from typing_extensions import Literal

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.controller.schema import Task, TaskStatus


class TaskManagerState(BaseModel):
    """Task Manager State

    Used for serialization and restoration of task manager state.
    """
    tasks: Dict[str, Task]
    priority_index: Dict[int, List[str]]
    parent_to_children: Dict[str, Set[str]]
    children_to_parent: Dict[str, str]
    root_tasks: Set[str]

    class Config:
        arbitrary_types_allowed = True


class TaskManager:
    """Task Manager

    Responsible for task CRUD operations, status management, priority management,
    and hierarchical relationship management.
    Provides efficient index structures for fast task queries.
    """

    def __init__(self, config):
        """Initialize task manager

        Args:
            config: Agent configuration dictionary
        """
        self._config: Dict[str, Any] = config
        self.tasks: Dict[str, Task] = {}  # task_id -> Task

        # ==================== Priority Index ====================
        # Priority index: priority -> List[task_id]
        # Used for fast task lookup and sorting by priority
        self._priority_index: Dict[int, List[str]] = defaultdict(list)  # priority_index -> List[Task]

        # ==================== Hierarchical Relationship Index ====================
        # Parent-to-children relationship index: parent_task_id -> Set[child_task_id]
        # Used for fast lookup of all direct child tasks of a task
        self._parent_to_children: Dict[str, Set[str]] = defaultdict(set)

        # Child-to-parent relationship index: child_task_id -> parent_task_id
        # Used for fast lookup of a task's parent task
        self._child_to_parent: Dict[str, str] = {}

        # Root task set: stores all task IDs that have no parent task
        # Used for fast lookup of root tasks (is_root=True queries)
        self._root_tasks: Set[str] = set()

    def get_state(self) -> TaskManagerState:
        """Get task manager state

        Returns:
            TaskManagerState: Task manager state object
        """
        return TaskManagerState(
            tasks=self.tasks.copy(),
            priority_index=dict(self._priority_index),
            parent_to_children={k: v.copy() for k, v in self._parent_to_children.items()},
            children_to_parent=self._child_to_parent.copy(),
            root_tasks=self._root_tasks.copy()
        )

    def load_state(self, state: TaskManagerState) -> None:
        """Load task manager state

        Args:
            state: Task manager state object
        """
        self.tasks = state.tasks.copy()
        self._priority_index = defaultdict(list, state.priority_index)
        self._parent_to_children = defaultdict(set, {k: v.copy() for k, v in state.parent_to_children.items()})
        self._child_to_parent = state.children_to_parent.copy()
        self._root_tasks = state.root_tasks.copy()

    # ==================== Task CRUD Operations ====================
    def add_task(self, task: Union[Task, List[Task]]):
        """Add task(s) to task queue

        Args:
            task: Single task or list of tasks
        """
        tasks = [task] if isinstance(task, Task) else task
        for t in tasks:
            # Add task to dictionary
            if t.task_id in self.tasks:
                raise JiuWenBaseException(
                    error_code=StatusCode.CONTROLLER_TASK_EXISTS_ERROR.code,
                    message=StatusCode.CONTROLLER_TASK_EXISTS_ERROR.errmsg.format(
                        error_msg=f"{t.task_id} already exists!"
                    )
                )
            self.tasks[t.task_id] = t.model_copy()

            # Update priority index
            self._priority_index[t.priority].append(t.task_id)

            # Update hierarchical relationship index
            if t.parent_task_id:
                self._parent_to_children[t.parent_task_id].add(t.task_id)
                self._child_to_parent[t.task_id] = t.parent_task_id
                # If this task was previously a root task, remove it from root task set
                if t.task_id in self._root_tasks:
                    self._root_tasks.remove(t.task_id)
            else:
                # No parent task, add to root task set
                self._root_tasks.add(t.task_id)

    def get_task(
            self,
            task_id: Optional[Union[str, List[str]]] = None,
            session_id: Optional[str] = None,
            user_id: Optional[str] = None,
            priority: int = None,
            status: Optional[TaskStatus] = None,
            with_children: bool = False,
            is_recursive: bool = False,
            is_root: bool = False,
    ) -> List["Task"]:
        """Query tasks

        Query tasks based on multiple conditions, supporting various query methods.

        Args:
            task_id: Task ID, can be a single ID or a list of IDs
            session_id: Session ID
            user_id: User ID
            priority: Priority
            status: Task execution status
            with_children: Whether to include child tasks
            is_recursive: Whether to recursively query child tasks
            is_root: Whether to only query root tasks (tasks without parent tasks)

        Returns:
            List[Task]: List of matching tasks
        """
        result_tasks = []
        candidate_ids = set()

        # Query by task_id
        if task_id is not None:
            task_ids = [task_id] if isinstance(task_id, str) else task_id
            candidate_ids.update(task_ids)

        # Query by session_id
        if session_id is not None:
            for t in self.tasks.values():
                if t.session_id == session_id:
                    candidate_ids.add(t.task_id)

        # Query by priority
        if priority is not None:
            if priority in self._priority_index:
                candidate_ids.update(self._priority_index[priority])

        # Query by is_root
        if is_root:
            candidate_ids.update(self._root_tasks)

        # If no query conditions are specified, return all tasks
        if task_id is None and session_id is None and priority is None and not is_root:
            candidate_ids = set(self.tasks.keys())

        # Filter tasks
        for tid in candidate_ids:
            if tid not in self.tasks:
                continue
            task = self.tasks[tid]

            # Apply other filter conditions
            if status is not None and task.Status != status:
                continue
            if user_id is not None:
                # user_id is not in Task model, may be stored in metadata
                if task.metadata and task.metadata.get("user_id") != user_id:
                    continue
                elif task.metadata is None or "user_id" not in task.metadata:
                    continue

            result_tasks.append(task)

        # Handle with_children and is_recursive
        if with_children:
            children_ids = set()
            for task in result_tasks:
                if task.task_id in self._parent_to_children:
                    if is_recursive:
                        # Recursively get all child tasks
                        self._collect_all_children(task.task_id, children_ids)
                    else:
                        # Only get direct child tasks
                        children_ids.update(self._parent_to_children[task.task_id])

            # Add child tasks
            for cid in children_ids:
                if cid in self.tasks:
                    result_tasks.append(self.tasks[cid])

        return result_tasks

    def _collect_all_children(self, parent_id: str, children_set: Set[str]):
        """Recursively collect all child task IDs

        Args:
            parent_id: Parent task ID
            children_set: Set for storing child task IDs
        """
        if parent_id in self._parent_to_children:
            for child_id in self._parent_to_children[parent_id]:
                children_set.add(child_id)
                self._collect_all_children(child_id, children_set)

    def pop_task(
            self,
            task_id: Optional[Union[str, List[str]]] = None,
            session_id: Optional[str] = None,
            user_id: Optional[str] = None,
            priority: Union[int, Literal["highest"]] = None,
            status: Optional[TaskStatus] = None,
            with_children: bool = False,
            is_recursive: bool = False,
            is_root: bool = False,
    ) -> List["Task"]:
        """Pop tasks (query and remove)

        Query tasks and remove them from the task manager. Parameters are the same as get_task.

        Args:
            task_id: Task ID, can be a single ID or a list of IDs
            session_id: Session ID
            user_id: User ID
            priority: Priority, can be an integer or "highest" (highest priority)
            status: Task execution status
            with_children: Whether to include child tasks
            is_recursive: Whether to recursively query child tasks
            is_root: Whether to only query root tasks (tasks without parent tasks)

        Returns:
            List[Task]: List of matching tasks (already removed from manager)
        """
        # Handle the highest priority
        if priority == "highest":
            # Find the highest priority (priority with the largest number)
            if self._priority_index:
                priority = max(self._priority_index.keys())
            else:
                return []

        # Get matching tasks
        tasks = self.get_task(
            task_id=task_id,
            session_id=session_id,
            user_id=user_id,
            priority=priority,
            status=status,
            with_children=with_children,
            is_recursive=is_recursive,
            is_root=is_root
        )

        # Remove these tasks
        task_ids_to_remove = [t.task_id for t in tasks]
        self.remove_task(
            task_id=task_ids_to_remove,
            with_children=False,  # Already handled in get_task
        )

        return tasks

    def update_task(self, task: Union[Task, List[Task]]) -> bool:
        """Update task(s)

        Update task information. If the task does not exist, it will not create a new task.

        Args:
            task: Task(s) to update, can be a single task or a list of tasks

        Returns:
            bool: Whether the update was successful
        """
        tasks = [task] if isinstance(task, Task) else task
        all_success = True

        for t in tasks:
            if t.task_id not in self.tasks:
                all_success = False
                continue

            old_task = self.tasks[t.task_id]

            # Update task
            self.tasks[t.task_id] = t

            # Update priority index
            # Always remove from old priority index first
            if old_task.priority in self._priority_index:
                try:
                    self._priority_index[old_task.priority].remove(t.task_id)
                except ValueError:
                    pass
            # Always add to new priority index
            if t.task_id not in self._priority_index.get(t.priority, []):
                self._priority_index[t.priority].append(t.task_id)

            # If parent_task_id changed, update hierarchical relationship index
            if old_task.parent_task_id != t.parent_task_id:
                # Remove from old parent task relationship
                if old_task.parent_task_id:
                    if old_task.parent_task_id in self._parent_to_children:
                        self._parent_to_children[old_task.parent_task_id].discard(t.task_id)
                    if t.task_id in self._child_to_parent:
                        del self._child_to_parent[t.task_id]
                    # If old parent task has no children left, no special handling needed

                # Update new parent task relationship
                if t.parent_task_id:
                    self._parent_to_children[t.parent_task_id].add(t.task_id)
                    self._child_to_parent[t.task_id] = t.parent_task_id
                    # Remove from root task set
                    if t.task_id in self._root_tasks:
                        self._root_tasks.remove(t.task_id)
                else:
                    # No parent task, add to root task set
                    self._root_tasks.add(t.task_id)
                    # Remove from child-to-parent relationship index
                    if t.task_id in self._child_to_parent:
                        del self._child_to_parent[t.task_id]

        return all_success

    def remove_task(
            self,
            task_id: Union[str, List[str], None] = None,
            session_id: str = None,
            user_id: str = None,
            priority: str = None,
            status: TaskStatus = None,
            with_children: bool = False,
            is_root: bool = False,
    ):
        """Remove task(s)

        Remove tasks based on conditions, supports removing child tasks.

        Args:
            task_id: Task ID, can be a single ID or a list of IDs. If None, must provide
                other filter criteria (session_id, priority, status, is_root, etc.),
                otherwise ValueError will be raised to prevent accidental deletion of all tasks
            session_id: Session ID
            user_id: User ID
            priority: Priority
            status: Task execution status
            with_children: Whether to also remove child tasks
            is_root: Whether to only remove root tasks

        Raises:
            ValueError: If task_id is None and no other filter criteria are provided
        """
        # Safety check: if task_id is None, must provide at least one other filter condition
        if task_id is None and session_id is None and user_id is None and priority is None and status is None and not is_root:
            raise JiuWenBaseException(
                StatusCode.CONTROLLER_REMOVE_TASK_ERROR.code,
                StatusCode.CONTROLLER_REMOVE_TASK_ERROR.errmsg.format(
                    error_msg =
                    "Cannot remove tasks: task_id is None and no other filter criteria provided. "
                    "This would delete all tasks. Please provide at least one filter: "
                    "task_id, session_id, user_id, priority, status, or is_root=True"
                )
            )

        # Get tasks to remove
        tasks_to_remove = self.get_task(
            task_id=task_id,
            session_id=session_id,
            user_id=user_id,
            priority=int(priority) if priority is not None else None,
            status=status,
            with_children=with_children,
            is_root=is_root
        )

        # Collect all task IDs to remove (including child tasks)
        task_ids_to_remove = set()
        for task in tasks_to_remove:
            task_ids_to_remove.add(task.task_id)
            if with_children:
                if task.task_id in self._parent_to_children:
                    task_ids_to_remove.update(self._parent_to_children[task.task_id])

        # Remove tasks
        for tid in task_ids_to_remove:
            if tid not in self.tasks:
                continue

            task = self.tasks[tid]

            # Remove from priority index
            if task.priority in self._priority_index:
                try:
                    self._priority_index[task.priority].remove(tid)
                except ValueError:
                    pass
                # If the list for this priority is empty, it can be kept (defaultdict will handle it automatically)

            # Remove from hierarchical relationship index
            if task.parent_task_id:
                # Remove from parent task's child task set
                if task.parent_task_id in self._parent_to_children:
                    self._parent_to_children[task.parent_task_id].discard(tid)
                # Remove from child-to-parent relationship index
                if tid in self._child_to_parent:
                    del self._child_to_parent[tid]
            else:
                # Remove from root task set
                if tid in self._root_tasks:
                    self._root_tasks.remove(tid)

            # Handle child tasks: promote child tasks to root tasks (if their parent task is deleted)
            # Only promote children that are NOT being removed
            if tid in self._parent_to_children:
                children = self._parent_to_children[tid].copy()
                for child_id in children:
                    # Only promote if child is not being removed
                    if child_id not in task_ids_to_remove:
                        # Set child task's parent_task_id to None
                        if child_id in self.tasks:
                            child_task = self.tasks[child_id]
                            child_task.parent_task_id = None
                            # Add to root task set
                            self._root_tasks.add(child_id)
                            # Remove from child-to-parent relationship index
                            if child_id in self._child_to_parent:
                                del self._child_to_parent[child_id]
                # Delete parent-to-children mapping
                del self._parent_to_children[tid]

            # Remove from task dictionary
            del self.tasks[tid]

    def get_child_task(
            self,
            task_id: Union[str, List[str]],
            is_recursive: bool = False,
    ):
        """Get child tasks

        Get all child tasks of the specified task(s).

        Args:
            task_id: Task ID, can be a single ID or a list of IDs
            is_recursive: Whether to recursively get all child tasks (including children of children)

        Returns:
            List[Task]: List of child tasks
        """
        task_ids = [task_id] if isinstance(task_id, str) else task_id
        children_ids = set()

        for tid in task_ids:
            if tid in self._parent_to_children:
                if is_recursive:
                    self._collect_all_children(tid, children_ids)
                else:
                    children_ids.update(self._parent_to_children[tid])

        # Return list of child task objects
        return [self.tasks[cid] for cid in children_ids if cid in self.tasks]

    # ==================== Task Execution Status Management ====================
    def update_task_status(
            self,
            task_id: Union[str, List[str]],
            new_status: "TaskStatus",
            with_children: bool = False,
            is_recursive: bool = False,
    ):
        """Update task status

        Update task status and synchronously update priority index.

        Args:
            task_id: Task ID, can be a single ID or a list of IDs
            new_status: New status
            with_children: Whether to also update child task status
            is_recursive: Whether to recursively update child task status
        """
        task_ids = [task_id] if isinstance(task_id, str) else task_id
        all_task_ids = set(task_ids)

        # If need to update child tasks, collect all child task IDs
        if with_children:
            for tid in task_ids:
                if is_recursive:
                    self._collect_all_children(tid, all_task_ids)
                else:
                    if tid in self._parent_to_children:
                        all_task_ids.update(self._parent_to_children[tid])

        # Update status for all tasks
        for tid in all_task_ids:
            if tid in self.tasks:
                self.tasks[tid].Status = new_status

    # ==================== Task Priority Management ====================
    def set_priority(
            self,
            task_id: Union[str, List[str]],
            new_priority: Union[int, str],
            with_children: bool = False,
            is_recursive: bool = False,
    ):
        """Set task priority

        Update task priority and synchronously update priority index.

        Args:
            task_id: Task ID, can be a single ID or a list of IDs
            new_priority: New priority (integer or string representation of integer)
            with_children: Whether to also update child task priority
            is_recursive: Whether to recursively update child task priority
        """
        # Convert priority to integer
        if isinstance(new_priority, str):
            new_priority = int(new_priority)

        task_ids = [task_id] if isinstance(task_id, str) else task_id
        all_task_ids = set(task_ids)

        # If need to update child tasks, collect all child task IDs
        if with_children:
            for tid in task_ids:
                if is_recursive:
                    self._collect_all_children(tid, all_task_ids)
                else:
                    if tid in self._parent_to_children:
                        all_task_ids.update(self._parent_to_children[tid])

        # Update priority for all tasks
        for tid in all_task_ids:
            if tid in self.tasks:
                task = self.tasks[tid]
                old_priority = task.priority

                # Update task priority
                task.priority = new_priority

                # Update priority index
                if old_priority != new_priority:
                    # Remove from old priority index
                    if old_priority in self._priority_index:
                        try:
                            self._priority_index[old_priority].remove(tid)
                        except ValueError:
                            pass
                    # Add to new priority index
                    self._priority_index[new_priority].append(tid)


