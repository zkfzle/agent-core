from typing import Optional, Dict
from enum import Enum

from jiuwen.core.agent.task.task import Task
from jiuwen.core.agent.task.task_context import AgentRuntime
from jiuwen.core.runtime.agent_context import AgentContext


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class TaskManager:
    def __init__(self, agent_context: "AgentContext") -> None:
        self.agent_context = agent_context
        self._tasks: Dict[str, Task] = {}

    def create_task(self, conversation_id: str) -> Task:
        """
        如果 conversation_id 已存在则复用其 Context，
        否则新建 Context 并返回新的 Task。
        """
        task_id = conversation_id  # 直接使用 conversation_id 作为 task_id

        # 复用已有context
        if task_id in self._tasks:
            return self._tasks[task_id]
        # 新建context
        context = AgentRuntime(trace_id=task_id)
        task = Task(task_id, context)

        self._tasks[task_id] = task
        self.agent_context.context_map[task_id] = context
        return task

    def get_task(self, conversation_id: str) -> Optional[Task]:
        return self._tasks.get(conversation_id)

    def remove_task(self, conversation_id: str) -> None:
        task = self._tasks.pop(conversation_id, None)
        if task is None:
            return
        self.agent_context.context_map.pop(conversation_id, None)
