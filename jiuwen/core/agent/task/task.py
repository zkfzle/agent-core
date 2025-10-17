from jiuwen.agent.common.enum import TaskStatus
from jiuwen.core.runtime.runtime import Runtime


class Task:
    def __init__(self, task_id: str, context: Runtime):
        self.task_id = task_id
        self.context = context
        self.status: TaskStatus = TaskStatus.PENDING

    def set_status(self, status: TaskStatus) -> None:
        self.status = status
