from typing import Any

from pydantic import BaseModel


class PriorityEvent(BaseModel):
    id: str
    assignee: str
    priority: int
    type: str = None
    task_description: str = None
    meta: dict = {}
    status: str = None
    result: Any = None
    inputs: dict = None
    callable: Any = None

    def state(self):
        return {
            "id": self.id,
            "assignee": self.assignee,
            "priority": self.priority,
            "type": self.type,
            "task_description": self.task_description,
            "meta": self.meta,
            "status": self.status,
            "result": self.result,
        }