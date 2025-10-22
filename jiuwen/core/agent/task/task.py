from typing import Optional, Dict, Any
from jiuwen.agent.common.enum import TaskStatus
from jiuwen.core.runtime.runtime import Runtime


class Task:
    def __init__(self, task_id: str, context: Runtime, description: Optional[str] = None):
        self.task_id = task_id
        self.context = context
        self.description = description
        self.status: TaskStatus = TaskStatus.PENDING
        self.metadata: Dict[str, Any] = {}
        self.agent_id: Optional[str] = None

    def set_status(self, status: TaskStatus) -> None:
        self.status = status
        
    def set_metadata(self, key: str, value: Any) -> None:
        """设置元数据"""
        self.metadata[key] = value
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self.metadata.get(key, default)
        
    def set_agent_id(self, agent_id: str) -> None:
        """设置Agent ID"""
        self.agent_id = agent_id
