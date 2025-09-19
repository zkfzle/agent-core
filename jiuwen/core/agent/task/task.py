from jiuwen.agent.common.enum import TaskStatus
from jiuwen.core.agent.task.task_context import AgentRuntime


class Task:
    def __init__(self, task_id: str, context: AgentRuntime):
        self.task_id = task_id
        self.context = context
        self.status: TaskStatus = TaskStatus.PENDING

    # 便捷方法：一键设置整体状态
    def set_status(self, status: TaskStatus) -> None:
        self.status = status
