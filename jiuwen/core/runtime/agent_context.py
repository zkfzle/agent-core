from jiuwen.core.agent.task.task_context import TaskContext
from jiuwen.core.runtime.store import Store


class AgentContext:
    context_map: dict[str, TaskContext] = {}
    store: Store = None
