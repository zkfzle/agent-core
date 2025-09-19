from jiuwen.core.agent.task.task_context import AgentRuntime
from jiuwen.core.runtime.store import Store


class AgentContext:
    context_map: dict[str, AgentRuntime] = {}
    store: Store = None
