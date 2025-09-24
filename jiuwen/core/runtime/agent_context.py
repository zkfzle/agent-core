from jiuwen.core.runtime.runtime import Runtime
from jiuwen.core.runtime.store import Store


class AgentContext:
    context_map: dict[str, Runtime] = {}
    store: Store = None
