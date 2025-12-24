from typing import Optional
from openjiuwen.core.session.interaction.checkpointer import Checkpointer

_default_inmemory_checkpointer: Optional[Checkpointer] = None


def get_default_inmemory_checkpointer() -> Checkpointer:
    global _default_inmemory_checkpointer

    if _default_inmemory_checkpointer is None:
        from openjiuwen.core.session.interaction.checkpointer import InMemoryCheckpointer
        _default_inmemory_checkpointer = InMemoryCheckpointer()

    return _default_inmemory_checkpointer
