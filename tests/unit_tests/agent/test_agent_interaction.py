import pytest

from jiuwen.core.agent.task.task_context import AgentRuntime
from jiuwen.core.runtime.interaction.base import AgentInterrupt

pytestmark = pytest.mark.asyncio


async def test_agent_checkpoint():
    session_id = "test"

    # round 1
    runtime = AgentRuntime(session_id)
    await runtime.initialize()
    assert runtime.get_state("a") is None
    runtime.update_state({"a": 1})
    try:
        await runtime.interact("feedback")
    except AgentInterrupt as e:
        assert e.message == "feedback"

    # round 2
    runtime2 = AgentRuntime(session_id)
    await runtime2.initialize()
    assert runtime2.get_state("a") == 1
    runtime2.update_state({"a": 2})

    # round 3
    runtime3 = AgentRuntime(session_id)
    await runtime3.initialize()
    assert runtime3.get_state("a") == 1
