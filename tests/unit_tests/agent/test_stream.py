import asyncio

import pytest

from jiuwen.core.agent.task.task_context import AgentRuntime

pytestmark = pytest.mark.asyncio

async def test_agent_stream():
    session_id = "test"
    runtime = AgentRuntime(session_id)

    async def consumer():
        i = runtime.stream_iterator()
        async for chunk in i:
            # logger.info("hello")
            print(chunk)

    async def producer():
        for i in range(10):
           await runtime.write_stream({"name": "hi"})
        await runtime.close()

    task1 = asyncio.create_task(consumer())
    task2 = asyncio.create_task(producer())

    await asyncio.gather(task1, task2)
