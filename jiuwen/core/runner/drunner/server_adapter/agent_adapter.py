from typing import Any, AsyncIterator

from jiuwen.core.runner.drunner.common.constants import AGENT_TOPIC_TEMPLATE
from jiuwen.core.runner.drunner.server_adapter.mq_server_adapter import MqServerAdapter


class MqAgentAdapter:
    """AgentAdapter"""

    def __init__(self, agent_id: str, version: str = ""):
        self.agent_id = agent_id
        self.version = version
        self.topic = AGENT_TOPIC_TEMPLATE.format(agent_id=agent_id, version=version)

        self.server = MqServerAdapter(
            adapter_id=agent_id,
            topic=self.topic,
            invoke_handler=self.handle_invoke,
            stream_handler=self.handle_stream
        )

    def start(self):
        self.server.start()

    def stop(self):
        self.server.stop()

    async def handle_invoke(self, inputs: dict) -> Any:
        return {"INVOKE": "INVOKE"}

    async def handle_stream(self, inputs: dict) -> AsyncIterator[Any]:
        for i in range(2):
            yield {"STREAM": i}
