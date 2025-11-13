import asyncio
from typing import Optional

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.runner.drunner.remote_client.mq_remote_clent import MqRemoteClient
from openjiuwen.core.runner.drunner.remote_client.remote_client import RemoteClient
from openjiuwen.core.runner.drunner.remote_client.remote_client_config import RemoteClientConfig, ProtocolEnum
from openjiuwen.core.runner.drunner.common.constants import AGENT_TOPIC_TEMPLATE


class RemoteAgent:

    def __init__(self, agent_id: str, version: str = "", description: str = None, topic: str = None,
                 protocol: str = ProtocolEnum.MQ, config: dict = None, ):
        self.agent_id = agent_id
        self.version = version
        self.description = description
        # Use template if topic not provided
        self.topic = topic or AGENT_TOPIC_TEMPLATE.format(agent_id=agent_id, version=self.version)
        self.protocol = protocol
        self.config = RemoteClientConfig(id=agent_id, protocol=protocol, topic=self.topic, **(config or {}))
        self.client = self._create_client()

    def _create_client(self) -> RemoteClient:
        if self.protocol == ProtocolEnum.MQ:
            client = MqRemoteClient(config=self.config)
            return client

    async def invoke(self, inputs: dict, timeout: float = None):
        try:
            await self.client.start()
            return await self.client.invoke(inputs, timeout=timeout)
        except asyncio.CancelledError as e:
            # Runner stop导致client取消
            raise JiuWenBaseException(StatusCode.RUNNER_DISTRIBUTED_MODE_REQUIRED.code,
                                      StatusCode.RUNNER_DISTRIBUTED_MODE_REQUIRED.errmsg.format(str(e)))

    async def stream(self, inputs: dict, timeout: float = None):
        try:
            await self.client.start()
            async for chunk in self.client.stream(inputs, timeout=timeout):
                yield chunk
        except asyncio.CancelledError as e:
            # Runner stop导致client取消
            raise JiuWenBaseException(StatusCode.RUNNER_DISTRIBUTED_MODE_REQUIRED.code,
                                      StatusCode.RUNNER_DISTRIBUTED_MODE_REQUIRED.errmsg.format(str(e)))
