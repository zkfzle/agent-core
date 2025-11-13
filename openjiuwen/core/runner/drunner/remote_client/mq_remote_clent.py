import asyncio
import time
import uuid
from typing import Optional, Dict, AsyncGenerator

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.logging import logger
from openjiuwen.core.runner.drunner.dmessage_queue.message_queue import FakeMQ
from openjiuwen.core.runner.drunner.dmessage_queue.message import DmqRequestMessage, DMessageType
from openjiuwen.core.runner.drunner.dmessage_queue.dsubscription.reply_topic_subscription import ReplyTopicSubscription
from openjiuwen.core.runner.drunner.remote_client.remote_client import RemoteClient
from openjiuwen.core.runner.drunner.remote_client.remote_client_config import RemoteClientConfig
from openjiuwen.core.runner.message_queue_base import SubscriptionBase

DEFAULT_TTL = 30.0


class MqRemoteClient(RemoteClient):
    def __init__(self, config: RemoteClientConfig):
        self.mq: FakeMQ | None = None
        self.topic = config.topic
        self.remote_id = config.id
        self.config = config
        self._started = False
        self.system_reply_sub: Optional[ReplyTopicSubscription] = None
        self.request_subscription: Optional[SubscriptionBase] = None

    async def start(self):
        if self._started:
            return
        from openjiuwen.core.runner.runner import Runner
        self.mq = Runner._mq
        self.system_reply_sub = Runner.system_reply_sub
        if self.system_reply_sub is None:
            raise JiuWenBaseException(StatusCode.RUNNER_DISTRIBUTED_MODE_REQUIRED.code,
                                      StatusCode.RUNNER_DISTRIBUTED_MODE_REQUIRED.errmsg.format(
                                          "reply topic not initialized"))
        self.reply_topic = self.system_reply_sub.topic
        self._started = True
        logger.debug(f"[MqRemoteClient] init success topic: {self.topic}, reply_topic: {self.reply_topic}")

    async def stop(self):
        # ReplyTopic handle all collector unregistration
        self._started = False
        logger.info(f"[MqRemoteClient] Stopped client for {self.remote_id}")

    async def invoke(self, input: Dict, timeout: float = DEFAULT_TTL) -> Dict:
        message_id = str(uuid.uuid4())
        if timeout is None:
            timeout = DEFAULT_TTL
        logger.info(f"[MqRemoteClient] Invoke with message_id: {message_id}")

        # Register response collector
        collector = await self.system_reply_sub.register_collector(message_id, self.remote_id)

        # Build request message
        request_msg = DmqRequestMessage(
            type=DMessageType.INPUT,
            reply_topic=self.reply_topic,
            message_id=message_id,
            sender_id=self.reply_topic,
            receiver_id=self.remote_id,
            enable_stream=False,
            payload=input,
            expire_at=time.time() + timeout,
        )
        # Send message
        logger.info(f"[MqRemoteClient] Publishing to topic: {self.topic}")
        await self.mq.produce_message(self.topic, request_msg)

        try:
            # Wait for response
            response = await collector.result()
            return response
        except asyncio.CancelledError:
            # 流被取消时发送 STOP 消息
            logger.info(f"[MQRemoteClient] Stream {message_id} cancelled, sending STOP")
            await self._send_stop_message(message_id)
            raise
        except JiuWenBaseException:
            raise
        except Exception as e:
            raise JiuWenBaseException(StatusCode.REMOTE_AGENT_REQUEST_TIMEOUT.code,
                                      StatusCode.REMOTE_AGENT_REQUEST_TIMEOUT.errmsg.format(
                                          self.remote_id))
        finally:
            # Cleanup collector
            await self.system_reply_sub.unregister_collector(message_id)

    async def stream(self, inputs: Dict, timeout: float = DEFAULT_TTL) -> AsyncGenerator[Dict, None]:
        message_id = str(uuid.uuid4())
        if timeout is None:
            timeout = DEFAULT_TTL
        logger.info(f"[MQRemoteClient] Stream with message_id: {message_id}")

        collector = await self.system_reply_sub.register_collector(message_id, self.remote_id, ttl=timeout)

        request_msg = DmqRequestMessage(
            type=DMessageType.INPUT,
            reply_topic=self.reply_topic,
            message_id=message_id,
            sender_id=self.reply_topic,
            receiver_id=self.remote_id,
            enable_stream=True,
            payload=inputs,
            expire_at=time.time() + timeout,
        )

        logger.info(f"[MQRemoteClient] Publishing to topic: {self.topic}")
        await self.mq.produce_message(self.topic, request_msg)

        try:
            async for response in collector.stream():
                yield response
        except asyncio.CancelledError:
            logger.info(f"[MQRemoteClient] Stream {message_id} cancelled, sending STOP")
            await self._send_stop_message(message_id)
            raise
        except JiuWenBaseException:
            raise
        except Exception as e:
            raise JiuWenBaseException(StatusCode.REMOTE_AGENT_REQUEST_TIMEOUT.code,
                                      StatusCode.REMOTE_AGENT_REQUEST_TIMEOUT.errmsg.format(
                                          self.remote_id))
        finally:
            await self.system_reply_sub.unregister_collector(message_id, self.remote_id)

    async def _send_stop_message(self, message_id: str):
        """发送 STOP 消息 message里带了过期时间，超时就不用发stop消息了，只有提前close的时候发"""
        try:
            stop_msg = DmqRequestMessage(
                type=DMessageType.STOP,
                message_id=message_id,
                sender_id=self.reply_topic,
                receiver_id=self.remote_id,
                expire_at=time.time() + DEFAULT_TTL,
            )
            await self.mq.produce_message(self.topic, stop_msg)
            logger.info(f"[MqRemoteClient] Sent STOP message for {message_id}")
        except Exception as e:
            logger.error(f"[MqRemoteClient] Failed to send STOP message: {e}")
