import asyncio
import socket
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

from jiuwen.core.runner.drunner.dmessage_queue.message import DmqResponseMessage
from jiuwen.core.runner.drunner.dmessage_queue.dsubscription.response_collector import ResponseCollector
from jiuwen.core.runner.drunner.dmessage_queue.dsubscription.subscription import DSubscription
from jiuwen.core.runner.drunner.dmessage_queue.message_queue import FakeMQ
from jiuwen.core.common.logging import logger
from jiuwen.core.runner.drunner.common.constants import AGENT_TOPIC_TEMPLATE, REPLY_TOPIC_TEMPLATE

MAX_COLLECTORS = 10000  # 系统最多允许的collector数


@dataclass(frozen=True)
class CollectorKey:
    sender_id: str
    message_id: str
    request_id: Optional[str] = None


class ReplyTopicSubscription(DSubscription):
    """负责监听 reply_topic 并分发响应到对应 ResponseCollector"""

    def __init__(self, mq: Optional[FakeMQ] = None, topic: str = None):
        super().__init__(mq, topic)
        # TOD   O: Get IP
        self.topic = topic or REPLY_TOPIC_TEMPLATE.format(instance_id=self._get_local_ip())
        self.collectors: dict[CollectorKey, ResponseCollector] = {}

    def activate(self):
        """初始化"""
        super().set_message_handler(self.on_message)
        super().activate()
        logger.info(f"[ReplyTopicSubscription] activated topic={self.topic}")

    async def deactivate(self):
        """清理所有 collectors"""
        if not self._active:
            return

        logger.info(f"[ReplyTopicSubscription] Stopping subscription")
        super().deactivate()

        # 清理所有 collectors
        await self.unregister_collector()
        logger.info(f"[ReplyTopicSubscription] Stopped")

    def _get_local_ip(self) -> str:
        """获取本地可用 IPv4 地址（非 127.0.0.1）"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip

    def _make_key(self, sender_id: str, message_id: str, request_id: Optional[str] = None) -> CollectorKey:
        """构造 collector 唯一键"""
        request_id = request_id or None
        return CollectorKey(sender_id, message_id, request_id)

    async def on_message(self, msg: DmqResponseMessage):
        """消息分发给对应的 ResponseCollector"""
        key = self._make_key(msg.sender_id, msg.message_id, msg.request_id)
        logger.info(f"[ReplyTopicSubscription] receive message key={key}")

        collector = self.collectors.get(key)
        if collector:
            await collector.put_message(msg)
        else:
            logger.info(f"[ReplyTopicSubscription] No collector for {key}, discard message")

    async def register_collector(self, message_id: str, remote_id: str, request_id: Optional[str] = None,
                                 ttl: float = 30.0) -> ResponseCollector:
        """注册 collector，用于等待对应返回"""
        if not self.is_active():
            raise asyncio.CancelledError(f"ReplyTopicSubscription was cancelled")
        if len(self.collectors) >= MAX_COLLECTORS:
            raise RuntimeError(f"[ReplyTopicSubscription] Too many collectors ({MAX_COLLECTORS})")

        key = self._make_key(remote_id, message_id, request_id)
        if key in self.collectors:
            raise RuntimeError(f"[ReplyTopicSubscription] Collector already exists for {key}")

        collector = ResponseCollector(message_id, remote_id, ttl)
        self.collectors[key] = collector
        logger.info(f"[ReplyTopicSubscription] register collector for {key}")
        return collector

    async def unregister_collector(
            self,
            message_id: Optional[str] = None,
            receiver_id: Optional[str] = None,
            request_id: Optional[str] = None,
    ):
        """
        按 message_id / receiver_id / request_id / 全部 清理
        - 若全为 None，则表示清理全部 collector
        """
        logger.info(
            f"[ReplyTopicSubscription] unregister_collector message_id: {message_id}, receiver_id: {receiver_id}, "
            f"receiver_id:{receiver_id}")

        if not self.collectors:
            return

        # 过滤目标
        keys_to_remove = []
        for key, collector in self.collectors.items():
            if (
                    message_id is None and receiver_id is None and request_id is None
            ) or (
                    (message_id and key.message_id == message_id)
                    or (receiver_id and key.sender_id == receiver_id)
                    or (request_id and key.request_id == request_id)
            ):
                keys_to_remove.append(key)

        if not keys_to_remove:
            logger.info(
                f"[ReplyTopicSub] No matching collectors for message_id={message_id}, "
                f"receiver_id={receiver_id}, request_id={request_id}"
            )
            return

        logger.info(
            f"[ReplyTopicSub] unregistering {len(keys_to_remove)} collectors "
            f"(msg_id={message_id}, recv_id={receiver_id}, req_id={request_id})"
        )

        tasks = []
        for key in keys_to_remove:
            collector = self.collectors.pop(key, None)
            if collector:
                tasks.append(collector.close())

        # 并发关闭
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"[ReplyTopicSub] unregistered {len(keys_to_remove)} collectors")
