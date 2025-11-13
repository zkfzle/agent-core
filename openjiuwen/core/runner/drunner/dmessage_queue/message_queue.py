import asyncio
from typing import Callable, List, Dict, Set, Optional

from openjiuwen.core.runner.drunner.dmessage_queue.dsubscription.subscription import DSubscription
from openjiuwen.core.runner.message_queue_base import MessageQueueBase, QueueMessage
from openjiuwen.core.common.logging import logger


class FakeMQ(MessageQueueBase):
    def __init__(self):
        # Multiple subscribers per topic
        self._topics: Dict[str, List[DSubscription]] = {}
        self._running = False

    def start(self):
        self._running = True
        logger.info("[FakeMQ] started")

    async def stop(self):
        self._running = False
        logger.info("[FakeMQ] stopped")

    def subscribe(self, topic: str, sub: Optional[DSubscription] = None) -> DSubscription:
        if sub is None:
            sub = DSubscription(self, topic)
            self._topics.setdefault(topic, []).append(sub)
            return sub
        else:
            self._topics.setdefault(topic, []).append(sub)
            return sub

    def unsubscribe(self, topic: str):
        subs = self._topics.get(topic)


    async def produce_message(self, topic: str, message: QueueMessage):
        subs = list(self._topics.get(topic, []))
        if not subs:
            return
        tasks = []
        for s in subs:
            if s.is_active():
                tasks.append(s.handle_message(message))
        if tasks:
            await asyncio.gather(*tasks)
            logger.info(f"[FakeMQ] Produced message to topic {s.topic} handler success")
