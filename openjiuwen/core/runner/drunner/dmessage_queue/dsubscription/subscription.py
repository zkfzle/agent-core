from typing import Optional

from openjiuwen.core.runner.message_queue_base import SubscriptionBase, AsyncMessageHandler, QueueMessage



class DSubscription(SubscriptionBase):
    def __init__(self, mq, topic: str):
        self._mq = mq
        self.topic = topic
        self._handler: Optional[AsyncMessageHandler] = None
        self._active = False

    def set_message_handler(self, handler: AsyncMessageHandler):
        self._handler = handler

    def activate(self):
        self._active = True

    def deactivate(self):
        self._active = False

    def is_active(self) -> bool:
        return self._active

    async def handle_message(self, msg: QueueMessage):
        if self._handler:
            await self._handler(msg)