from jiuwen.core.runner.message_queue_base import MessageQueueType, MessageQueueBase
from jiuwen.core.runner.message_queue_inmemory import MessageQueueInMemory

class MessageQueueFactory:
    _product_map = {
        MessageQueueType.MessageQueueInMemory: MessageQueueInMemory
    }

    @classmethod
    def get_message_queue(cls, type: MessageQueueType = MessageQueueType.MessageQueueInMemory) -> MessageQueueBase:
        queue_class = cls._product_map.get(type)
        if not queue_class:
            raise ValueError(f"Unknown message queue type: {type.value}")
        return queue_class()