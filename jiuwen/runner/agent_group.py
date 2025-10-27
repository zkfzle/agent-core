from jiuwen.runner.message_queue import Subscription


class AgentGroup:
    def __init__(self):
        self._subscription: Subscription = None
        self._message_handler = None
        self._topic: str = None

    def set_subscription(self, subscription: Subscription):
        self._subscription = subscription
        self._subscription.set_message_handler(message_handler=self._message_handler)
        self._subscription.activate()

    def get_topic(self):
        pass
