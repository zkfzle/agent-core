"""事件队列模块

该模块实现了事件队列，负责事件的发布和订阅：
- EventQueue: 事件队列，负责事件的发布和订阅

工作流程：
1. 事件订阅：通过subscribe订阅特定类型的事件
2. 事件发布：通过publish_event发布事件到消息队列
3. 事件处理：订阅的事件会被EventHandler相应方法处理
4. 事件取消订阅：通过unsubscribe取消订阅

支持的事件类型：
- INPUT: 用户输入事件
- TASK_INTERACTION: 任务交互事件（任务执行过程中需要用户交互）
- TASK_COMPLETION: 任务完成事件
- TASK_FAILED: 任务失败事件
"""
from typing import Optional, Callable, Awaitable

from openjiuwen.core.controller.base import ControllerConfig
from openjiuwen.core.controller.schema.event import Event, EventType
from openjiuwen.core.controller.modules.event_handler import EventHandlerInput, EventHandler
from openjiuwen.core.runner import Runner
from openjiuwen.core.runner.message_queue.message_queue import MessageQueueBase


class EventQueue:
    """事件队列
    
    负责事件的发布和订阅，将事件分发给事件处理器的相应方法。
    
    基于消息队列实现，支持：
    - 事件的发布（publish_event）
    - 事件的订阅（subscribe）
    - 事件的取消订阅（unsubscribe）
    - 多种事件类型的处理
    
    事件主题格式：{agent_id}_{session_id}_{event_type}
    """
    
    def __init__(
            self,
            config: ControllerConfig,
    ):
        """初始化事件队列
        
        Args:
            config: 控制器配置
        """
        self._config = config
        self._queue: MessageQueueBase = Runner().pubsub()
        self._event_handler: Optional[EventHandler] = None

    def set_event_handler(self, event_handler: EventHandler):
        """设置事件处理器
        
        Args:
            event_handler: 事件处理器实例
        """
        self._event_handler = event_handler

    def _subscribe_event(
            self,
            topic: str,
            event_handle_func: Callable[[EventHandlerInput], Awaitable[...]]
    ):
        """订阅单个事件主题
        
        Args:
            topic: 事件主题
            event_handle_func: 事件处理函数
            
        Returns:
            str: 事件主题
        """
        # 创建订阅
        subscription = self._queue.subscribe(topic)

        # 设置事件处理器（传递 topic 以便访问缓存的 session）
        subscription.set_message_handler(event_handle_func)

        # 激活订阅
        subscription.activate()

        # 保存订阅和控制器映射
        return topic

    async def subscribe(
            self,
            agent_id: str,
            session_id: str
    ) -> (dict[str, str], dict[str, str]):
        """订阅所有事件类型
        
        Args:
            agent_id: Agent ID
            session_id: 会话ID
            
        Returns:
            Tuple[dict, dict]: (订阅字典, 主题字典)
        """
        topics = {}
        subscriptions = {}
        # 订阅输入事件
        topic = self._build_topic(agent_id, session_id, EventType.INPUT)
        sub = self._subscribe_event(topic, self._event_handler.handle_input)
        subscriptions[EventType.INPUT] = sub
        topics[EventType.INPUT] = topic

        # 订阅任务执行中交互事件
        topic = self._build_topic(agent_id, session_id, EventType.TASK_INTERACTION)
        sub = self._subscribe_event(topic, self._event_handler.handle_task_interaction)
        subscriptions[EventType.TASK_INTERACTION] = sub
        topics[EventType.TASK_INTERACTION] = topic

        # 订阅任务完成事件
        topic = self._build_topic(agent_id, session_id, EventType.TASK_COMPLETION)
        sub = self._subscribe_event(topic, self._event_handler.handle_task_completion)
        subscriptions[EventType.TASK_COMPLETION] = sub
        topics[EventType.TASK_COMPLETION] = topic

        # 订阅任务失败事件
        topic = self._build_topic(agent_id, session_id, EventType.TASK_FAILED)
        sub = self._subscribe_event(topic, self._event_handler.handle_task_failed)
        subscriptions[EventType.TASK_FAILED] = sub
        topics[EventType.TASK_FAILED] = topic

        return subscriptions, topics

    async def _unsubscribe_event(
            self,
            topic: str
    ):
        """取消订阅单个事件主题
        
        Args:
            topic: 事件主题
            
        Returns:
            bool: 是否成功
        """
        # 取消订阅
        await self._queue.unsubscribe(topic)
        return True

    async def unsubscribe(
            self,
            agent_id: str,
            session_id: str,
    ):
        """取消订阅所有事件类型
        
        Args:
            agent_id: Agent ID
            session_id: 会话ID
            
        Returns:
            dict: 主题字典
        """
        topics = {}
        # 取消订阅输入事件
        topic = self._build_topic(agent_id, session_id, EventType.INPUT)
        await self._unsubscribe_event(topic)

        # 取消订阅任务执行中交互事件
        topic = self._build_topic(agent_id, session_id, EventType.TASK_INTERACTION)
        await self._unsubscribe_event(topic)

        # 取消订阅任务完成事件
        topic = self._build_topic(agent_id, session_id, EventType.TASK_COMPLETION)
        await self._unsubscribe_event(topic)

        # 取消订阅任务失败事件
        topic = self._build_topic(agent_id, session_id, EventType.TASK_FAILED)
        await self._unsubscribe_event(topic)

        return topics
    
    async def publish_event(
        self,
        agent_id: str,
        session_id: str,
        event: Event
    ) -> None:
        """发布事件到事件队列
        
        Args:
            agent_id: Agent ID
            session_id: 会话ID
            event: 要发布的事件
        """
        topic = self._build_topic(agent_id, session_id, event.event_type)
        # 发布消息
        await self._queue.produce_message(topic, event.model_dump())

    async def unsubscribe_all(self) -> None:
        """取消所有订阅"""
        ...

    @staticmethod
    def _build_topic(agent_id: str, session_id: str, event_type: str) -> str:
        """构建事件主题
        
        Args:
            agent_id: Agent ID
            session_id: 会话ID
            event_type: 事件类型
            
        Returns:
            str: 事件主题字符串
        """
        return f"{agent_id}_{session_id}_{event_type}"


