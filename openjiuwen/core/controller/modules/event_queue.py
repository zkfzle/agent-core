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

注意：
- EventQueue 在 _subscribe_event 中创建 wrapper，负责将 MessageQueue 的 payload 
  转换为 EventHandler 需要的 EventHandlerInput 格式
- payload 格式：{"session_id": "xxx", "event": {...}}
- wrapper 从 TaskScheduler.sessions 获取 Session 对象
"""
from typing import Callable, Awaitable, Dict, Any

from openjiuwen.core.controller.config import ControllerConfig
from openjiuwen.core.controller.schema.event import (
    Event, EventType, InputEvent, TaskCompletionEvent, 
    TaskInteractionEvent, TaskFailedEvent
)
from openjiuwen.core.controller.modules.event_handler import EventHandlerInput, EventHandler
from openjiuwen.core.common.logging import logger


class EventQueue:
    """事件队列
    
    负责事件的发布和订阅，将事件分发给事件处理器的相应方法。
    
    基于消息队列实现，支持：
    - 事件的发布（publish_event）
    - 事件的订阅（subscribe）
    - 事件的取消订阅（unsubscribe）
    - 多种事件类型的处理
    
    事件主题格式：{agent_id}_{session_id}_{event_type}
    
    工作原理：
    1. MessageQueue 启动后台消费任务
    2. 当 publish_event 被调用时，消息被放入队列
    3. MessageQueue 自动调用注册的回调函数
    4. 回调函数将消息转换为 EventHandlerInput 并调用相应的 EventHandler 方法
    """
    
    def __init__(
            self,
            config: ControllerConfig,
    ):
        """初始化事件队列
        
        Args:
            config: 控制器配置
        """
        # 延迟导入
        from openjiuwen.core.runner.message_queue_inmemory import MessageQueueInMemory
        
        self._config = config
        self._queue: MessageQueueInMemory = MessageQueueInMemory()

    def set_event_handler(self, event_handler: EventHandler):
        """设置事件处理器
        
        Args:
            event_handler: 事件处理器实例
        """
        self._event_handler = event_handler
    
    @staticmethod
    def _deserialize_event(event_data: Dict[str, Any]) -> Event:
        """根据 event_type 反序列化为正确的 Event 子类
        
        Args:
            event_data: 事件数据字典
            
        Returns:
            Event: 反序列化后的事件对象（正确的子类）
            
        Note:
            使用基类 Event.model_validate() 会丢失子类特有的字段（如 task）。
            必须根据 event_type 选择正确的子类进行反序列化。
        """
        event_type = event_data.get("event_type")
        
        if event_type == EventType.INPUT:
            return InputEvent.model_validate(event_data)
        elif event_type == EventType.TASK_COMPLETION:
            return TaskCompletionEvent.model_validate(event_data)
        elif event_type == EventType.TASK_INTERACTION:
            return TaskInteractionEvent.model_validate(event_data)
        elif event_type == EventType.TASK_FAILED:
            return TaskFailedEvent.model_validate(event_data)
        else:
            # 未知类型，使用基类
            return Event.model_validate(event_data)

    def start(self):
        """启动事件队列的消息处理
        
        启动MessageQueue的后台消费任务。
        """
        self._queue.start()

    async def stop(self):
        """停止事件队列的消息处理
        
        停止MessageQueue的后台消费任务。
        """
        await self._queue.stop()

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
            
        Note:
            创建一个 wrapper 函数，负责将 MessageQueue 的 payload 转换为
            EventHandlerInput 格式。payload 包含 session_id 和 event，wrapper 
            从 TaskScheduler.sessions 获取真实的 Session 对象。
        """
        # 创建订阅
        subscription = self._queue.subscribe(topic)

        # 创建 wrapper：从 payload 构造 EventHandlerInput
        async def message_handler_wrapper(payload: Any):
            """适配层：将 MessageQueue 的 payload 转换为 EventHandlerInput
            
            Args:
                payload: 消息队列传来的 payload，格式：
                        {"session_id": "xxx", "event": {...}}
            
            Returns:
                EventHandler 方法的返回值
                
            Raises:
                KeyError: 当 payload 缺少必要字段时
                RuntimeError: 当无法获取 Session 时
            """
            # 1. 提取 session_id 和 event
            session_id = payload.get("session_id")
            event_data = payload.get("event")
            
            if session_id is None:
                raise KeyError("Missing 'session_id' in payload")
            if event_data is None:
                raise KeyError("Missing 'event' in payload")
            
            # 2. 从 TaskScheduler 获取 session
            if self._event_handler is None:
                raise RuntimeError("EventHandler not set in EventQueue")
            if self._event_handler._task_scheduler is None:
                raise RuntimeError("TaskScheduler not set in EventHandler")
            
            session = self._event_handler.task_scheduler.sessions.get(session_id)
            if session is None:
                raise RuntimeError(f"Session {session_id} not found in TaskScheduler")
            
            # 3. 反序列化 Event - 根据 event_type 选择正确的子类
            event = self._deserialize_event(event_data)
            
            # 4. 构造 EventHandlerInput
            handler_input = EventHandlerInput(event=event, session=session)
            
            # 5. 调用 EventHandler 的原始方法（签名未变）
            return await event_handle_func(handler_input)
        
        # 设置 wrapper 作为 message_handler
        subscription.set_message_handler(message_handler_wrapper)

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
        """发布事件到事件队列并等待处理完成
        
        Args:
            agent_id: Agent ID
            session_id: 会话ID
            event: 要发布的事件
            
        Note:
            - payload 包含 session_id 和 event，session_id 用于 EventQueue 的
              wrapper 从 TaskScheduler.sessions 获取 Session 对象。
            - 此方法会等待 EventHandler 处理完成后才返回，确保事件处理的顺序性。
            - 这样设计支持未来的分布式扩展，session_id 可用于从分布式存储获取 Session。
        
        Raises:
            Exception: 如果 EventHandler 处理事件时抛出异常
        """
        topic = self._build_topic(agent_id, session_id, event.event_type)
        
        # 延迟导入，避免循环导入
        from openjiuwen.core.runner.message_queue_base import InvokeQueueMessage
        
        queue_message = InvokeQueueMessage()
        # 构造 payload，包含 session_id 和 event
        # 使用 mode='json' 确保所有对象都能正确序列化
        # exclude_none=True 排除 None 值，减少数据量
        queue_message.payload = {
            "session_id": session_id,
            "event": event.model_dump(mode='json', exclude_none=True)
        }
        
        # 发布消息
        await self._queue.produce_message(topic, queue_message)
        
        # 等待 EventHandler 处理完成
        try:
            await queue_message.response
        except Exception as e:
            logger.error(f"Event handler failed for {event.event_type}: {e}", exc_info=True)
            raise

    async def unsubscribe_all(self) -> None:
        """取消所有订阅"""
        await self._queue.stop()

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
