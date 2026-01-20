"""意图识别模块

该模块实现了基于意图识别的事件处理，包括：
- IntentRecognizer: 意图识别器，识别用户输入中的意图
- EventHandlerWithIntentRecognition: 基于意图识别的事件处理器

工作流程：
1. 接收输入事件
2. 通过IntentRecognizer识别意图
3. 根据意图类型调用相应的处理方法

支持的意图类型：
- CREATE_TASK: 创建新任务
- PAUSE_TASK: 暂停任务
- RESUME_TASK: 恢复任务
- CONTINUE_TASK: 接续任务
- SUPPLEMENT_TASK: 补充任务信息
- CANCEL_TASK: 取消任务
- MODIFY_TASK: 修改任务
- SWITCH_TASK: 切换任务
- UNKNOWN_TASK: 未知意图
"""
from abc import ABC, abstractmethod

from openjiuwen.core.context_engine import ContextEngine
from openjiuwen.core.controller.base import ControllerConfig
from openjiuwen.core.controller.modules.event_handler import EventHandler, EventHandlerInput
from openjiuwen.core.controller.modules.task_manager import TaskManager
from openjiuwen.core.controller.schema import Intent
from openjiuwen.core.controller.schema.event import Event
from openjiuwen.core.session import Session
from openjiuwen.core.single_agent.base import AbilityManager


class IntentRecognizer:
    """意图识别器
    
    负责识别用户输入中的意图，将事件转换为Intent对象。
    """
    
    def __init__(
            self,
            config: ControllerConfig,
            task_manager: TaskManager,
            ability_manager: AbilityManager,
            context_engine: ContextEngine
    ):
        """初始化意图识别器
        
        Args:
            config: 控制器配置
            task_manager: 任务管理器
            ability_manager: 能力包
            context_engine: 上下文引擎
        """
        self._config = config
        self._task_manager = task_manager
        self._context_engine = context_engine
        self._ability_manager = ability_manager

    async def recognize(self, event: Event, session: Session) -> Intent:
        """识别意图
        
        Args:
            event: 输入事件
            session: 会话对象
            
        Returns:
            Intent: 识别出的意图对象
        """
        ...


class EventHandlerWithIntentRecognition(ABC, EventHandler):
    """基于意图识别的事件处理器
    
    在EventHandler的基础上增加意图识别功能，根据识别出的意图调用相应的处理方法。
    """
    def __init__(self):
        super().__init__()
        self.recognizer = IntentRecognizer(
            self._config,
            self.task_manager,
            self.ability_manager,
            self.context_engine
        )

    async def handle_input(self, inputs: EventHandlerInput):
        """处理输入事件
        
        识别输入意图，并调用相应方法处理意图，可重写。
        
        Args:
            inputs: 事件处理器输入
        """
        ...

    async def handle_task_interaction(self, inputs: EventHandlerInput):
        """处理任务交互事件
        
        将interaction直接抛出给用户，可重写。
        
        Args:
            inputs: 事件处理器输入
        """
        ...

    async def handle_task_completion(self, inputs: EventHandlerInput):
        """处理任务完成事件
        
        将任务完成信息抛出给用户，可重写。
        
        Args:
            inputs: 事件处理器输入
        """
        ...

    async def handle_task_failed(self, inputs: EventHandlerInput):
        """处理任务失败事件
        
        将错误信息抛出给用户，可重写。
        
        Args:
            inputs: 事件处理器输入
        """
        ...

    @abstractmethod
    async def _process_create_task_intent(self, inputs: EventHandlerInput):
        """处理创建任务意图
        
        用户自定义执行新任务逻辑。
        
        Args:
            inputs: 事件处理器输入
        """
        ...

    async def _process_pause_task_intent(self, inputs: EventHandlerInput):
        """处理暂停任务意图
        
        调用 task_scheduler 的 pause_task 方法打断目标任务。
        
        Args:
            inputs: 事件处理器输入
        """
        ...

    async def _process_resume_task_intent(self, inputs: EventHandlerInput):
        """处理恢复任务意图
        
        将要恢复的任务的状态置为 submitted。
        
        Args:
            inputs: 事件处理器输入
        """
        ...

    async def _process_continue_task_intent(self, inputs: EventHandlerInput):
        """处理接续任务意图
        
        根据依赖任务的上下文调用 _process_create_task_intent 执行目标任务。
        
        Args:
            inputs: 事件处理器输入
        """
        ...

    async def _process_supplement_task_intent(self, inputs: EventHandlerInput):
        """处理补充任务意图
        
        基于补充的信息调用_process_create_task_intent继续执行目标任务。
        
        Args:
            inputs: 事件处理器输入
        """
        ...

    async def _process_cancel_task_intent(self, inputs: EventHandlerInput):
        """处理取消任务意图
        
        调用 task_scheduler 的 cancel_task 方法取消目标任务。
        
        Args:
            inputs: 事件处理器输入
        """
        ...

    async def _process_modify_task_intent(self, inputs: EventHandlerInput):
        """处理修改任务意图
        
        修改目标任务后，将其状态置为 submitted。
        
        Args:
            inputs: 事件处理器输入
        """
        ...

    async def _process_switch_task_intent(self, inputs: EventHandlerInput):
        """处理切换任务意图
        
        打断所有正在执行的任务，再调用 _process_create_task_intent 执行目标任务。
        
        Args:
            inputs: 事件处理器输入
        """
        ...

    async def _process_unknown_task_intent(self, event: Event):
        """处理未知任务意图
        
        返回 Intent 的 clarification_prompt 字段给用户。
        
        Args:
            event: 输入事件
        """
        ...

