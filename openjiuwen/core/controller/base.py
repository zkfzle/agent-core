"""控制器基类定义

包含的主要类：
- ControllerConfig: 控制器状态枚举
- BaseController: 控制器基类

控制器负责处理事件，管理任务生命周期，并进行意图识别和处理。
"""
import asyncio
from typing import AsyncIterator, Optional, List, Callable, TYPE_CHECKING

from openjiuwen.core.context_engine import ContextEngine
from openjiuwen.core.controller.schema import ControllerOutput, ControllerOutputChunk, EventType, InputEvent
from openjiuwen.core.controller.modules import TaskManager, TaskScheduler, TaskExecutor, EventQueue, EventHandler
from openjiuwen.core.controller.config import ControllerConfig
from openjiuwen.core.session import Session
from openjiuwen.core.session.stream.base import BaseStreamMode, StreamMode
from openjiuwen.core.session.stream.manager import StreamWriterManager
from openjiuwen.core.session.stream.emitter import StreamEmitter
from openjiuwen.core.session.tracer.tracer import Tracer
from openjiuwen.core.single_agent import AgentCard
from openjiuwen.core.common.logging import logger
from openjiuwen.core.common.exception.errors import build_error, BaseError
from openjiuwen.core.common.exception.codes import StatusCode

if TYPE_CHECKING:
    from openjiuwen.core.single_agent.agent import AbilityKit


class Controller:
    """控制器
    
    负责处理事件，管理任务生命周期。
    是ControllerAgent的核心组件。
    """
    
    def __init__(self):
        """初始化控制器"""
        super().__init__()
        self._card: Optional[str] = None
        self._ability_kit: Optional['AbilityKit']  = None
        self._config: Optional[ControllerConfig] = None
        self._context_engine: Optional[ContextEngine] = None
        self._task_manager: Optional[TaskManager] = None
        self._event_queue: Optional[EventQueue] = None
        self._task_scheduler: Optional[TaskScheduler] = None
        self._event_handler: Optional[EventHandler] = None

        # Event loop tracking
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def init(
            self,
            card: AgentCard,
            config: ControllerConfig,
            ability_kit: 'AbilityKit',
            context_engine: ContextEngine
    ):
        """初始化控制器
        
        Args:
            card: Agent名片
            config: 控制器配置
            ability_kit: 能力包
            context_engine: 上下文引擎
        """
        self._card = card
        self._config = config
        self._ability_kit = ability_kit
        self._context_engine = context_engine
        self._task_manager = TaskManager(config=self._config)
        self._event_queue = EventQueue(config=self._config)
        self._task_scheduler = TaskScheduler(
            config=self._config,
            task_manager=self._task_manager,
            context_engine=self._context_engine,
            ability_kit=self._ability_kit,
            event_queue=self._event_queue,
            card=card
        )

    @property
    def event_queue(self) -> EventQueue:
        """获取事件队列"""
        return self._event_queue

    @property
    def config(self) -> ControllerConfig:
        """获取控制器配置"""
        return self._config

    @property
    def context_engine(self) -> ContextEngine:
        """获取上下文引擎"""
        return self._context_engine

    @property
    def ability_kit(self) -> 'AbilityKit':
        """获取能力包"""
        return self._ability_kit

    @config.setter
    def config(self, config: ControllerConfig):
        """设置控制器配置"""
        self._config = config

    @context_engine.setter
    def context_engine(self, context_engine: ContextEngine):
        """设置上下文引擎"""
        self._context_engine = context_engine

    @ability_kit.setter
    def ability_kit(self, ability_kit: 'AbilityKit'):
        """设置能力包"""
        self._ability_kit = ability_kit

    def set_event_handler(self, event_handler: EventHandler):
        """设置事件处理器
        
        Args:
            event_handler: 事件处理器实例
        """
        self._event_handler = event_handler
        self._event_handler.config = self._config
        self._event_handler.context_engine = self._context_engine
        self._event_handler.task_scheduler = self._task_scheduler
        self._event_handler.task_manager = self._task_manager
        self._event_handler.ability_kit = self._ability_kit

    def add_task_executor(
            self,
            task_type: str,
            task_executor_builder: Callable[
                [ControllerConfig, 'AbilityKit', ContextEngine, TaskManager, EventQueue], TaskExecutor
            ]
    ) -> "Controller":
        """添加任务执行器
        
        Args:
            task_type: 任务类型
            task_executor_builder: 任务执行器构建函数
            
        Returns:
            self（支持链式调用）
        """
        self._task_scheduler.task_executor_registry.add_task_executor(task_type, task_executor_builder)
        return self

    def remove_task_executor(self, task_type: str):
        """移除任务执行器
        
        Args:
            task_type: 任务类型
        """
        self._task_scheduler.task_executor_registry.remove_task_executor(task_type)

    def _restore_task_manager_state(self, session: Session) -> bool:
        """从 session 恢复 TaskManager 状态
        
        Args:
            session: 会话对象
            
        Returns:
            bool: 恢复是否成功。True-成功恢复，False-无状态或恢复失败
        """
        controller_state = session.get_state("controller")
        
        if not controller_state or "task_manager_state" not in controller_state:
            # No saved state, clear all task manager state
            logger.info(f"No saved state found for session {session.session_id()}, clearing task manager")
            self._task_manager.clear_state()
            return False
        
        try:
            logger.info(f"Restoring TaskManager state for session {session.session_id()}")
            from openjiuwen.core.controller.modules.task_manager import TaskManagerState
            
            # Deserialize TaskManagerState from dict
            state_dict = controller_state["task_manager_state"]
            task_manager_state = TaskManagerState.model_validate(state_dict)
            
            # Load state into task_manager
            self._task_manager.load_state(task_manager_state)
            logger.info(
                f"Successfully restored TaskManager state: "
                f"{len(task_manager_state.tasks)} tasks, "
                f"{len(task_manager_state.root_tasks)} root tasks"
            )
            return True
            
        except Exception as e:
            logger.error(
                f"Failed to restore TaskManager state for session {session.session_id()}: {e}, "
                f"clearing task manager state instead",
                exc_info=True
            )
            # Fallback: clear all task manager state to allow user to continue
            self._task_manager.clear_state()
            return False

    def _save_task_manager_state(self, session: Session) -> None:
        """保存 TaskManager 状态到 session
        
        Args:
            session: 会话对象
        """
        try:
            task_manager_state = self._task_manager.get_state()
            controller_state = {
                "task_manager_state": task_manager_state.model_dump()
            }
            
            # Clear old state first, then update with new state (ensures proper cleanup of nested dict keys)
            session.update_state({"controller": None})
            session.update_state({"controller": controller_state})
            
            logger.info(
                f"Saved TaskManager state for session {session.session_id()}: "
                f"{len(task_manager_state.tasks)} tasks, "
                f"{len(task_manager_state.root_tasks)} root tasks"
            )
        except Exception as e:
            logger.error(f"Failed to save TaskManager state for session {session.session_id()}: {e}")

    def _init_session_stream_manager(self, session: Session, stream_modes: Optional[List[StreamMode]]) -> None:
        """初始化 Session 的 StreamWriterManager
        
        Args:
            session: 会话对象
            stream_modes: 流式输出模式列表
            
        Note:
            - 如果 session 已有 stream_writer_manager，不做处理（尊重外部配置）
            - 如果 stream_modes 为 None，使用默认值（OUTPUT, TRACE, CUSTOM）
            - 确保 OUTPUT mode 始终启用，以保证 all_tasks_processed 消息能正常传递
            - 如果指定了 TRACE mode，会初始化 Tracer
        """
        # Check if session supports stream_writer_manager
        if not hasattr(session, 'stream_writer_manager'):
            logger.warning(f"Session {session.session_id()} does not support stream_writer_manager")
            return
        
        # Check if already initialized (e.g., AgentSession has default manager)
        existing_manager = session.stream_writer_manager()
        if existing_manager is not None:
            logger.warning(
                f"Session {session.session_id()} already has stream_writer_manager, "
                f"skipping initialization"
            )
            return
        
        # Validate stream_modes: OUTPUT must be included for all_tasks_processed message
        if stream_modes is not None:
            if BaseStreamMode.OUTPUT not in stream_modes:
                logger.warning(
                    f"OUTPUT mode not in stream_modes {stream_modes}, adding it to ensure "
                    f"all_tasks_processed message can be delivered"
                )
                stream_modes = [BaseStreamMode.OUTPUT] + list(stream_modes)
        
        # Initialize StreamWriterManager with specified modes
        if hasattr(session, 'set_stream_writer_manager'):
            manager = StreamWriterManager(stream_emitter=StreamEmitter(), modes=stream_modes)
            session.set_stream_writer_manager(manager)
            logger.info(
                f"Initialized stream_writer_manager for session {session.session_id()} "
                f"with modes: {stream_modes or 'default (OUTPUT, TRACE, CUSTOM)'}"
            )
            
            # Initialize Tracer if TRACE mode is enabled
            if hasattr(session, 'set_tracer') and hasattr(session, 'tracer'):
                try:
                    if session.tracer() is None and (stream_modes is None or BaseStreamMode.TRACE in stream_modes):
                        tracer = Tracer()
                        if hasattr(session, 'callback_manager'):
                            tracer.init(manager, session.callback_manager())
                            session.set_tracer(tracer)
                            logger.info(f"Initialized tracer for session {session.session_id()}")
                except Exception as e:
                    logger.error(f"Failed to initialize tracer: {e}")

    def get_task_executor(
            self,
            config: ControllerConfig,
            ability_kit: 'AbilityKit',
            context_engine: ContextEngine,
            task_manager: TaskManager
    ) -> TaskExecutor:
        """获取任务执行器
        
        Args:
            config: 控制器配置
            ability_kit: 能力包
            context_engine: 上下文引擎
            task_manager: 任务管理器
            
        Returns:
            TaskExecutor: 任务执行器实例
        """
        return self._task_scheduler.task_executor_registry.get_task_executor(
            config,
            ability_kit,
            context_engine,
            task_manager
        )

    async def start(self):
        """启动控制器
        
        启动任务调度器，开始处理任务，启动事件队列
        """
        await self._event_queue.start()
        await self._task_scheduler.start()

    async def stop(self):
        """停止控制器
        
        停止任务调度器，停止处理任务，停止事件队列
        """
        await self._task_scheduler.stop()
        await self._event_queue.stop()

    async def invoke(
            self,
            inputs: 'InputEvent',
            session: Session,
            **kwargs
    ) -> ControllerOutput:
        """批执行控制器
        
        Args:
            inputs: 输入事件
            session: 会话对象
            **kwargs: 其他参数
            
        Returns:
            ControllerOutput: 控制器输出结果
            
        Note:
            1. 调用 stream 方法
            2. 将流式消息转为批消息返回
        """
        # 收集所有流式输出块，只获取 OUTPUT 模式数据
        chunks: List[ControllerOutputChunk] = []
        async for chunk in self.stream(
            inputs=inputs,
            session=session,
            stream_modes=[BaseStreamMode.OUTPUT],
            **kwargs
        ):
            chunks.append(chunk)

        # 确定最终输出类型，默认为 processing
        final_type = "processing"
        
        for chunk in chunks:
            if chunk.payload:
                # 优先级：TASK_INTERACTION/TASK_FAILED > TASK_COMPLETION > processing
                if chunk.payload.type in [EventType.TASK_INTERACTION, EventType.TASK_FAILED]:
                    final_type = chunk.payload.type
                    break
                elif chunk.payload.type == EventType.TASK_COMPLETION:
                    final_type = EventType.TASK_COMPLETION

        return ControllerOutput(
            type=final_type,
            data=chunks,
            input_event_id=inputs.event_id if hasattr(inputs, 'event_id') else None
        )

    async def stream(
            self,
            inputs: 'InputEvent',
            session: Session,
            stream_modes: Optional[List[StreamMode]] = None,
            **kwargs
    ) -> AsyncIterator[ControllerOutputChunk]:
        """流式执行控制器
        
        Args:
            inputs: 输入事件
            session: 会话对象（调用方创建管理）
            stream_modes: 流式输出模式列表（可选）
            **kwargs: 其他参数
            
        Yields:
            ControllerOutputChunk: 控制器输出块

        Note:
            1. 恢复 controller 状态（包括 task_manager 的状态等）
            2. 将 Session 放到 task_scheduler 的 sessions 字典中
            3. 调用 self._event_queue 的 subscribe 方法订阅
            4. 将输入的事件放到 self._event_queue 中
            5. 获取事件处理结果并流式输出
            6. deactivate 所有 subscription
            7. 保存controller 状态（包括 task_manager 的状态等）
            8. 将 Session 从 task_scheduler 的 sessions 字典中移除
        """
        # Lazy start with event loop detection
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError as e:
            logger.error("No running event loop")
            raise build_error(
                StatusCode.CONTROLLER_RUNTIME_ERROR,
                error_msg="No running event loop",
                cause=e
            )
        
        if self._event_loop is not current_loop:
            # Event loop changed or first start
            if self._event_loop is not None:
                logger.info(
                    f"Event loop changed, restarting event_queue and task_scheduler "
                    f"(old: {id(self._event_loop)}, new: {id(current_loop)})"
                )
                try:
                    await self._task_scheduler.stop()
                    await self._event_queue.stop()
                except Exception as e:
                    logger.warning(f"Failed to stop old components: {e}")
                
                # Recreate components (bound to new event loop)
                self._event_queue = EventQueue(config=self._config)
                self._task_scheduler = TaskScheduler(
                    config=self._config,
                    task_manager=self._task_manager,
                    context_engine=self._context_engine,
                    ability_kit=self._ability_kit,
                    event_queue=self._event_queue,
                    card=self._card
                )

            self._event_queue.start()
            self._event_queue.set_event_handler(self._event_handler)
            await self._task_scheduler.start()
            self._event_loop = current_loop
            logger.info(f"Controller started in event loop {id(current_loop)}")

        agent_id = self._card.id
        session_id = session.session_id()

        # 恢复controller状态
        state_restored = self._restore_task_manager_state(session)
        if not state_restored:
            logger.info(f"Starting with clean TaskManager state for session {session_id}")
        
        # Initialize stream_writer_manager based on stream_modes
        self._init_session_stream_manager(session, stream_modes)
        
        # Register session
        self._task_scheduler.sessions[session_id] = session

        # 订阅4种事件
        await self._event_queue.subscribe(agent_id, session_id)

        try:
            # 发布输入事件，触发事件处理与任务调度
            await self._event_queue.publish_event(agent_id, session_id, inputs)

            # 从 session 流式读取并转发，直到所有任务完成
            async for chunk in session.stream_iterator():
                if isinstance(chunk, ControllerOutputChunk):
                    # 检查是否是完成消息
                    if chunk.payload and chunk.payload.type == "all_tasks_processed":
                        logger.info(f"All tasks handled for session {session_id}, stopping stream")
                        break

                yield chunk
        except Exception as e:
            logger.error(f"Controller runtime error: {e}")
            raise build_error(
                StatusCode.CONTROLLER_RUNTIME_ERROR,
                error_msg=e.to_dict().get("message") if isinstance(e, BaseError) else "Failed to execute inputEvent",
                cause=e
            )
        finally:
            # 保存controller状态
            self._save_task_manager_state(session)

            # Unsubscribe events
            await self._event_queue.unsubscribe(agent_id, session_id)

            # Remove session
            if session_id in self._task_scheduler.sessions:
                del self._task_scheduler.sessions[session_id]

            # Keep Controller running which allows multiple stream() calls without restart overhead
            logger.info(
                f"Session {session_id} completed, "
                f"{len(self._task_scheduler.sessions)} active sessions remaining"
            )
