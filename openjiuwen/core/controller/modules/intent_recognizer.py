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
import asyncio
from abc import abstractmethod
from typing import List, TYPE_CHECKING
import json

from openjiuwen.core.context_engine import ContextEngine, ModelContext
from openjiuwen.core.controller.schema.event.event import InputEvent, JsonDataFrame, \
    TaskCompletionEvent, TaskFailedEvent
from openjiuwen.core.controller.schema.intent import IntentType
from openjiuwen.core.controller.schema.dataframe import DataFrame, TextDataFrame, FileDataFrame
from openjiuwen.core.controller.config import ControllerConfig
from openjiuwen.core.controller.modules.event_handler import EventHandler, EventHandlerInput
from openjiuwen.core.controller.modules.intent_toolkits import IntentToolkits
from openjiuwen.core.controller.modules.task_manager import TaskManager, TaskFilter
from openjiuwen.core.controller.schema import Intent
from openjiuwen.core.controller.schema.event import Event, EventType, TaskInteractionEvent
from openjiuwen.core.session import Session
from openjiuwen.core.foundation.llm import UserMessage, SystemMessage, ToolMessage, AssistantMessage
from openjiuwen.core.controller.schema.task import TaskStatus, Task

if TYPE_CHECKING:
    from openjiuwen.core.single_agent.agent import AbilityManager
    from openjiuwen.core.runner import Runner


class IntentRecognizer:
    """意图识别器

    负责识别用户输入中的意图，将事件转换为Intent对象。
    """

    def __init__(
            self,
            config: ControllerConfig,
            task_manager: TaskManager,
            ability_manager: 'AbilityManager',
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

        self._system_message = SystemMessage(content="""# 角色
你是一个任务管理助手，专门使用工具创建和管理任务。你的核心理念是：**任何用户请求都可以转化为一个任务**，并由任务管理器处理。

# 核心原则
1. **任务化一切**：对于任何用户请求（包括信息查询、事务处理、提醒等），你的第一反应不是直接执行或拒绝，而是思考如何将它创建为一个任务。
2. **透明管理**：如果任务需要外部能力（如天气API），你仍然创建它，并明确告知用户任务的状态。

# 工作流程
1. **解析请求**：理解用户想做什么。
2. **任务操作**：使用工具创建一个对应的任务或修改已有任务。
3. **永远不拒绝**：不声称“超出能力范围”，而是告知用户任务会由其他执行器处理。

# 任务目标
- 根据用户输入，**总是优先创建对应的任务**。
- 使用工具进行任务操作（创建、更新、列表、删除）。
- 只有纯粹闲聊或问候时不调用工具。
""")

        self._user_prompt_template = """你当前拥有的任务有：
{task_descriptions}

当前用户的输入为：
{query}

请根据你当前的任务和用户输入，进行合适的任务操作。
"""

    def _prepare_user_message(self, query):
        tasks = self._task_manager.get_task()
        task_prompt = []
        if tasks:
            for task in tasks:
                task_prompt.append(
                    f"## Task id: {task.task_id}\n### Task description: {task.description}\nStatus: {task.status}\n")
        else:
            task_prompt.append("无")
        task_prompt = "\n".join(task_prompt)

        prompt = self._user_prompt_template.format(
            task_descriptions=task_prompt,
            query=query
        )
        return UserMessage(content=prompt)

    async def recognize(self, event: Event, session: Session) -> List[Intent]:
        """识别意图

        Args:
            event: 输入事件
            session: 会话对象

        Returns:
            Intent: 识别出的意图对象
        """
        from openjiuwen.core.runner import Runner

        context: ModelContext = await self._context_engine.create_context(session=session)

        if not isinstance(event, InputEvent):
            raise ValueError

        inputs: List[DataFrame] = event.input_data
        texts = [df for df in inputs if isinstance(df, TextDataFrame)]
        files = [df for df in inputs if isinstance(df, FileDataFrame)]
        jsons = [df for df in inputs if isinstance(df, JsonDataFrame)]

        if files or jsons:
            raise NotImplementedError

        if len(texts) > 1:
            raise NotImplementedError

        model = await Runner.resource_mgr.get_model(id=self._config.intent_llm_id)
        user_message = self._prepare_user_message(query=texts[0].text)
        await context.add_messages(user_message)

        toolkits = IntentToolkits(event, self._config.intent_confidence_threshold)
        # todo make it configurable
        max_message_len = 50
        response = await model.invoke(
            messages=[self._system_message] + context.get_messages(size=max_message_len),
            tools=toolkits.get_openai_tool_schemas()
        )
        await context.add_messages(response)

        intents = []
        while True:
            if not response.tool_calls:
                break
            else:
                for tool_call in response.tool_calls:
                    instance = getattr(toolkits, tool_call.name)
                    intent, result = instance(**json.loads(tool_call.arguments))
                    intents.append(intent)
                    await context.add_messages(ToolMessage(
                        tool_call_id=tool_call.id,
                        content=result
                    ))
                response = await model.invoke(
                    messages=[self._system_message] + context.get_messages(size=max_message_len),
                    tools=toolkits.get_openai_tool_schemas()
                )
                await context.add_messages(response)

        return intents


class EventHandlerWithIntentRecognition(EventHandler):
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
        intents = await self.recognizer.recognize(inputs.event, inputs.session)
        tasks = []
        for intent in intents:
            if intent.intent_type == IntentType.CREATE_TASK:
                tasks.append(asyncio.create_task(self._process_create_task_intent(intent, inputs.session)))
            elif intent.intent_type == IntentType.PAUSE_TASK:
                tasks.append(asyncio.create_task(self._process_pause_task_intent(intent, inputs.session)))
            elif intent.intent_type == IntentType.RESUME_TASK:
                tasks.append(asyncio.create_task(self._process_resume_task_intent(intent, inputs.session)))
            elif intent.intent_type == IntentType.CONTINUE_TASK:
                tasks.append(asyncio.create_task(self._process_continue_task_intent(intent, inputs.session)))
            elif intent.intent_type == IntentType.SUPPLEMENT_TASK:
                tasks.append(asyncio.create_task(self._process_supplement_task_intent(intent, inputs.session)))
            elif intent.intent_type == IntentType.CANCEL_TASK:
                tasks.append(asyncio.create_task(self._process_cancel_task_intent(intent, inputs.session)))
            elif intent.intent_type == IntentType.MODIFY_TASK:
                tasks.append(asyncio.create_task(self._process_modify_task_intent(intent, inputs.session)))
            elif intent.intent_type == IntentType.SWITCH_TASK:
                tasks.append(asyncio.create_task(self._process_switch_task_intent(intent, inputs.session)))
            else:
                tasks.append(asyncio.create_task(self._process_unknown_task_intent(intent, inputs.session)))
        return await asyncio.gather(*tasks)

    async def handle_task_interaction(self, inputs: EventHandlerInput):
        """处理任务交互事件

        将interaction直接抛出给用户，可重写。

        Args:
            inputs: 事件处理器输入
        """
        if not isinstance(inputs.event, TaskInteractionEvent):
            raise ValueError
        await inputs.session.write_stream({
                "interaction": inputs.event.interaction
            })

    async def handle_task_completion(self, inputs: EventHandlerInput):
        """处理任务完成事件

        将任务完成信息抛出给用户，可重写。

        Args:
            inputs: 事件处理器输入
        """
        if not isinstance(inputs.event, TaskCompletionEvent):
            raise ValueError
        await inputs.session.write_stream({
                "result": inputs.event.task_result
            })

    async def handle_task_failed(self, inputs: EventHandlerInput):
        """处理任务失败事件

        将错误信息抛出给用户，可重写。

        Args:
            inputs: 事件处理器输入
        """
        if not isinstance(inputs.event, TaskFailedEvent):
            raise ValueError
        await inputs.session.write_stream({
                "error_message": inputs.event.error_message
            })

    @abstractmethod
    async def _process_create_task_intent(self, intent: Intent, session: Session):
        """处理创建任务意图

        用户自定义执行新任务逻辑。

        Args:
            intent: 意图
            session: Session
        """
        task = Task(
            session_id=session.session_id(),
            task_id=intent.target_task_id,
            task_type="default_task_type",
            description=intent.target_task_description,
            priority=1,
            inputs=[intent.event] if isinstance(intent.event, InputEvent) else None,
            status=TaskStatus.SUBMITTED,
            error_message=None,
            metadata=intent.metadata,
        )
        self.task_manager.add_task(task)

    async def _process_pause_task_intent(self, intent: Intent, session: Session):
        """处理暂停任务意图

        调用 task_scheduler 的 pause_task 方法打断目标任务。

        Args:
            intent: 意图
            session: Session
        """
        await self.task_scheduler.pause_task(intent.task_id)

    async def _process_resume_task_intent(self, intent: Intent, session: Session):
        """处理恢复任务意图

        将要恢复的任务的状态置为 submitted。

        Args:
            intent: 意图
            session: Session
        """
        task = self.task_manager.get_task(TaskFilter(task_id=intent.task_id))
        if task.state == TaskStatus.PAUSED:
            task.state = TaskStatus.SUBMITTED
            self.task_manager.update_task(task)
        else:
            raise ValueError

    async def _process_continue_task_intent(self, intent: Intent, session: Session):
        """处理接续任务意图

        Args:
            intent: 意图
            session: Session
        """
        task = self.task_manager.get_task(TaskFilter(task_id=intent.task_id))

    async def _process_supplement_task_intent(self, intent: Intent, session: Session):
        """处理补充任务意图

        Args:
            intent: 意图
            session: Session
        """
        if intent.intent_type != IntentType.SUPPLEMENT_TASK:
            raise ValueError

        task = self.task_manager.get_task(TaskFilter(task_id=intent.task_id))
        await self.task_scheduler.pause_task(intent.task_id)
        task.description += "\n\n任务补充信息:\n{}".format(json.dumps(intent.supplementary_info))
        task.state = TaskStatus.SUBMITTED
        self.task_manager.update_task(task)

    async def _process_cancel_task_intent(self, intent: Intent, session: Session):
        """处理取消任务意图

        调用 task_scheduler 的 cancel_task 方法取消目标任务。

        Args:
            intent: 意图
            session: Session
        """
        if intent.intent_type != IntentType.CANCEL_TASK:
            raise ValueError

        await self.task_scheduler.cancel_task(intent.task_id)

    async def _process_modify_task_intent(self, intent: Intent, session: Session):
        """处理修改任务意图

        修改目标任务后，将其状态置为 submitted。

        Args:
            intent: 意图
            session: Session
        """
        if intent.intent_type != IntentType.MODIFY_TASK:
            raise ValueError

        await self.task_scheduler.cancel_task(intent.task_id)
        task = self.task_manager.get_task(intent.task_id)
        # todo 修改逻辑待确认

        task.status = TaskStatus.SUBMITTED
        self.task_manager.update_task(task)

    async def _process_switch_task_intent(self, intent: Intent, session: Session):
        """处理切换任务意图

        打断所有正在执行的任务，再调用 _process_create_task_intent 执行目标任务。

        Args:
            intent: 意图
            session: Session
        """
        if intent.intent_type != IntentType.PAUSE_TASK:
            raise ValueError

        tasks = self.task_manager.get_task()
        for task in tasks:
            await self.task_scheduler.pause_task(task.task_id)

    async def _process_unknown_task_intent(self, intent: Intent, session: Session):
        """处理未知任务意图

        返回 Intent 的 clarification_prompt 字段给用户。

        Args:
            intent: 意图
            session: Session
        """
        if intent.intent_type != IntentType.UNKNOWN_TASK:
            raise ValueError
        await session.write_stream({
                "clarification_prompt": intent.clarification_prompt
            })
