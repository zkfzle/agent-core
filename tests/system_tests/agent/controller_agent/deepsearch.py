"""Arxiv研究报告智能体示例

本模块展示如何实现一个基于控制器的Arxiv研究报告生成智能体。
该智能体通过三个阶段完成研究报告的生成：
1. 数据收集（Data Collection）：从Arxiv接口收集相关的论文数据
2. 数据分析（Data Analysis）：对收集的数据进行分析
3. 报告生成（Report Generation）：基于分析结果生成研究报告和图标

工作流程：
1. 用户输入研究需求，触发输入事件
2. 事件处理器进行任务规划，创建三个阶段的任务
3. 任务调度器按优先级顺序执行任务
4. 每个阶段完成后，自动触发下一阶段的执行

主要组件：
- 任务执行器：DataCollectTaskExecutor, DataAnalysisTaskExecutor, ReportGenerateTaskExecutor
- 事件处理器：DeepSearchEventHandler
- Agent构建：build_deepsearch_agent
"""
from typing import Dict, List, AsyncIterator, Tuple

from openjiuwen.core.context_engine import ContextEngine
from openjiuwen.core.controller.base import ControllerConfig, Controller
from openjiuwen.core.controller.modules import EventHandler, EventHandlerInput
from openjiuwen.core.controller.modules import TaskManager
from openjiuwen.core.controller.modules import TaskExecutor
from openjiuwen.core.controller.modules import EventQueue
from openjiuwen.core.controller.modules.task_manager import TaskFilter
from openjiuwen.core.controller.schema import ControllerOutputChunk, ControllerOutputPayload, \
    EventType
from openjiuwen.core.controller.schema import Event
from openjiuwen.core.controller.schema.task import Task, TaskStatus
from openjiuwen.core.controller.schema import TextDataFrame
# from openjiuwen.core.runner import Runner
from openjiuwen.core.session import Session, TaskSession
from openjiuwen.core.session.stream import BaseStreamMode
from openjiuwen.core.single_agent import AgentCard
from openjiuwen.core.single_agent.agent import AbilityKit, ControllerAgent
from openjiuwen.core.common.logging import logger


class DataCollectTaskExecutor(TaskExecutor):
    """数据收集任务执行器

    负责执行数据收集任务，从各种数据源收集所需的金融数据。

    主要职责：
    - 从各种渠道收集数据
    - 将所有收集的数据保存到上下文引擎中

    Args:
        config: 控制器配置，包含任务调度相关参数
        ability_kit: 能力包，提供数据收集所需的工具和能力
        context_engine: 上下文引擎，用于存储和管理任务上下文数据
        task_manager: 任务管理器，用于管理任务状态
        event_queue: 事件队列，用于发布任务完成等事件
    """

    def __init__(
            self,
            config: ControllerConfig,
            ability_kit: AbilityKit,
            context_engine: ContextEngine,
            task_manager: TaskManager,
            event_queue: EventQueue
    ):
        super().__init__(
            config=config,
            ability_kit=ability_kit,
            context_engine=context_engine,
            task_manager=task_manager,
            event_queue=event_queue
        )

    async def execute_ability(self, task_id: str, session: Session) -> AsyncIterator[ControllerOutputChunk]:
        """执行数据收集任务

        从各种数据源收集金融数据，并将收集结果流式输出。
        收集到的数据会被存储在上下文引擎中，供后续的数据分析任务使用。

        Args:
            task_id: 任务ID，用于标识当前执行的任务
            session: 会话对象，包含会话上下文信息

        Yields:
            ControllerOutputChunk: 数据收集过程的输出块，可能包括：
                - 数据收集进度信息
                - 收集到的数据摘要
                - 收集过程中的状态更新

        Note:
            - 数据收集流程的上下文放在 context engine 中 Task.context_id 对应的 context 中
            - 执行完成后会自动触发任务完成事件，进而启动数据分析任务
            - 如果数据收集失败，会触发任务失败事件
        """
        # 简单执行：返回处理中信息
        yield ControllerOutputChunk(
            index=0,
            type="controller_output",
            payload=ControllerOutputPayload(
                type="processing",
                data=[TextDataFrame(type="text", text="正在收集芯片相关的Arxiv论文数据...")]
            ),
            last_chunk=False
        )

        # 模拟收集到一些数据
        collected_data = {"topic": "芯片", "count": 10, "source": "arxiv"}

        # 任务完成信号
        yield ControllerOutputChunk(
            index=1,
            type="controller_output",
            payload=ControllerOutputPayload(
                type=EventType.TASK_COMPLETION,
                data=[TextDataFrame(type="text", text="芯片相关Arxiv论文数据收集完成啦")]
            ),
            last_chunk=True
        )

        await session.write_stream({
            "type": "result",
            "index": 0,
            "payload": {"result": "芯片相关Arxiv论文数据收集完成"}
        })

    async def can_pause(self, task_id: str, session: Session) -> Tuple[bool, str]:
        """检查任务是否可暂停

        Args:
            task_id: 任务ID
            session: 会话对象

        Returns:
            Tuple[bool, str]: (是否可暂停, 原因说明)

        """
        # 不涉及，可不实现
        ...

    async def pause(self, task_id: str, session: Session) -> bool:
        """暂停任务执行

        Args:
            task_id: 任务ID
            session: 会话对象

        Returns:
            bool: 是否成功暂停
        """
        # 不涉及，可不实现
        ...

    async def can_cancel(self, task_id: str, session: Session) -> bool:
        """检查任务是否可取消

        Args:
            task_id: 任务ID
            session: 会话对象

        Returns:
            bool: 是否可取消
        """
        # 不涉及，可不实现
        ...

    async def cancel(self, task_id: str, session: Session) -> bool:
        """取消任务执行

        Args:
            task_id: 任务ID
            session: 会话对象

        Returns:
            bool: 是否成功取消
        """
        ...


class DataAnalysisTaskExecutor(TaskExecutor):
    """数据分析任务执行器

    负责执行数据分析任务，对收集到的金融数据进行深入分析。

    主要职责：
    - 读取数据收集任务收集的数据
    - 进行财务指标计算和分析
    - 进行趋势分析和预测
    - 识别关键洞察和模式
    - 将分析结果保存到上下文引擎中

    Args:
        config: 控制器配置，包含任务调度相关参数
        ability_kit: 能力包，提供数据分析所需的工具和能力（如AI分析模型）
        context_engine: 上下文引擎，用于读取收集的数据和存储分析结果
        task_manager: 任务管理器，用于管理任务状态
        event_queue: 事件队列，用于发布任务完成等事件
    """

    def __init__(
            self,
            config: ControllerConfig,
            ability_kit: AbilityKit,
            context_engine: ContextEngine,
            task_manager: TaskManager,
            event_queue: EventQueue
    ):
        super().__init__(
            config=config,
            ability_kit=ability_kit,
            context_engine=context_engine,
            task_manager=task_manager,
            event_queue=event_queue
        )

    async def execute_ability(self, task_id: str, session: Session) -> AsyncIterator[ControllerOutputChunk]:
        """执行数据分析任务

        对收集到的金融数据进行深入分析，并将分析结果流式输出。
        分析结果会被存储在上下文引擎中，供后续的报告生成任务使用。

        Args:
            task_id: 任务ID，用于标识当前执行的任务
            session: 会话对象，包含会话上下文信息

        Yields:
            ControllerOutputChunk: 数据分析过程的输出块，可能包括：
                - 分析进度信息
                - 分析结果摘要
                - 关键发现和洞察
                - 分析过程中的状态更新

        Note:
            - 数据分析流程的上下文放在 context engine 中 Task.context_id 对应的 context 中
            - 需要通过 ref_task_id 从数据收集任务获取收集的数据
            - 执行完成后会自动触发任务完成事件，进而启动报告生成任务
            - 如果数据分析失败，会触发任务失败事件
        """

        # 简单执行：返回处理中信息
        yield ControllerOutputChunk(
            index=0,
            type="controller_output",
            payload=ControllerOutputPayload(
                type="processing",
                data=[TextDataFrame(type="text", text="正在分析芯片相关的Arxiv论文数据...")]
            ),
            last_chunk=False
        )

        # 获取收集到的数据
        task = self._task_manager.get_task(task_filter=TaskFilter(task_id=task_id))

        # 简单分析：生成一个分析结果
        analysis_result = {
            "topic": "芯片",
            "trend": "近年来芯片领域研究热度持续上升",
            "key_areas": ["AI芯片", "量子芯片", "先进制程"]
        }

        # 返回任务完成信息
        yield ControllerOutputChunk(
            index=1,
            type="controller_output",
            payload=ControllerOutputPayload(
                type=EventType.TASK_COMPLETION,
                data=[TextDataFrame(type="text", text="芯片相关Arxiv论文数据分析完成")]
            ),
            last_chunk=True
        )

    async def can_pause(self, task_id: str, session: Session) -> Tuple[bool, str]:
        """检查任务是否可暂停

        Args:
            task_id: 任务ID
            session: 会话对象

        Returns:
            Tuple[bool, str]: (是否可暂停, 原因说明)
        """
        # 不涉及，可不实现
        ...

    async def pause(self, task_id: str, session: Session) -> bool:
        """暂停任务执行

        Args:
            task_id: 任务ID
            session: 会话对象

        Returns:
            bool: 是否成功暂停
        """
        # 不涉及，可不实现
        ...

    async def can_cancel(self, task_id: str, session: Session) -> bool:
        """检查任务是否可取消

        Args:
            task_id: 任务ID
            session: 会话对象

        Returns:
            bool: 是否可取消
        """
        # 不涉及，可不实现
        ...

    async def cancel(self, task_id: str, session: Session) -> bool:
        """取消任务执行

        Args:
            task_id: 任务ID
            session: 会话对象

        Returns:
            bool: 是否成功取消
        """
        # 不涉及，可不实现
        ...


class ReportGenerateTaskExecutor(TaskExecutor):
    """报告生成任务执行器

    负责执行报告生成任务，基于数据分析结果生成最终的研究报告。

    主要职责：
    - 读取数据分析任务的分析结果
    - 组织报告结构和内容
    - 生成报告文本（可能使用AI生成模型）
    - 格式化报告（添加图表、表格等）
    - 将最终报告保存到上下文引擎中

    Args:
        config: 控制器配置，包含任务调度相关参数
        ability_kit: 能力包，提供报告生成所需的工具和能力（如文本生成模型）
        context_engine: 上下文引擎，用于读取分析结果和存储生成的报告
        task_manager: 任务管理器，用于管理任务状态
        event_queue: 事件队列，用于发布任务完成等事件
    """

    def __init__(
            self,
            config: ControllerConfig,
            ability_kit: AbilityKit,
            context_engine: ContextEngine,
            task_manager: TaskManager,
            event_queue: EventQueue
    ):
        super().__init__(
            config=config,
            ability_kit=ability_kit,
            context_engine=context_engine,
            task_manager=task_manager,
            event_queue=event_queue
        )

    async def execute_ability(self, task_id: str, session: Session) -> AsyncIterator[ControllerOutputChunk]:
        """执行报告生成任务

        基于数据分析结果生成最终的研究报告，并将报告内容流式输出。
        生成的报告会被存储在上下文引擎中，用户可以通过会话获取完整报告。

        Args:
            task_id: 任务ID，用于标识当前执行的任务
            session: 会话对象，包含会话上下文信息

        Yields:
            ControllerOutputChunk: 报告生成过程的输出块，可能包括：
                - 报告生成进度信息
                - 报告各个部分的流式输出（摘要、分析、结论等）
                - 报告生成完成的状态更新

        Note:
            - 报告生成流程的上下文放在 context engine 中 Task.context_id 对应的 context 中
            - 需要从上下文引擎中读取数据分析任务的结果
            - 执行完成后会自动触发任务完成事件，整个研究报告生成流程结束
            - 如果报告生成失败，会触发任务失败事件
        """

        # 简单执行：返回处理中信息
        yield ControllerOutputChunk(
            index=0,
            type="controller_output",
            payload=ControllerOutputPayload(
                type="processing",
                data=[TextDataFrame(type="text", text="正在生成芯片研究报告...")]
            ),
            last_chunk=False
        )

        await session.write_stream({
            "type": "result",
            "index": 0,
            "payload": {"result": "芯片领域研究报告已生成"}
        })

        # 返回任务完成信息
        yield ControllerOutputChunk(
            index=0,
            type="controller_output",
            payload=ControllerOutputPayload(
                type=EventType.TASK_COMPLETION,
                data=[TextDataFrame(type="text", text="芯片研究报告生成完成")]
            ),
            last_chunk=True
        )

    async def can_pause(self, task_id: str, session: Session) -> Tuple[bool, str]:
        """检查任务是否可暂停

        Args:
            task_id: 任务ID
            session: 会话对象

        Returns:
            Tuple[bool, str]: (是否可暂停, 原因说明)
        """
        # 不涉及，可不实现
        ...

    async def pause(self, task_id: str, session: Session) -> bool:
        """暂停任务执行

        Args:
            task_id: 任务ID
            session: 会话对象

        Returns:
            bool: 是否成功暂停
        """
        # 不涉及，可不实现
        ...

    async def can_cancel(self, task_id: str, session: Session) -> bool:
        """检查任务是否可取消

        Args:
            task_id: 任务ID
            session: 会话对象

        Returns:
            bool: 是否可取消
        """
        # 不涉及，可不实现
        ...

    async def cancel(self, task_id: str, session: Session) -> bool:
        """取消任务执行

        Args:
            task_id: 任务ID
            session: 会话对象

        Returns:
            bool: 是否成功取消
        """
        # 不涉及，可不实现
        ...


def build_data_collect_task_executor(
        config: ControllerConfig,
        ability_kit: AbilityKit,
        context_engine: ContextEngine,
        task_manager: TaskManager,
        event_queue: EventQueue
) -> DataCollectTaskExecutor:
    """构建数据收集任务执行器

    工厂函数，用于创建数据分析任务执行器实例。
    任务执行器注册到控制器后，当数据分析任务需要执行时，
    控制器会调用此函数创建执行器实例。

    Args:
        config: 控制器配置
        ability_kit: 能力包
        context_engine: 上下文引擎
        task_manager: 任务管理器
        event_queue: 事件队列

    Returns:
        DataCollectTaskExecutor: 数据收集任务执行器实例

    Note:
    此函数会被注册到控制器的任务执行器注册表中，
    当遇到类型为 "data_collect" 的任务时会被调用
    """
    return DataCollectTaskExecutor(
        config=config,
        ability_kit=ability_kit,
        context_engine=context_engine,
        task_manager=task_manager,
        event_queue=event_queue
    )


def build_data_analysis_task_executor(
        config: ControllerConfig,
        ability_kit: AbilityKit,
        context_engine: ContextEngine,
        task_manager: TaskManager,
        event_queue: EventQueue
) -> DataAnalysisTaskExecutor:
    """构建数据分析任务执行器

    工厂函数，用于创建数据分析任务执行器实例。
    任务执行器注册到控制器后，当数据分析任务需要执行时，
    控制器会调用此函数创建执行器实例。

    Args:
        config: 控制器配置，包含任务调度、超时等配置参数
        ability_kit: 能力包，提供数据分析所需的工具和能力
        context_engine: 上下文引擎，用于数据存储和检索
        task_manager: 任务管理器，用于管理任务生命周期
        event_queue: 事件队列，用于发布和订阅事件

    Returns:
        DataAnalysisTaskExecutor: 数据分析任务执行器实例

    Note:
        此函数会被注册到控制器的任务执行器注册表中，
        当遇到类型为 "data_analysis" 的任务时会被调用
    """
    return DataAnalysisTaskExecutor(
        config=config,
        ability_kit=ability_kit,
        context_engine=context_engine,
        task_manager=task_manager,
        event_queue=event_queue
    )


def build_report_generate_task_executor(
        config: ControllerConfig,
        ability_kit: AbilityKit,
        context_engine: ContextEngine,
        task_manager: TaskManager,
        event_queue: EventQueue
) -> ReportGenerateTaskExecutor:
    """构建报告生成任务执行器

    工厂函数，用于创建报告生成任务执行器实例。
    任务执行器注册到控制器后，当报告生成任务需要执行时，
    控制器会调用此函数创建执行器实例。

    Args:
        config: 控制器配置，包含任务调度、超时等配置参数
        ability_kit: 能力包，提供报告生成所需的工具和能力
        context_engine: 上下文引擎，用于数据存储和检索
        task_manager: 任务管理器，用于管理任务生命周期
        event_queue: 事件队列，用于发布和订阅事件

    Returns:
        ReportGenerateTaskExecutor: 报告生成任务执行器实例

    Note:
        此函数会被注册到控制器的任务执行器注册表中，
        当遇到类型为 "report_generate" 的任务时会被调用
    """
    return ReportGenerateTaskExecutor(
        config=config,
        ability_kit=ability_kit,
        context_engine=context_engine,
        task_manager=task_manager,
        event_queue=event_queue
    )


class DeepSearchEventHandler(EventHandler):
    """Arxiv论文搜索事件处理器

    深度搜索智能体的事件处理器，负责处理各种类型的事件并协调任务执行流程。

    主要职责：
    1. 处理输入事件：接收用户请求，进行任务规划，创建三个阶段的任务
    2. 处理任务完成事件：监控任务执行状态，自动触发下一阶段任务的执行
    3. 处理任务失败事件：处理任务执行失败的情况

    工作流程：
    1. 用户输入研究需求 -> handle_input 被调用
    2. 进行任务规划，创建数据收集、数据分析、报告生成三类任务
    3. 数据收集任务完成后 -> handle_task_completion 被调用
    4. 检查数据收集是否全部完成，如果是则将数据分析任务状态改为 submitted
    5. 数据分析任务完成后 -> handle_task_completion 被调用
    6. 检查数据分析是否全部完成，如果是则将报告生成任务状态改为 submitted
    7. 报告生成任务完成后 -> handle_task_completion 被调用，流程结束
    """

    def __init__(self):
        """初始化事件处理器"""
        super().__init__()

    async def _planning(self, event: Event, session: Session) -> Dict:
        """规划任务

        根据用户输入的 research 需求，进行任务规划，确定需要执行的具体任务。
        规划结果包含数据收集、数据分析、报告生成三个阶段的具体任务信息。

        Args:
            event: 输入事件，包含用户的研究需求
            session: 会话对象，包含会话上下文和历史信息

        Returns:
            Dict: 规划结果，包含以下信息：
                - 数据收集任务列表：需要收集哪些数据（股价、财报、新闻等）
                - 数据分析任务列表：需要分析哪些方面（财务指标、趋势、原因等）
                - 报告生成任务信息：报告的结构和格式要求

        Note:
            - 可以使用 ability_kit 中的规划能力（如 LLM）进行任务规划
            - 规划结果会被用于创建具体的任务实例
            - 规划过程可以使用 context_engine 获取历史上下文信息
        """
        # 简单规划：创建一个数据收集任务、一个数据分析任务和一个报告生成任务
        return {
            "data_collect_tasks": [{"topic": "芯片", "type": "arxiv"}],
            "data_analysis_tasks": [{"type": "trend_analysis"}],
            "report_generate_tasks": [{"format": "markdown", "type": "research_report"}]
        }

    def _create_data_collect_task(self, planning_task: Dict, session: Session) -> List[Task]:
        """创建数据收集任务

        根据规划结果创建数据收集任务列表。这些任务会立即被提交执行。

        Args:
            planning_task: 规划任务结果，包含数据收集任务的具体信息
            session: 会话对象，包含会话上下文和历史信息

        Returns:
            List[Task]: 数据收集任务列表，每个任务代表一个数据收集操作

        Note:
            - 任务优先级应设置为 1（最高优先级）
            - 任务状态应设置为 submitted（已提交），表示可以立即执行
            - 每个任务需要设置唯一的 task_id 和相关的 context_id
            - 任务会被添加到 task_manager 中进行管理
        """

        tasks = []
        for i, task_info in enumerate(planning_task["data_collect_tasks"]):
            task = Task(
                session_id=session.session_id(),
                task_id="task_DC_id{}".format(i),
                task_type="data_collect",
                priority=1,
                status=TaskStatus.SUBMITTED,
                context_id="context_DC_id{}".format(i),
                params={"topic": task_info["topic"], "type": task_info["type"]}
            )
            tasks.append(task)
        return tasks

    def _create_data_analysis_task(self, planning_task: Dict, session: Session) -> List[Task]:
        """创建数据分析任务

        根据规划结果创建数据分析任务列表。这些任务会等待数据收集任务完成后再执行。

        Args:
            planning_task: 规划任务结果，包含数据分析任务的具体信息

        Returns:
            List[Task]: 数据分析任务列表，每个任务代表一个数据分析操作

        Note:
            - 任务优先级应设置为 2（中等优先级）
            - 任务状态应设置为 waiting（等待），表示需要等待前置任务完成
            - 每个任务的 ref_task_id 应指向对应的数据收集任务ID
            - 任务会被添加到 task_manager 中进行管理，等待被激活执行
        """
        tasks = []
        for i, task_info in enumerate(planning_task["data_analysis_tasks"]):
            task = Task(
                session_id=session.session_id(),
                task_id="task_DA_id{}".format(i),
                task_type="data_analysis",
                priority=2,
                status=TaskStatus.WAITING,
                context_id="context_DA_id{}".format(i),
                params={"type": task_info["type"]}
            )
            tasks.append(task)
        return tasks

    def _create_report_generate_task(self, planning_task: Dict, session: Session) -> List[Task]:
        """创建报告生成任务

        根据规划结果创建报告生成任务列表。这些任务会等待数据分析任务完成后再执行。

        Args:
            planning_task: 规划任务结果，包含报告生成任务的具体信息

        Returns:
            List[Task]: 报告生成任务列表，每个任务代表一个报告生成操作

        Note:
            - 任务优先级应设置为 3（最低优先级）
            - 任务状态应设置为 waiting（等待），表示需要等待前置任务完成
            - 任务会被添加到 task_manager 中进行管理，等待被激活执行
            - 通常一个研究报告只需要一个报告生成任务
        """
        tasks = []
        for i, task_info in enumerate(planning_task["report_generate_tasks"]):
            task = Task(
                session_id=session.session_id(),
                task_id="task_RG_id{}".format(i),
                task_type="report_generate",
                priority=3,
                status=TaskStatus.WAITING,
                context_id="context_RG_id{}".format(i),
                params={"format": task_info["format"], "type": task_info["type"]}
            )
            tasks.append(task)
        return tasks

    async def handle_input(self, inputs: EventHandlerInput):
        """处理输入事件

        当用户输入新的研究需求时，此方法会被调用。
        它会进行任务规划，创建三个阶段的任务，并将任务添加到任务管理器中。

        Args:
            inputs: 事件处理器输入，包含事件和会话信息

        Process:
            1. 调用 _planning 方法进行任务规划
            2. 创建数据收集任务列表（优先级1，状态submitted）
            3. 创建数据分析任务列表（优先级2，状态waiting）
            4. 创建报告生成任务列表（优先级3，状态waiting）
            5. 将所有任务添加到 task_manager 中

        Note:
            - 数据收集任务会立即开始执行
            - 数据分析和报告生成任务会等待前置任务完成
        """
        logger.info("handle input called")
        planning_result = await self._planning(inputs.event, inputs.session)
        tasks = []
        tasks.extend(self._create_data_collect_task(planning_result, inputs.session))
        tasks.extend(self._create_data_analysis_task(planning_result, inputs.session))
        tasks.extend(self._create_report_generate_task(planning_result, inputs.session))
        self.task_manager.add_task(tasks)
        logger.info("handle input end, successfully add tasks to task manager")
        await inputs.session.write_stream({
            "type": "result",
            "index": 0,
            "payload": {"result": "成功调用hanle_input回调"}
        })
        return {"status": "success", "tasks_created": 1}  # 返回确认信息

    async def handle_task_interaction(self, inputs: EventHandlerInput):
        """处理任务交互事件

        当任务需要与用户交互时（如需要用户确认、输入等），此方法会被调用。

        Args:
            inputs: 事件处理器输入，包含事件和会话信息

        Note:
            Finsight 智能体不需要与用户进行交互，因此此方法不需要实现。
            如果确实需要实现，可以在此处处理用户交互逻辑。
        """
        # 不存在该情况，不需要实现
        ...

    async def handle_task_completion(self, inputs: EventHandlerInput):
        """处理任务完成事件

        当某个任务执行完成时，此方法会被调用。
        它会检查当前阶段的所有任务是否都已完成，如果是，则激活下一阶段的任务。

        Args:
            inputs: 事件处理器输入，包含已完成的任务信息和会话信息

        Process:
            阶段1（数据收集任务完成）：
                1. 检查所有优先级为 1 的数据收集任务是否都已完成
                2. 如果有未完成的任务，则退出（等待其他任务完成）
                3. 如果全部完成，检查优先级为 2 的数据分析任务是否还在等待
                4. 如果还在等待，将所有优先级为 2 的任务状态改为 submitted，触发执行

            阶段2（数据分析任务完成）：
                1. 检查所有优先级为 2 的数据分析任务是否都已完成
                2. 如果有未完成的任务，则退出（等待其他任务完成）
                3. 如果全部完成，检查优先级为 3 的报告生成任务是否还在等待
                4. 如果还在等待，将所有优先级为 3 的任务状态改为 submitted，触发执行

            阶段3（报告生成任务完成）：
                - 所有任务都已完成，整个流程结束

        Note:
            - 每个阶段的任务必须全部完成后，才能启动下一阶段
            - 通过修改任务状态为 submitted 来触发任务执行
            - 任务调度器会自动检测状态变化并执行任务
        """
        logger.info("handle task completion called")
        # 获取所有任务
        all_tasks = self.task_manager.get_task(task_filter=None)

        await inputs.session.write_stream({
            "type": "result",
            "index": 0,
            "payload": {"result": f"成功调用handle_task_completion回调 event: {inputs.event.event_id}"}
        })
        # 按优先级顺序处理任务完成逻辑
        # 获取所有任务的优先级，并按升序排序
        priorities = sorted(set(task.priority for task in all_tasks))

        for i, current_priority in enumerate(priorities):
            # 检查当前优先级的所有任务是否都已完成
            current_priority_tasks = [task for task in all_tasks if task.priority == current_priority]
            all_current_completed = all(task.status == TaskStatus.COMPLETED for task in current_priority_tasks)

            if all_current_completed:
                # 检查是否有下一个优先级
                if i + 1 < len(priorities):
                    next_priority = priorities[i + 1]
                    # 检查下一个优先级的任务是否在等待状态，如果是则激活
                    next_priority_waiting_tasks = [task for task in all_tasks if
                                                   task.priority == next_priority and task.status == TaskStatus.WAITING]

                    if next_priority_waiting_tasks:
                        for task in next_priority_waiting_tasks:
                            task.status = TaskStatus.SUBMITTED
                        return {"status": "success", "tasks_created": 1}

        return {"status": "success", "tasks_created": 1}

    async def handle_task_failed(self, inputs: EventHandlerInput):
        """处理任务失败事件

        当某个任务执行失败时，此方法会被调用。
        任务失败会导致整个研究报告生成流程终止。

        Args:
            inputs: 事件处理器输入，包含失败的任务信息和错误信息

        Raises:
            Exception: 抛出异常，表示任务执行失败，整个流程终止

        Note:
            - 可以选择记录失败原因到日志或上下文引擎
            - 可以选择清理已创建的其他任务
            - 可以选择通知用户任务失败的原因
        """
        # 抛异常报错
        logger.info("handle task failed called")
        ...


async def build_deepsearch_agent(agent_card: AgentCard) -> ControllerAgent:
    """构建DeepSearch Agent

    工厂函数，用于创建和配置完整的 DeepSearch 芯片相关研究论文智能体。
    该函数会创建控制器、注册任务执行器、设置事件处理器，最终返回配置好的智能体。

    Args:
        agent_card: Agent名片，包含智能体的基本信息（id、name、description等）

    Returns:
        ControllerAgent: 配置完成的 DeepSearch Agent 实例，可以直接使用

    Process:
        1. 创建 Controller 实例
        2. 创建并设置 DeepSearchEventHandler 事件处理器
        3. 注册三个任务执行器：
           - data_collect: 数据收集任务执行器
           - data_analysis: 数据分析任务执行器
           - report_generate: 报告生成任务执行器
        4. 创建 ControllerAgent 实例，关联控制器和名片
        5. 返回配置完成的智能体

    Example:
        ```python
        agent_card = AgentCard(
            id="deepsearch",
            name="DeepSearch",
            description="Arxiv论文研究报告智能体"
        )
        agent = await build_deepsearch_agent(agent_card)
        ```
    """
    deepsearch_controller = Controller()

    deepsearch_agent = ControllerAgent(
        card=agent_card,
        controller=deepsearch_controller
    )

    deepsearch_controller.set_event_handler(DeepSearchEventHandler())
    deepsearch_controller.add_task_executor(
        "data_collect", build_data_collect_task_executor
    ).add_task_executor(
        "data_analysis", build_data_analysis_task_executor
    ).add_task_executor(
        "report_generate", build_report_generate_task_executor
    )
    return deepsearch_agent

async def main():
    """异步入口：驱动 DeepSearch 智能体"""
    # 1. 创建并注册
    deepsearch_agent_card = AgentCard(
        id="deepsearch",
        name="DeepSearch",
        description="Arxiv研究报告智能体，可以通过收集、分析数据生成Arxiv研究报告",
    )
    agent_factory = build_deepsearch_agent          # 只保留工厂引用
    # Runner().resource_mgr.add_agent(deepsearch_agent_card, agent_factory)

    # 2. 实例化并运行
    deepsearch_agent = await agent_factory(deepsearch_agent_card)
    deepsearch_agent.configure(ControllerConfig())
    session = TaskSession(trace_id="test_deepsearch")
    async for chunk in deepsearch_agent.stream("帮我查找芯片相关研究论文", session):
        print(chunk, end="\n", flush=True)            # 或 yield / 收集 / 前端推送

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
