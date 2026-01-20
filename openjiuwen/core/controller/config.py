"""控制器配置模块

该模块定义了控制器配置相关的类：
- ControllerConfig: 控制器配置类
"""
from typing import Optional
from pydantic import BaseModel, Field


class ControllerConfig(BaseModel):
    """控制器配置

    定义控制器的配置参数，用于控制控制器的行为。
    配置项分为几个类别：任务调度、任务管理、事件队列和意图识别。

    Attributes:
        # ==================== 任务调度配置 ====================
        max_concurrent_tasks: 最大并发任务数，控制同时执行的任务数量上限。
                             默认为5，设置为0表示不限制。
        schedule_interval: 任务调度间隔（秒），调度器定期扫描待执行任务的间隔时间。
                           默认为1.0秒，较小的值可以提高响应速度但会增加CPU使用。
        task_timeout: 任务超时时间（秒），超过此时间的任务将被标记为失败。
                     默认为None，表示不设置超时。

        # ==================== 任务管理配置 ====================
        default_task_priority: 默认任务优先级，创建任务时如果未指定优先级则使用此值。
                               默认为1，数字越大优先级越高。
        enable_task_persistence: 是否启用任务持久化，启用后任务状态会被保存以便恢复。
                                 默认为False。

        # ==================== 事件队列配置 ====================
        event_queue_size: 事件队列大小，限制队列中可存储的事件数量。
                         默认为None，表示不限制队列大小。
        event_timeout: 事件处理超时时间（秒），超过此时间未处理的事件将被丢弃。
                      默认为None，表示不设置超时。

        # ==================== 意图识别配置 ====================
        enable_intent_recognition: 是否启用意图识别功能。
                                   默认为True，启用后会自动识别用户意图并路由到相应处理。
        intent_confidence_threshold: 意图识别置信度阈值，低于此值的意图将被视为UNKNOWN_TASK。
                                    默认为0.7，范围0.0-1.0。

    Example:
        ```python
        config = ControllerConfig(
            max_concurrent_tasks=10,
            schedule_interval=0.5,
            default_task_priority=5,
            enable_intent_recognition=True
        )
        ```
    """
    # ==================== 任务调度配置 ====================
    max_concurrent_tasks: int = Field(
        default=5,
        description="最大并发任务数，控制同时执行的任务数量上限。设置为0表示不限制。"
    )
    schedule_interval: float = Field(
        default=1.0,
        ge=0.1,
        description="任务调度间隔（秒），调度器定期扫描待执行任务的间隔时间。"
    )
    task_timeout: Optional[float] = Field(
        default=None,
        ge=600,
        description="任务超时时间（秒），超过此时间的任务将被标记为失败。None表示不设置超时。"
    )

    # ==================== 任务管理配置 ====================
    default_task_priority: int = Field(
        default=1,
        description="默认任务优先级，创建任务时如果未指定优先级则使用此值。数字越大优先级越高。"
    )

    # ==================== 事件队列配置 ====================
    event_queue_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="事件队列大小，限制队列中可存储的事件数量。None表示不限制队列大小。"
    )
    event_timeout: Optional[float] = Field(
        default=None,
        ge=600,
        description="事件处理超时时间（秒），超过此时间未处理的事件将被丢弃。None表示不设置超时。"
    )

    # ==================== 意图识别配置 ====================
    intent_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="意图识别置信度阈值，低于此值的意图将被视为UNKNOWN_TASK。范围0.0-1.0。"
    )

