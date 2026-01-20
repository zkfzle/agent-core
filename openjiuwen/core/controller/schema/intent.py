"""意图模式定义

包含的主要类：
- IntentType: 意图类型枚举
- Intent: 意图数据模型

意图模式定义了用户意图的结构和类型，用于控制器识别和处理用户请求。
"""
from enum import Enum
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field

from openjiuwen.core.controller.schema.event import Event


class IntentType(Enum):
    """意图类型枚举

    定义用户可能表达的所有意图类型，用于控制器识别用户请求并路由到相应的处理逻辑。

    意图类型说明：
    - CREATE_TASK: 创建新任务。执行新任务，如果当前有正在执行的任务，会先打断它们。
    - PAUSE_TASK: 暂停任务。暂停当前正在执行的任务，任务状态变为paused，可以后续恢复。
    - RESUME_TASK: 恢复任务。恢复之前被暂停的任务，将任务状态从paused改为submitted。
    - CONTINUE_TASK: 接续任务。在已完成任务的基础上继续执行新任务，新任务会依赖已完成任务的上下文。
    - SUPPLEMENT_TASK: 补充任务信息。为需要用户输入的任务补充必要信息，然后继续执行。
    - CANCEL_TASK: 取消任务。取消当前正在执行的任务，任务状态变为cancelled。
    - MODIFY_TASK: 修改任务。修改正在执行的任务的参数或配置，修改后任务状态变为submitted重新执行。
    - SWITCH_TASK: 切换任务。中断当前所有正在执行的任务，然后执行新任务。
    - UNKNOWN_TASK: 未知意图。无法识别的用户意图，需要向用户请求澄清。
    """
    CREATE_TASK = "create_task"  # 执行新任务/打断正在执行的任务，并执行新任务
    PAUSE_TASK = "pause_task"  # 暂停正在执行的任务
    RESUME_TASK = "resume_task"  # 恢复任务（恢复之前暂停的任务）
    CONTINUE_TASK = "continue_task"  # 接续任务（在已完成任务的基础上继续执行任务）
    SUPPLEMENT_TASK = "supplement_task"  # 为任务补充必要信息
    CANCEL_TASK = "cancel_task"  # 取消当前执行的任务
    MODIFY_TASK = "modify_task"  # 修改正在执行的任务
    SWITCH_TASK = "switch_task"  # 切换任务（中断当前任务执行另一个任务）
    UNKNOWN_TASK = "unknown_task"  # 未知意图，需要用户澄清


class Intent(BaseModel):
    """意图数据模型

    表示用户的一个意图，包含意图类型、关联事件、目标任务等信息。
    意图识别器（IntentRecognizer）会将用户输入的事件转换为Intent对象，
    然后根据意图类型路由到相应的处理逻辑。

    Attributes:
        intent_type: 意图类型，标识用户想要执行的操作类型
        event: 关联的事件对象，通常是InputEvent，包含用户的原始输入
        target_task_id: 目标任务ID，标识意图针对的任务（如果适用）
        target_task_description: 目标任务描述，当创建新任务时使用，描述要执行的任务
        depend_task_id: 依赖任务ID，用于CONTINUE_TASK意图，标识要接续的任务
        supplementary_info: 补充信息，用于SUPPLEMENT_TASK意图，包含需要补充的信息
        modification_details: 修改详情，用于MODIFY_TASK意图，包含要修改的内容
        confidence: 置信度，表示意图识别的置信度，范围0.0-1.0，默认为1.0
        metadata: 元数据，可以存储额外的意图相关信息
        clarification_prompt: 澄清提示，用于UNKNOWN_TASK意图，向用户请求澄清的问题

    Note:
        - 创建Intent对象后会自动调用_validate()进行验证
    """
    intent_type: IntentType
    event: "Event"
    target_task_id: Optional[str]
    target_task_description: Optional[str] = None
    depend_task_id: str = None
    supplementary_info: Optional[Dict[str, Any]] = None
    modification_details: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    clarification_prompt: Optional[str] = None

    def __post_init__(self):
        """初始化后处理

        确保metadata不为None，并调用验证方法。
        """
        if self.metadata is None:
            self.metadata = {}
        self._validate()

    def _validate(self):
        """验证意图数据

        验证意图对象的字段是否符合要求，例如：
        - 某些意图类型必须包含特定字段
        - 字段值的有效性检查

        Raises:
            ValueError: 如果意图数据无效
        """
        ...