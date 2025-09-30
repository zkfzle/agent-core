"""运行状态核心类"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Any

from jiuwen.core.multi_agent.runner.agent_message_queue import MessageQueueState


@dataclass
class RunState:
    """运行状态核心类"""
    # 消息队列状态
    message_queue_state: MessageQueueState = field(default_factory=MessageQueueState)
    # 运行时间信息
    start_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None
    is_running: bool = False

    def update_timestamp(self) -> None:
        """更新时间戳"""
        self.last_update_time = datetime.now(tz=timezone.utc)
