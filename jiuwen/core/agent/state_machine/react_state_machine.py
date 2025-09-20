"""ReAct状态机实现"""
from typing import Dict, Callable, Any

from jiuwen.agent.common.enum import ReActStatus, ReActEvent
from jiuwen.core.common.logging import logger
from jiuwen.core.runtime.runtime import Runtime


class ReActStateMachine:
    """ReAct状态机，管理状态转换和事件处理"""

    def __init__(self, runtime: Runtime):
        self._runtime = runtime
        self._current_status = ReActStatus.INITIALIZED
        self._current_event = ReActEvent.NO_EVENT

        # 状态处理映射
        self._state_handlers: Dict[ReActStatus, Callable] = {}
        self._stream_state_handlers: Dict[ReActStatus, Callable] = {}

    def register_state_handler(self, status: ReActStatus, handler: Callable, is_stream: bool = False):
        """注册状态处理器"""
        if is_stream:
            self._stream_state_handlers[status] = handler
        else:
            self._state_handlers[status] = handler

    def get_current_status(self) -> ReActStatus:
        """获取当前状态"""
        state_data = self._runtime.get_state("react_state")
        if state_data and "status" in state_data:
            self._current_status = ReActStatus(state_data["status"])
        return self._current_status

    def set_current_status(self, status: ReActStatus):
        """设置当前状态"""
        self._current_status = status
        current_data = self.get_state_data()
        current_data["status"] = status.value
        logger.info(f"React State Machine set current status to: {status}")
        self._update_state_to_runtime(current_data)

    def get_current_event(self) -> ReActEvent:
        """获取当前事件"""
        return self._current_event

    def set_current_event(self, event: ReActEvent):
        """设置当前事件"""
        self._current_event = event

    def is_completed(self) -> bool:
        """检查是否已完成"""
        return self.get_current_status() == ReActStatus.COMPLETED

    def is_interrupted(self) -> bool:
        return self.get_current_status() == ReActStatus.INTERRUPTED

    def can_handle_status(self, status: ReActStatus, is_stream: bool = False) -> bool:
        """检查是否可以处理指定状态"""
        handlers = self._stream_state_handlers if is_stream else self._state_handlers
        return status in handlers

    async def handle_state(self, status: ReActStatus, *args, **kwargs) -> Any:
        """处理状态"""
        if status not in self._state_handlers:
            raise ValueError(f"No handler registered for status: {status}")

        logger.info(f"Handling state: {status} with event: {self._current_event}")
        handler = self._state_handlers[status]
        return await handler(*args, **kwargs)

    async def handle_stream_state(self, status: ReActStatus, *args, **kwargs):
        """处理流式状态"""
        if status not in self._stream_state_handlers:
            raise ValueError(f"No stream handler registered for status: {status}")

        logger.info(f"Handling stream state: {status} with event: {self._current_event}")
        handler = self._stream_state_handlers[status]
        async for result in handler(*args, **kwargs):
            yield result

    def get_state_data(self) -> Dict[str, Any]:
        """获取状态数据"""
        return self._runtime.get_state("react_state") or {}

    def update_state_data(self, data: Dict[str, Any]):
        """更新状态数据"""
        current_data = self.get_state_data()
        current_data.update(data)
        self._update_state_to_runtime(current_data)

    def get_final_result(self) -> str:
        """获取最终结果"""
        state_data = self.get_state_data()
        return state_data.get("final_result", "")

    def _update_state_to_runtime(self, data: Dict[str, Any]):
        """更新状态到Runtime"""
        self._runtime.update_state({"react_state": data})