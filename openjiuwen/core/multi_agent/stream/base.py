from enum import Enum
from typing import Dict, Any

class StreamCode(Enum):

    """
    流式输出状态码定义
    """
    START = 2000 #流式开始
    WORKFLOW_START = 3000 # workflow start组件开始执行 当本次调用经过start组件时会存在
    WORKFLOW_END = 4000 # workflow end 组件执行完毕后输出
    MESSAGE_END = 5000 #一个组件的流式结束标识，带有该组件的总结信息
    PARTIAL_CONTENT = 1206 #部分输出
    FINISH = 0 #表示最后一条消息
    ERROR = -1  # 流式过程中组件执行错误
    CONTROLLER_AGENT_HANDOFF_MESSAGE = 14000  # 控制器调用其他Agent事件
    CONTROLLER_AGENT_INTERRUPT_MESSAGE = 15000

class StreamDataMsg(Enum):
    """
    流式数据msg枚举
    """
    SUCCESS = "success"
    FAIL = "fail"
    MESSAGE_END = "message_end"
    FINISH = "finish"

class StreamData:
    def __init__(self, code: StreamCode, msg: str, data: Any, execution_id: str, index: int = 0):
        self.code = code
        self.msg = msg
        self.data = data
        self.execution_id = execution_id
        self.index = index

    def __str__(self):
        return f"StreamData(code={self.code}, msg={self.msg}, data={self.data}, execution_id={self.execution_id}, index={self.index})"

