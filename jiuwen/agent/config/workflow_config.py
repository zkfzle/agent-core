from typing import List, Dict, Any

from pydantic import Field

from jiuwen.agent.common.schema import WorkflowSchema
from jiuwen.agent.config.base import AgentConfig
from jiuwen.agent.config.react_config import ConstrainConfig


class WorkflowAgentConfig(AgentConfig):
    # 全局超时（秒）
    timeout: int = Field(default=60, ge=1)
    # 开始工作流
    start_workflow: WorkflowSchema = Field(default_factory=WorkflowSchema)
    # 结束工作流
    end_workflow: WorkflowSchema = Field(default_factory=WorkflowSchema)
    # 全局变量
    global_variables: List[dict] = Field(default_factory=list)
    # 全局参数模板（可选）
    global_params: Dict[str, Any] = Field(default_factory=dict)

    constrain: ConstrainConfig = Field(default=ConstrainConfig())

    @property
    def is_single_workflow(self) -> bool:
        return len(self.workflows) == 1
