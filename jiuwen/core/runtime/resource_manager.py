from abc import ABC, abstractmethod
from typing import TypeVar

from jiuwen.core.context.controller_context.tool_manager import ToolMgr
from jiuwen.core.context.controller_context.workflow_manager import WorkflowMgr
from jiuwen.core.context.controller_context.prompt_manager import PromptMgr
from jiuwen.core.context.controller_context.model_manager import ModelMgr

Workflow = TypeVar("Workflow", contravariant=True)

class ResourceManager(ABC):
    @abstractmethod
    def tool(self) -> ToolMgr:
        pass

    @abstractmethod
    def prompt(self) -> PromptMgr:
        pass

    @abstractmethod
    def model(self) -> ModelMgr:
        pass

    @abstractmethod
    def workflow(self) -> WorkflowMgr:
        pass

class ResourceMgr(ResourceManager):
    """线程安全单机资源管理器，封装多个管理器"""
    def __init__(self) -> None:
        self._tool_mgr = ToolMgr()
        self._workflow_mgr = WorkflowMgr()
        self._prompt_mgr = PromptMgr()
        self._model_mgr = ModelMgr()

    def tool(self) -> ToolMgr:
        return self._tool_mgr

    def prompt(self) -> PromptMgr:
        return self._prompt_mgr

    def model(self) -> ModelMgr:
        return self._model_mgr

    def workflow(self) -> WorkflowMgr:
        return self._workflow_mgr