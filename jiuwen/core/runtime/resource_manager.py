from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, TypeVar
from jiuwen.core.utils.tool.base import Tool
from jiuwen.core.utils.prompt.template.template import Template
from jiuwen.core.utils.llm.base import BaseChatModel

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

    def add_tool(self, tool_id: str, tool: Tool) -> None:
        self._tool_mgr.add_tool(tool_id, tool)

    def add_tools(self, tools: List[Tuple[str, Tool]]) -> None:
        self._tool_mgr.add_tools(tools)

    def get_tool(self, tool_id: str) -> Optional[Tool]:
        return self._tool_mgr.get_tool(tool_id)

    def remove_tool(self, tool_id: str):
        self._tool_mgr.remove_tool(tool_id)

    def add_workflow(self, workflow_id: str, workflow: Workflow) -> None:
        self._workflow_mgr.add_workflow(workflow_id, workflow)

    def add_workflows(self, workflows: List[Tuple[str, Workflow]]) -> None:
        self._workflow_mgr.add_workflows(workflows)

    def remove_workflow(self, workflow_id: str) -> None:
        self._workflow_mgr.remove_workflow(workflow_id)

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        return self._workflow_mgr.get_workflow(workflow_id)

    def add_prompt(self, template_id: str, template: Template) -> None:
        self._prompt_mgr.add_prompt(template_id, template)

    def add_prompts(self, templates: List[Tuple[str, Template]]) -> None:
        self._prompt_mgr.add_prompts(templates)

    def remove_prompt(self, template_id: str) -> bool:
        return self._prompt_mgr.remove_prompt(template_id)

    def get_prompt(self, template_id: str) -> Optional[Template]:
        return self._prompt_mgr.get_prompt(template_id)

    def add_model(self, model_id: str, model: BaseChatModel) -> None:
        self._model_mgr.add_model(model_id, model)

    def add_models(self, models: List[Tuple[str, BaseChatModel]]) -> None:
        self._model_mgr.add_models(models)

    def remove_model(self, model_id: str) -> bool:
        return self._model_mgr.remove_model(model_id)

    def get_model(self, model_id: str) -> Optional[BaseChatModel]:
        return self._model_mgr.get_model(model_id)