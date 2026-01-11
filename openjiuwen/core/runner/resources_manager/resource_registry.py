from functools import wraps
from typing import Type, TypeVar, Dict, Union

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.utils.singleton import Singleton

from openjiuwen.core.runner.resources_manager.agent_group_manager import AgentGroupMgr
from openjiuwen.core.runner.resources_manager.agent_manager import AgentMgr
from openjiuwen.core.runner.resources_manager.model_manager import ModelMgr
from openjiuwen.core.runner.resources_manager.prompt_manager import PromptMgr
from openjiuwen.core.runner.resources_manager.thread_safe_dict import ThreadSafeDict
from openjiuwen.core.runner.resources_manager.tool_manager import ToolMgr
from openjiuwen.core.runner.resources_manager.workflow_manager import WorkflowMgr

T = TypeVar("T", AgentGroupMgr, AgentMgr, ModelMgr, PromptMgr, ToolMgr, WorkflowMgr)
ResourceType = Union[AgentGroupMgr, AgentMgr, ModelMgr, PromptMgr, ToolMgr, WorkflowMgr]


class ResourceRegistry(metaclass=Singleton):
    def __init__(self):
        self._constructors: Dict[str, Type[ResourceType]] = {}
        self._instances: ThreadSafeDict[str, ResourceType] = ThreadSafeDict()

        self._register("tool_mgr", ToolMgr)
        self._register("workflow_mgr", WorkflowMgr)
        self._register("prompt_mgr", PromptMgr)
        self._register("model_mgr", ModelMgr)
        self._register("agent_mgr", AgentMgr)
        self._register("agent_group_mgr", AgentGroupMgr)

    def _register(self, key: str, constructor: Type[T]) -> None:
        if key in self._constructors:
            raise ValueError(f"Resource '{key}' is already registered.")
        self._constructors[key] = constructor

    def _get(self, key: str) -> ResourceType:
        if key not in self._constructors:
            raise KeyError(f"Resource '{key}' is not registered.")
        return self._instances.get_or_create(key, lambda: self._constructors[key]())

    def _get_typed(self, key: str, expected_type: Type[T]) -> T:
        instance = self._get(key)
        if not isinstance(instance, expected_type):
            raise TypeError(f"Resource '{key}' is not of type {expected_type.__name__}")
        return instance

    @staticmethod
    def _safe_resource_access(key: str, expected_type: Type[T]):
        def decorator(func):
            @wraps(func)
            def wrapper(self: "ResourceRegistry"):
                try:
                    return self._get_typed(key, expected_type)
                except Exception as e:
                    raise JiuWenBaseException(
                        StatusCode.SESSION_RESOURCE_REGISTRY_FAILED.code,
                        StatusCode.SESSION_RESOURCE_REGISTRY_FAILED.errmsg.format(
                            reason=f"Resource registry {expected_type} error: {str(e)}"
                        )
                    )

            return wrapper

        return decorator

    @_safe_resource_access("tool_mgr", ToolMgr)
    def tool(self) -> ToolMgr:
        ...

    @_safe_resource_access("workflow_mgr", WorkflowMgr)
    def workflow(self) -> WorkflowMgr:
        ...

    @_safe_resource_access("prompt_mgr", PromptMgr)
    def prompt(self) -> PromptMgr:
        ...

    @_safe_resource_access("model_mgr", ModelMgr)
    def model(self) -> ModelMgr:
        ...

    @_safe_resource_access("agent_mgr", AgentMgr)
    def agent(self) -> AgentMgr:
        ...

    @_safe_resource_access("agent_group_mgr", AgentGroupMgr)
    def agent_group(self) -> AgentGroupMgr:
        ...
