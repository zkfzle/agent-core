from typing import List, Tuple, TypeVar
from jiuwen.core.context.controller_context.thread_safe_dict import ThreadSafeDict
from jiuwen.core.utils.llm.messages import ToolInfo

Workflow = TypeVar("Workflow", contravariant=True)


def generate_workflow_key(workflow_id: str, workflow_version: str) -> str:
    return f"{workflow_id}_{workflow_version}"


class WorkflowMgr:
    def __init__(self):
        self._workflows: ThreadSafeDict[str, Workflow] = ThreadSafeDict()
        self._workflow_tool_infos: ThreadSafeDict[str, ToolInfo] = ThreadSafeDict()

    def add_workflow(self, workflow_id: str, workflow: Workflow) -> None:
        self._workflows[workflow_id] = workflow
        self._workflow_tool_infos[workflow_id] = workflow.get_tool_info()

    def add_workflows(self, workflows: List[Tuple[str, Workflow]]):
        if not workflows:
            return
        for key, workflow in workflows:
            self._workflows.update({key: workflow})
            self._workflow_tool_infos.update({key: workflow.get_tool_info()})

    def get_workflow(self, workflow_id: str) -> Workflow:
        return self._workflows.get(workflow_id)

    def find_workflow_by_id_and_version(self, workflow_id: str):
        return self._workflows.get(workflow_id)

    def remove_workflow(self, workflow_id: str):
        self._workflow_tool_infos.pop(workflow_id, None)
        return self._workflows.pop(workflow_id, None) is not None

    def get_tool_infos(self, workflow_id: List[str]):
        if not workflow_id:
            return []
        return [self._workflow_tool_infos.get(id) for id in workflow_id]
