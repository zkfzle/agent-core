from typing import List, Tuple, TypeVar
from jiuwen.core.context.controller_context.thread_safe_dict import ThreadSafeDict

Workflow = TypeVar("Workflow", contravariant=True)

def generate_workflow_key(workflow_id: str, workflow_version: str) -> str:
    return f"{workflow_id}_{workflow_version}"

class WorkflowMgr:
    def __init__(self):
        self._workflows: ThreadSafeDict[str, Workflow] = ThreadSafeDict()

    def add_workflow(self, workflow_id: str, workflow: Workflow) -> None:
        self._workflows[workflow_id] = workflow

    def add_workflows(self, workflows: List[Tuple[str, Workflow]]):
        if not workflows:
            return
        for key, workflow in workflows:
            self._workflows.update({key: workflow})

    def get_workflow(self, workflow_id: str) -> Workflow:
        return self._workflows.get(workflow_id)

    def find_workflow_by_id_and_version(self, workflow_id: str):
        return self._workflows.get(workflow_id)

    def remove_workflow(self, workflow_id: str):
        return self._workflows.pop(workflow_id, None) is not None