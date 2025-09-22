from typing import List
from jiuwen.core.workflow.base import Workflow
from jiuwen.core.context.controller_context.thread_safe_dict import ThreadSafeDict


def generate_workflow_key(workflow_id: str, workflow_version: str) -> str:
    return f"{workflow_id}_{workflow_version}"

class WorkflowMgr:
    def __init__(self, agent_config):
        self._workflows: ThreadSafeDict[str, Workflow] = ThreadSafeDict()

    def add_workflow(self, workflow: Workflow) -> None:
        meta = workflow.config().metadata
        key = f"{meta.id}_{meta.version}"
        self._workflows[key] = workflow

    def add_workflows(self, workflows: List[Workflow]):
        if not workflows:
            return
        for workflow in workflows:
            workflow_id = workflow.config().metadata.id
            workflow_version = workflow.config().metadata.version
            self._workflows.update({f"{workflow_id}_{workflow_version}": workflow})

    def get_workflow(self, workflow_key: str) -> Workflow:
        return self._workflows.get(workflow_key)

    def find_workflow_by_id_and_version(self, workflow_id: str, workflow_version: str):
        return self._workflows.get(f"{workflow_id}_{workflow_version}")

    def remove_workflow(self, workflow_key: str):
        return self._workflows.pop(workflow_key, None) is not None