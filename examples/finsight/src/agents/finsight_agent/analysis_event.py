import uuid

from pydantic import Field
import os
import yaml
import json_repair

from src.common.event import PriorityEvent
from openjiuwen.core.utils.llm.messages import HumanMessage

CURRENT_DIR = os.path.dirname(__file__)


class AnalysisEvent(PriorityEvent):
    priority: int = Field(default=2, description="Priority of task")

    @classmethod
    async def async_create_tasks(cls, goal: str, llm, assignee=None, task_type="general", max_num=10, existing_tasks=None):
        existing_tasks_str = "\n".join(
            [f"- {task.task_description}" for task in existing_tasks]) if existing_tasks else "None"

        if task_type.lower() == "financial_company":
            with open(os.path.join(CURRENT_DIR, "./prompts/financial_prompts.yaml"), "r", encoding="utf-8") as f:
                prompt_template = yaml.safe_load(f)

        else:
            with open(os.path.join(CURRENT_DIR, "./prompts/general_prompts.yaml"), "r", encoding="utf-8") as f:
                prompt_template = yaml.safe_load(f)

        prompt = prompt_template.get("generate_task").format(query=goal, existing_tasks=existing_tasks_str,
                                                             max_num=max_num)
        message = HumanMessage(content=prompt)
        output = await llm.ainvoke([message], response_format={"type": "json_object"})
        output = json_repair.loads(output.content)
        # Handle both list and dict responses
        if isinstance(output, dict):
            output = output.get('tasks', output.get('tasks', output.get('analysis_tasks', [])))

        tasks_to_run = []
        for o in output[:max_num]:
            tasks_to_run.append(
                cls(
                    id=uuid.uuid4().hex[:8],
                    assignee=assignee,
                    type=task_type,
                    task_description=o,
                    status="pending",
                )
            )
        return tasks_to_run

    @classmethod
    def from_description(cls, description, assignee, task_type="general"):
        return cls(
            id=uuid.uuid4().hex[:8],
            assignee=assignee,
            type=task_type,
            task_description=description,
            meta={},
            status="pending",
        )