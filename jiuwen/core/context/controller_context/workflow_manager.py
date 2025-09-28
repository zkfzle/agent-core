from typing import List, Tuple, TypeVar
from jiuwen.core.context.controller_context.thread_safe_dict import ThreadSafeDict
from jiuwen.core.runtime.state import DEFAULT_WORKFLOW_ID
from jiuwen.core.tracer.decorator import decrate_workflow_with_trace
from jiuwen.core.utils.llm.messages import ToolInfo, Parameters, Function
from jiuwen.core.workflow.workflow_config import WorkflowInputsSchema

Workflow = TypeVar("Workflow", contravariant=True)


def generate_workflow_key(workflow_id: str, workflow_version: str) -> str:
    return f"{workflow_id}_{workflow_version}"


class WorkflowMgr:
    def __init__(self):
        self._workflows: ThreadSafeDict[str, Workflow] = ThreadSafeDict()
        self._workflow_tool_infos: ThreadSafeDict[str, ToolInfo] = ThreadSafeDict()
        self._workflow_schema: ThreadSafeDict[str, WorkflowInputsSchema] = ThreadSafeDict()

    def add_workflow(self, workflow_id: str, workflow: Workflow) -> None:
        self._workflows[workflow_id] = workflow
        self._workflow_tool_infos[workflow_id] = workflow.get_tool_info()

    def add_workflows(self, workflows: List[Tuple[str, Workflow]]):
        if not workflows:
            return
        for key, workflow in workflows:
            self._workflows.update({key: workflow})
            self._workflow_tool_infos.update({key: workflow.get_tool_info()})

    def get_workflow(self, workflow_id: str, runtime=None) -> Workflow:
        workflow = self._workflows.get(workflow_id)
        if not workflow or not runtime or not runtime.tracer():
            return workflow
        return decrate_workflow_with_trace(WrappedWorkflow(workflow), runtime)

    def find_workflow_by_id_and_version(self, workflow_id: str):
        return self._workflows.get(workflow_id)

    def remove_workflow(self, workflow_id: str):
        self._workflow_tool_infos.pop(workflow_id, None)
        return self._workflows.pop(workflow_id, None) is not None

    def get_tool_infos(self, workflow_id: List[str]):
        if not workflow_id:
            return []
        return [self._workflow_tool_infos.get(id) for id in workflow_id]

    def add_schema(self, workflow_id: str, schema: WorkflowInputsSchema):
        if not schema:
            return
        self._workflow_schema.update({workflow_id: schema})

    def get_schema(self, workflow_id: str, runtime=None) -> ToolInfo:
        if not workflow_id:
            workflow_id = DEFAULT_WORKFLOW_ID

        workflow = self._workflows.get(workflow_id)
        workflow_inputs_schema = self._workflow_inputs_schema.get(workflow_id)
        if not workflow or not workflow_inputs_schema:
            return ToolInfo(
                type="function",
                function=Function(
                    name=workflow_id,
                    description="",
                    parameters=Parameters(
                        type="object",
                        properties={},
                        required=[]
                    )
                )
            )

        parameters = Parameters(
            type=workflow_inputs_schema.type,
            properties=workflow_inputs_schema.properties,
            required=workflow_inputs_schema.required
        )

        function = Function(
            name=workflow_id,
            parameters=parameters,
            description=workflow.config().metadata.description or "",
        )

        return ToolInfo(
            type="function",
            function=function
        )

class WrappedWorkflow:
    def __init__(self, workflow):
        self.inner = workflow

    async def invoke(self, inputs, runtime, context = None):
        return await self.inner.invoke(inputs, runtime, context)

    async def stream(self, inputs, runtime, context=None, stream_modes=None):
        result = self.inner.stream(inputs, runtime, context, stream_modes)
        async for item in result:
            yield item

    def get_tool_info(self):
        return self.inner.get_tool_info()

    async def sub_invoke(self, inputs, runtime, config = None):
        return await self.inner.sub_invoke(inputs, runtime, config)

    def get_workflow_metadata(self):
        return self.inner.workflow_config().metadata
