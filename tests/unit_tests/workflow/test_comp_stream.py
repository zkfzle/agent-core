import asyncio
import os
from typing import AsyncIterator

import pytest

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.common.logging import logger
from jiuwen.core.component.base import WorkflowComponent
from jiuwen.core.component.end_comp import End, EndConfig
from jiuwen.core.component.start_comp import Start
from jiuwen.core.context_engine.base import Context
from jiuwen.core.graph.executable import Executable
from jiuwen.core.runtime.base import ComponentExecutable, Input, Output
from jiuwen.core.runtime.runtime import BaseRuntime, Runtime
from jiuwen.core.runtime.workflow import WorkflowRuntime
from jiuwen.core.stream.base import StreamMode, BaseStreamMode
from jiuwen.core.workflow.base import Workflow, WorkflowOutput, WorkflowChunk
from jiuwen.core.workflow.workflow_config import WorkflowConfig

pytestmark = pytest.mark.asyncio

os.environ.setdefault("LLM_SSL_VERIFY", "false")


class MockStreamNode(ComponentExecutable, WorkflowComponent):
    def __init__(self):
        super().__init__()

    async def invoke(self, inputs, runtime: BaseRuntime, context: Context = None) -> WorkflowOutput:
        return inputs

    async def stream(
            self,
            inputs,
            runtime: BaseRuntime,
            context: Context = None,
            stream_modes: list[StreamMode] = None
    ) -> AsyncIterator[WorkflowChunk]:
        await asyncio.sleep(0.3)
        yield inputs

    def to_executable(self) -> Executable:
        return self


async def test_no_stream_called():
    with pytest.raises(JiuWenBaseException) as error:
        config = WorkflowConfig(stream_timeout=0.2)
        flow = Workflow(config)
        flow.set_start_comp("start", Start())
        flow.set_end_comp("end", End(), inputs_schema={}, response_mode="streaming")
        flow.add_workflow_comp("stream", MockStreamNode(), inputs_schema={})
        flow.add_connection("start", "stream")
        flow.add_stream_connection("stream", "end")

        await flow.invoke({"a": "生成markdown回复"}, WorkflowRuntime())

    assert error.value.error_code == StatusCode.STREAM_FRAME_TIMEOUT_FAILED.code
    with pytest.raises(JiuWenBaseException) as error:
        async for chunk in flow.stream({"a": "生成markdown回复"}, WorkflowRuntime(),
                                       stream_modes=[BaseStreamMode.OUTPUT]):
            print(chunk)
    assert error.value.error_code == StatusCode.STREAM_FRAME_TIMEOUT_FAILED.code


class Producer(ComponentExecutable, WorkflowComponent):
    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        return {"output": inputs.get("array")}

    async def stream(self, inputs: Input, runtime: Runtime, context: Context) -> AsyncIterator[Output]:
        for v in inputs.get("array"):
            logger.info(f"send stream frame {v}")
            yield {"output": v}


async def test_multi_stream_workflow():
    workflow = Workflow()
    workflow.set_start_comp("start", Start(), inputs_schema={"array": "${inputs}"})
    workflow.add_workflow_comp("a", Producer(), inputs_schema={"array": "${start.array}"})
    workflow.add_workflow_comp("b", Producer(), inputs_schema={"array": "${start.array}"})
    workflow.add_workflow_comp("c", Producer(), inputs_schema={"array": "${start.array}"})
    workflow.add_workflow_comp("batch", Producer(), inputs_schema={"array": "${start.array}"})
    end = End(EndConfig(responseTemplate="a: {{a}}; c: {{c}}; batch: {{batch}}; b: {{b}}"))
    end2= End()
    workflow.set_end_comp("end", end,
                          inputs_schema={"batch": "${batch.output}"},
                          stream_inputs_schema={"a": "${a.output}", "b": "${b.output}", "c": "${c.output}"},
                          response_mode="streaming")

    workflow.add_connection("start", "a")
    workflow.add_connection("start", "b")
    workflow.add_connection("start", "c")
    workflow.add_connection("start", "batch")
    workflow.add_stream_connection("a", "end")
    workflow.add_stream_connection("b", "end")
    workflow.add_stream_connection("c", "end")
    workflow.add_connection("batch", "end")

    async for chunk in workflow.stream({"inputs": [1, 2, 3]}, WorkflowRuntime(), stream_modes=[BaseStreamMode.OUTPUT]):
        assert chunk is not None
        print(chunk.model_dump_json(indent=4))
