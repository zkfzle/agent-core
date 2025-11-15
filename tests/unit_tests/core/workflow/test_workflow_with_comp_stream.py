import asyncio
import os
from typing import AsyncIterator

import pytest

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.logging import logger
from openjiuwen.core.component.base import WorkflowComponent
from openjiuwen.core.component.end_comp import End, EndConfig
from openjiuwen.core.component.start_comp import Start
from openjiuwen.core.component.workflow_comp import SubWorkflowComponent
from openjiuwen.core.context_engine.base import Context
from openjiuwen.core.graph.executable import Executable
from openjiuwen.core.runtime.base import ComponentExecutable, Input, Output
from openjiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from openjiuwen.core.runtime.runtime import BaseRuntime, Runtime
from openjiuwen.core.runtime.workflow import WorkflowRuntime
from openjiuwen.core.stream.base import StreamMode, BaseStreamMode
from openjiuwen.core.workflow.base import Workflow, WorkflowOutput, WorkflowChunk
from openjiuwen.core.workflow.workflow_config import WorkflowConfig, WorkflowMetadata

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
        logger.debug(f"producer inputs: {inputs}")
        for v in inputs.get("array"):
            logger.info(f"send stream frame {v}")
            yield {"output": v}


async def test_multi_stream_workflow():
    wf = create_component_stream_workflow_with_template()

    async for chunk in wf.stream({"inputs": [1, 2, 3]}, WorkflowRuntime(), stream_modes=[BaseStreamMode.OUTPUT]):
        assert chunk is not None
        print(chunk.model_dump_json(indent=4))

    res = await wf.invoke({"inputs": [1, 2, 3]}, WorkflowRuntime())
    print(res.model_dump_json(indent=4))

async def test_batch_multi_stream_workflow():
    def create_component_workflow_with_template() -> Workflow:
        workflow = Workflow()
        workflow.set_start_comp("start", Start(), inputs_schema={"array": "${inputs}"})
        workflow.add_workflow_comp("a", Producer(), inputs_schema={"array": "${start.array}"})
        workflow.add_workflow_comp("b", Producer(), inputs_schema={"array": "${start.array}"})
        workflow.add_workflow_comp("c", Producer(), inputs_schema={"array": "${start.array}"})
        workflow.add_workflow_comp("batch", Producer(), inputs_schema={"array": "${start.array}"})
        end = End(EndConfig(responseTemplate="a: {{a}}; c: {{c}}; batch: {{batch}}; b: {{b}}"))
        workflow.set_end_comp("end", end,
                              inputs_schema={"batch": "${batch.output}"},
                              stream_inputs_schema={"a": "${a.output}", "b": "${b.output}", "c": "${c.output}"})

        workflow.add_connection("start", "a")
        workflow.add_connection("start", "b")
        workflow.add_connection("start", "c")
        workflow.add_connection("start", "batch")
        workflow.add_stream_connection("a", "end")
        workflow.add_stream_connection("b", "end")
        workflow.add_stream_connection("c", "end")
        workflow.add_connection("batch", "end")
        return workflow
    wf = create_component_workflow_with_template()

    res = await wf.invoke({"inputs": [1, 2, 3]}, WorkflowRuntime())
    print(res.model_dump_json(indent=4))

    async for chunk in wf.stream({"inputs": [1, 2, 3]}, WorkflowRuntime(), stream_modes=[BaseStreamMode.OUTPUT]):
        assert chunk is not None
        print(chunk.model_dump_json(indent=4))

def create_component_stream_workflow_with_template() -> Workflow:
    workflow = Workflow()
    workflow.set_start_comp("start", Start(), inputs_schema={"array": "${inputs}"})
    workflow.add_workflow_comp("a", Producer(), inputs_schema={"array": "${start.array}"})
    workflow.add_workflow_comp("b", Producer(), inputs_schema={"array": "${start.array}"})
    workflow.add_workflow_comp("c", Producer(), inputs_schema={"array": "${start.array}"})
    workflow.add_workflow_comp("batch", Producer(), inputs_schema={"array": "${start.array}"})
    end = End(EndConfig(responseTemplate="a: {{a}}; c: {{c}}; batch: {{batch}}; b: {{b}}"))
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
    return workflow


async def test_stream_component_in_sub_workflow_with_invoke():
    def create_component_invoke_workflow_without_template() -> Workflow:
        workflow = Workflow()
        workflow.set_start_comp("start", Start(), inputs_schema={"array": "${inputs}"})
        workflow.add_workflow_comp("a", Producer(), inputs_schema={"array": "${start.array}"})
        workflow.add_workflow_comp("b", Producer(), inputs_schema={"array": "${start.array}"})
        workflow.add_workflow_comp("c", Producer(), inputs_schema={"array": "${start.array}"})
        workflow.add_workflow_comp("batch", Producer(), inputs_schema={"array": "${start.array}"})
        workflow.set_end_comp("end", End(),
                              inputs_schema={"batch": "${batch.output}"},
                              stream_inputs_schema={"a": "${a.output}", "b": "${b.output}", "c": "${c.output}"})

        workflow.add_connection("start", "a")
        workflow.add_connection("start", "b")
        workflow.add_connection("start", "c")
        workflow.add_connection("start", "batch")
        workflow.add_stream_connection("a", "end")
        workflow.add_stream_connection("b", "end")
        workflow.add_stream_connection("c", "end")
        workflow.add_connection("batch", "end")
        return workflow

    wf = Workflow()
    wf.set_start_comp("main_start", Start(), inputs_schema={"array": "${inputs}"})
    wf.add_workflow_comp("workflow", SubWorkflowComponent(create_component_invoke_workflow_without_template()),
                         inputs_schema={"inputs": "${main_start.array}"})
    end = End(EndConfig(responseTemplate="sub_workflow: {{sub_workflow}}"))
    wf.set_end_comp("main_end", end,
                    inputs_schema={"sub_workflow": "${workflow.batch}"},
                    response_mode="streaming")

    wf.add_connection("main_start", "workflow")
    wf.add_connection("workflow", "main_end")

    async for chunk in wf.stream({"inputs": [1, 2, 3]}, WorkflowRuntime(), stream_modes=[BaseStreamMode.OUTPUT]):
        assert chunk is not None
        print(chunk.model_dump_json(indent=4))


async def test_stream_component_in_sub_workflow_with_stream():
    wf = Workflow()
    wf.set_start_comp("main_start", Start(), inputs_schema={"array": "${inputs}"})
    wf.add_workflow_comp("workflow", SubWorkflowComponent(create_component_stream_workflow_with_template()),
                         inputs_schema={"inputs": "${main_start.array}"})
    end = End(EndConfig(responseTemplate="sub_workflow: {{sub_workflow}}"))
    wf.set_end_comp("main_end", end,
                    inputs_schema={"sub_workflow": "${workflow.stream}"},
                    response_mode="streaming")

    wf.add_connection("main_start", "workflow")
    wf.add_connection("workflow", "main_end")

    async for chunk in wf.stream({"inputs": [1, 2, 3]}, WorkflowRuntime(), stream_modes=[BaseStreamMode.OUTPUT]):
        assert chunk is not None
        print(chunk.model_dump_json(indent=4))


class Interaction(WorkflowComponent, ComponentExecutable):
    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        result = await runtime.interact("please enter any input")
        return {"output": result}


async def test_interaction_with_stream():
    def create_workflow() -> Workflow:
        wf = Workflow(workflow_config=WorkflowConfig(metadata=WorkflowMetadata(id="test_interaction_with_stream"),
                                                     stream_timeout=0.5))
        wf.set_start_comp("start", Start(), inputs_schema={"array": "${inputs}"})
        wf.add_workflow_comp("interaction", Interaction())
        wf.add_workflow_comp("stream", Producer(), inputs_schema={"array": "${start.array}"})
        end = End(EndConfig(responseTemplate="a: {{a}}; batch: {{batch}}"))
        wf.set_end_comp("end", end,
                        inputs_schema={"batch": "${interaction.output}"},
                        stream_inputs_schema={"a": "${stream.output}"},
                        response_mode="streaming")

        wf.add_connection("start", "interaction")
        wf.add_connection("start", "stream")
        wf.add_connection("interaction", "end")
        wf.add_stream_connection("stream", "end")
        return wf

    wf1 = create_workflow()
    wf2 = create_workflow()

    async for chunk in wf1.stream({"inputs": [1, 2, 3]}, WorkflowRuntime(session_id="123"),
                                  stream_modes=[BaseStreamMode.OUTPUT]):
        assert chunk is not None
        print(chunk.model_dump_json(indent=4))

    logger.debug("human in the loop...")

    async for chunk in wf2.stream(InteractiveInput({"inputs": [1, 2, 3]}), WorkflowRuntime(session_id="123"),
                                  stream_modes=[BaseStreamMode.OUTPUT]):
        assert chunk is not None
        print(chunk.model_dump_json(indent=4))


async def test_interaction_with_exception():
    run_times = 0

    class ExceptionComp(WorkflowComponent, ComponentExecutable):
        async def stream(self, inputs: Input, runtime: Runtime, context: Context) -> AsyncIterator[Output]:
            if run_times == 0:
                raise Exception("first time")
            else:
                for i in range(10):
                    yield dict(output=i)

    def create_workflow_with_exception() -> Workflow:
        wf = Workflow(workflow_config=WorkflowConfig(metadata=WorkflowMetadata(id="test_interaction_with_exception"),
                                                     stream_timeout=0.5))
        wf.set_start_comp("start", Start(), inputs_schema={"array": "${inputs}"})
        wf.add_workflow_comp("exception", ExceptionComp())
        end = End(EndConfig(responseTemplate="a: {{a}}; batch: {{batch}}"))
        wf.set_end_comp("end", end,
                        stream_inputs_schema={"a": "${exception.output}"},
                        response_mode="streaming")

        wf.add_connection("start", "exception")
        wf.add_stream_connection("exception", "end")
        return wf

    wf1 = create_workflow_with_exception()
    wf2 = create_workflow_with_exception()

    try:
        res = await wf1.invoke({"inputs": [1, 2, 3]}, WorkflowRuntime(session_id="123"))
        print(res.model_dump_json(indent=4))
    except Exception as e:
        logger.error(e)
    run_times += 1
    logger.debug("human in the loop...")

    res = await wf2.invoke(InteractiveInput({"inputs": [1, 2, 3]}), WorkflowRuntime(session_id="123"))

    print(res.model_dump_json(indent=4))
