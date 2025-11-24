import asyncio
import os
from typing import AsyncIterator

import pytest

from openjiuwen.core.common.constants.constant import INTERACTION
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
from openjiuwen.core.runtime.constants import END_COMP_TEMPLATE_RENDER_POSITION_TIMEOUT_KEY, WORKFLOW_INVOKE_TIMEOUT, \
    WORKFLOW_STREAM_TIMEOUT
from openjiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from openjiuwen.core.runtime.runtime import BaseRuntime, Runtime
from openjiuwen.core.runtime.workflow import WorkflowRuntime
from openjiuwen.core.stream.base import StreamMode, BaseStreamMode, OutputSchema
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
        config = WorkflowConfig()
        flow = Workflow(config)
        flow.set_start_comp("start", Start())
        flow.set_end_comp("end", End(), inputs_schema={}, response_mode="streaming")
        flow.add_workflow_comp("stream", MockStreamNode(), inputs_schema={})
        flow.add_connection("start", "stream")
        flow.add_stream_connection("stream", "end")
        runtime = WorkflowRuntime()
        runtime.config().set_envs({WORKFLOW_INVOKE_TIMEOUT:0.2})
        await flow.invoke({"a": "生成markdown回复"}, runtime)

    assert error.value.error_code == StatusCode.WORKFLOW_INVOKE_TIMEOUT.code
    with pytest.raises(JiuWenBaseException) as error:
        runtime = WorkflowRuntime()
        runtime.config().set_envs({WORKFLOW_STREAM_TIMEOUT:0.2})
        async for chunk in flow.stream({"a": "生成markdown回复"}, runtime,
                                       stream_modes=[BaseStreamMode.OUTPUT]):
            print(chunk)
    assert error.value.error_code == StatusCode.WORKFLOW_STREAM_TIMEOUT.code


class Producer(ComponentExecutable, WorkflowComponent):
    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        return {"output": inputs.get("array")}

    async def stream(self, inputs: Input, runtime: Runtime, context: Context) -> AsyncIterator[Output]:
        logger.debug(f"producer inputs: {inputs}")
        for v in inputs.get("array"):
            logger.debug(f"send stream frame {v}")
            yield {"output": v}


async def test_multi_stream_workflow():
    wf = create_component_stream_workflow_with_template()
    chunks = []
    expect_chunks = [
        OutputSchema(type='end node stream', index=0, payload={'answer': 'a: '}),
        OutputSchema(type='end node stream', index=1, payload={'answer': 1}),
        OutputSchema(type='end node stream', index=2, payload={'answer': 2}),
        OutputSchema(type='end node stream', index=3, payload={'answer': 3}),
        OutputSchema(type='end node stream', index=4, payload={'answer': '; c: '}),
        OutputSchema(type='end node stream', index=5, payload={'answer': 1}),
        OutputSchema(type='end node stream', index=6, payload={'answer': 2}),
        OutputSchema(type='end node stream', index=7, payload={'answer': 3}),
        OutputSchema(type='end node stream', index=8, payload={'answer': '; batch: '}),
        OutputSchema(type='end node stream', index=9, payload={'answer': [1, 2, 3]}),
        OutputSchema(type='end node stream', index=10, payload={'answer': '; b: '}),
        OutputSchema(type='end node stream', index=11, payload={'answer': 1}),
        OutputSchema(type='end node stream', index=12, payload={'answer': 2}),
        OutputSchema(type='end node stream', index=13, payload={'answer': 3})]

    async for chunk in wf.stream({"inputs": [1, 2, 3]}, WorkflowRuntime(), stream_modes=[BaseStreamMode.OUTPUT]):
        assert chunk is not None
        print(chunk)
        chunks.append(chunk)
    assert chunks == expect_chunks


    res = await wf.invoke({"inputs": [1, 2, 3]}, WorkflowRuntime())
    print(res)
    assert res.result == expect_chunks

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
    print(res)
    assert res.result == {'responseContent': 'a: 123; c: 123; batch: [1, 2, 3]; b: 123', 'output': {}}

    chunks = []
    expect_chunks = [OutputSchema(type='workflow_final', index=0, payload={'responseContent': 'a: 123; c: 123; batch: [1, 2, 3]; b: 123', 'output': {}})]

    async for chunk in wf.stream({"inputs": [1, 2, 3]}, WorkflowRuntime(), stream_modes=[BaseStreamMode.OUTPUT]):
        assert chunk is not None
        print(chunk)
        chunks.append(chunk)
    assert chunks == expect_chunks

def create_component_stream_workflow_with_template() -> Workflow:
    config = WorkflowConfig()
    workflow = Workflow(config)
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

def create_component_stream_workflow_without_template() -> Workflow:
    config = WorkflowConfig()
    workflow = Workflow(config)
    workflow.set_start_comp("start", Start(), inputs_schema={"array": "${inputs}"})
    workflow.add_workflow_comp("a", Producer(), inputs_schema={"array": "${start.array}"})
    workflow.add_workflow_comp("b", Producer(), inputs_schema={"array": "${start.array}"})
    workflow.add_workflow_comp("c", Producer(), inputs_schema={"array": "${start.array}"})
    workflow.add_workflow_comp("batch", Producer(), inputs_schema={"array": "${start.array}"})
    workflow.set_end_comp("end", End(),
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
    chunks = []
    expect_chunks = [
        OutputSchema(type='end node stream', index=0, payload={'answer': 'sub_workflow: '}),
        OutputSchema(type='end node stream', index=1, payload={'answer': [1, 2, 3]})]

    async for chunk in wf.stream({"inputs": [1, 2, 3]}, WorkflowRuntime(), stream_modes=[BaseStreamMode.OUTPUT]):
        assert chunk is not None
        print(chunk)
        chunks.append(chunk)
    assert expect_chunks == chunks

async def test_stream_component_in_sub_workflow_with_stream():
    wf = Workflow()
    wf.set_start_comp("main_start", Start(), inputs_schema={"array": "${inputs}"})
    wf.add_workflow_comp("workflow", SubWorkflowComponent(create_component_stream_workflow_with_template()),
                         inputs_schema={"inputs": "${main_start.array}"})
    end = End(EndConfig(responseTemplate="sub_workflow: {{sub_workflow}}"))
    wf.set_end_comp("main_end", end,
                    stream_inputs_schema={"sub_workflow": "${workflow.answer}"},
                    response_mode="streaming")

    wf.add_connection("main_start", "workflow")
    wf.add_stream_connection("workflow", "main_end")

    chunks = []
    expect_chunks = [
        OutputSchema(type='end node stream', index=0, payload={'answer': 'sub_workflow: '}),
        OutputSchema(type='end node stream', index=1, payload={'answer': 'a: '}),
        OutputSchema(type='end node stream', index=2, payload={'answer': 1}),
        OutputSchema(type='end node stream', index=3, payload={'answer': 2}),
        OutputSchema(type='end node stream', index=4, payload={'answer': 3}),
        OutputSchema(type='end node stream', index=5, payload={'answer': '; c: '}),
        OutputSchema(type='end node stream', index=6, payload={'answer': 1}),
        OutputSchema(type='end node stream', index=7, payload={'answer': 2}),
        OutputSchema(type='end node stream', index=8, payload={'answer': 3}),
        OutputSchema(type='end node stream', index=9, payload={'answer': '; batch: '}),
        OutputSchema(type='end node stream', index=10, payload={'answer': [1, 2, 3]}),
        OutputSchema(type='end node stream', index=11, payload={'answer': '; b: '}),
        OutputSchema(type='end node stream', index=12, payload={'answer': 1}),
        OutputSchema(type='end node stream', index=13, payload={'answer': 2}),
        OutputSchema(type='end node stream', index=14, payload={'answer': 3})]

    async for chunk in wf.stream({"inputs": [1, 2, 3]}, WorkflowRuntime(), stream_modes=[BaseStreamMode.OUTPUT]):
        assert chunk is not None
        print(chunk)
        chunks.append(chunk)
    assert chunks == expect_chunks


async def test_stream_component_in_sub_workflow_with_stream_collect():
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

    chunks = []
    expect_chunks = [
        OutputSchema(type='end node stream', index=0, payload={'answer': 'sub_workflow: '}),
        OutputSchema(type='end node stream', index=1, payload={'answer': [
            {'answer': 'a: '},
            {'answer': 1},
            {'answer': 2},
            {'answer': 3},
            {'answer': '; c: '},
            {'answer': 1},
            {'answer': 2},
            {'answer': 3},
            {'answer': '; batch: '},
            {'answer': [1, 2, 3]},
            {'answer': '; b: '},
            {'answer': 1},
            {'answer': 2},
            {'answer': 3}]})]

    async for chunk in wf.stream({"inputs": [1, 2, 3]}, WorkflowRuntime(), stream_modes=[BaseStreamMode.OUTPUT]):
        assert chunk is not None
        print(chunk)
        chunks.append(chunk)
    assert expect_chunks == chunks


# Test the ability of workflow components to stream between components
async def test_stream_component_in_sub_workflow_with_substream():
    wf = Workflow(workflow_config=WorkflowConfig())
    wf.set_start_comp("main_start", Start(), inputs_schema={"array": "${inputs}"})
    wf.add_workflow_comp("workflow", SubWorkflowComponent(create_component_stream_workflow_without_template()),
                         inputs_schema={"inputs": "${main_start.array}"})
    end = End()
    wf.set_end_comp("main_end", end,
                    stream_inputs_schema={"sub_workflow": "${workflow.output}"},
                    response_mode="streaming")

    wf.add_connection("main_start", "workflow")
    wf.add_stream_connection("workflow", "main_end")
    chunks = []
    expect_chunks = [
         {'output': {'sub_workflow': {'a': 1}}},
         {'output': {'sub_workflow': {'a': 2}}},
         {'output': {'sub_workflow': {'a': 3}}},
         {'output': {'sub_workflow': {'b': 1}}},
         {'output': {'sub_workflow': {'b': 2}}},
         {'output': {'sub_workflow': {'b': 3}}},
         {'output': {'sub_workflow': {'c': 1}}},
         {'output': {'sub_workflow': {'c': 2}}},
         {'output': {'sub_workflow': {'c': 3}}},
         {'output': {'sub_workflow': {'batch': [1, 2, 3]}}}
    ]
    async for chunk in wf.stream({"inputs": [1, 2, 3]}, WorkflowRuntime(), stream_modes=[BaseStreamMode.OUTPUT]):
        assert chunk is not None
        print(chunk)
        chunks.append(chunk.payload)
    assert len(chunks) == len(expect_chunks)
    for chunk in chunks:
        assert chunk in expect_chunks

# Test the ability of workflow components to stream between components with templates
async def test_stream_component_in_sub_workflow_with_substream_template():
    wf = Workflow(workflow_config=WorkflowConfig())
    wf.set_start_comp("main_start", Start(), inputs_schema={"array": "${inputs}"})
    wf.add_workflow_comp("workflow", SubWorkflowComponent(create_component_stream_workflow_with_template()),
                         inputs_schema={"inputs": "${main_start.array}"})
    end = End()
    wf.set_end_comp("main_end", end,
                    stream_inputs_schema={"sub_workflow": "${workflow.answer}"},
                    response_mode="streaming")

    wf.add_connection("main_start", "workflow")
    wf.add_stream_connection("workflow", "main_end")

    chunks = []
    expect_chunks = [
        OutputSchema(type='end node stream', index=0, payload={'output': {'sub_workflow': 'a: '}}),
        OutputSchema(type='end node stream', index=1, payload={'output': {'sub_workflow': 1}}),
        OutputSchema(type='end node stream', index=2, payload={'output': {'sub_workflow': 2}}),
        OutputSchema(type='end node stream', index=3, payload={'output': {'sub_workflow': 3}}),
        OutputSchema(type='end node stream', index=4, payload={'output': {'sub_workflow': '; c: '}}),
        OutputSchema(type='end node stream', index=5, payload={'output': {'sub_workflow': 1}}),
        OutputSchema(type='end node stream', index=6, payload={'output': {'sub_workflow': 2}}),
        OutputSchema(type='end node stream', index=7, payload={'output': {'sub_workflow': 3}}),
        OutputSchema(type='end node stream', index=8, payload={'output': {'sub_workflow': '; batch: '}}),
        OutputSchema(type='end node stream', index=9, payload={'output': {'sub_workflow': [1, 2, 3]}}),
        OutputSchema(type='end node stream', index=10, payload={'output': {'sub_workflow': '; b: '}}),
        OutputSchema(type='end node stream', index=11, payload={'output': {'sub_workflow': 1}}),
        OutputSchema(type='end node stream', index=12, payload={'output': {'sub_workflow': 2}}),
        OutputSchema(type='end node stream', index=13, payload={'output': {'sub_workflow': 3}})]

    async for chunk in wf.stream({"inputs": [1, 2, 3]}, WorkflowRuntime(), stream_modes=[BaseStreamMode.OUTPUT]):
        assert chunk is not None
        print(chunk)
        chunks.append(chunk)

    print(chunks)
    assert expect_chunks == chunks


class Interaction(WorkflowComponent, ComponentExecutable):
    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        result = await runtime.interact("please enter any input")
        return {"output": result}


async def test_interaction_with_stream():
    def create_workflow() -> Workflow:
        wf = Workflow(workflow_config=WorkflowConfig(metadata=WorkflowMetadata(id="test_interaction_with_stream")))
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
    chunks = []
    interaction = False
    async for chunk in wf1.stream({"inputs": [1, 2, 3]}, WorkflowRuntime(session_id="123"),
                                  stream_modes=[BaseStreamMode.OUTPUT]):
        assert chunk is not None
        print(chunk)
        if chunk.type == INTERACTION:
            interaction = True
    assert interaction

    logger.debug("human in the loop...")
    runtime = WorkflowRuntime(session_id="123")
    actual_chunks = []
    expect_chunks = [OutputSchema(type='end node stream', index=0, payload={'answer': 'a: '}),
                     OutputSchema(type='end node stream', index=1, payload={'answer': '; batch: '}),
                     OutputSchema(type='end node stream', index=2, payload={'answer': {'inputs': [1, 2, 3]}})]

    async for chunk in wf2.stream(InteractiveInput({"inputs": [1, 2, 3]}), runtime,
                                  stream_modes=[BaseStreamMode.OUTPUT]):
        assert chunk is not None
        print(chunk)
        actual_chunks.append(chunk)
    assert actual_chunks == expect_chunks


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
        wf = Workflow(workflow_config=WorkflowConfig(metadata=WorkflowMetadata(id="test_interaction_with_exception")))
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
        print(res)
    except Exception as e:
        logger.error(e)
    run_times += 1
    logger.debug("human in the loop...")
    runtime = WorkflowRuntime(session_id="123")
    runtime.config().set_envs({END_COMP_TEMPLATE_RENDER_POSITION_TIMEOUT_KEY:1})
    res = await wf2.invoke(InteractiveInput({"inputs": [1, 2, 3]}), runtime)

    expect_result = [
              OutputSchema(type='end node stream', index=0, payload={'answer': 'a: '}),
              OutputSchema(type='end node stream', index=1, payload={'answer': 0}),
              OutputSchema(type='end node stream', index=2, payload={'answer': 1}),
              OutputSchema(type='end node stream', index=3, payload={'answer': 2}),
              OutputSchema(type='end node stream', index=4, payload={'answer': 3}),
              OutputSchema(type='end node stream', index=5, payload={'answer': 4}),
              OutputSchema(type='end node stream', index=6, payload={'answer': 5}),
              OutputSchema(type='end node stream', index=7, payload={'answer': 6}),
              OutputSchema(type='end node stream', index=8, payload={'answer': 7}),
              OutputSchema(type='end node stream', index=9, payload={'answer': 8}),
              OutputSchema(type='end node stream', index=10, payload={'answer': 9}),
              OutputSchema(type='end node stream', index=11, payload={'answer': '; batch: '})
    ]

    assert res.result == expect_result


class StreamNodeWithException(WorkflowComponent, ComponentExecutable):
    def __init__(self):
        super().__init__()
        self._raise_error: bool = True

    async def stream(self, inputs: Input, runtime: Runtime, context: Context) -> AsyncIterator[Output]:
        array = inputs.get("array")
        for item in array:
            yield {"array": item}

    async def transform(self, inputs: Input, runtime: Runtime, context: Context) -> AsyncIterator[Output]:
        iter = inputs.get("array")
        i = 0
        async for item in iter:
            yield {'array': item}
            i += 1
            if i >=3:
                if self._raise_error:
                    self._raise_error = False
                    raise JiuWenBaseException(-1, "mock error")

    async def collect(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        iter = inputs.get("array")
        results = []
        async for item in iter:
            results.append(item)
        return {"collect": results}

async def test_workflow_stream_with_exception():
    workflow = Workflow()
    workflow.set_start_comp("start", Start(), inputs_schema={"array": "${user_inputs.array}"})
    workflow.add_workflow_comp("stream", StreamNodeWithException(), inputs_schema={"array": "${start.array}"})
    workflow.add_workflow_comp("transform", StreamNodeWithException(), stream_inputs_schema={"array": "${stream.array}"})
    workflow.add_workflow_comp("collect", StreamNodeWithException(), stream_inputs_schema={"array": "${transform.array}"})
    workflow.set_end_comp("end", End(), inputs_schema={"result": "${collect.collect}"})
    workflow.add_connection("start", "stream")
    workflow.add_stream_connection("stream", "transform")
    workflow.add_stream_connection("transform", "collect")
    workflow.add_connection("collect", "end")
    with pytest.raises(JiuWenBaseException) as e:
        await workflow.invoke(inputs={"user_inputs": {"array": [1, 2, 3, 4, 5, 6, 7]}},
                                       runtime=WorkflowRuntime())
    assert e.value.error_code == StatusCode.COMPONENT_EXECUTE_ERROR.code
    assert e.value.message == StatusCode.COMPONENT_EXECUTE_ERROR.errmsg.format(node_id="transform", ability="transform", error=JiuWenBaseException(-1, "mock error"))

    result = await workflow.invoke(inputs={"user_inputs": {"array": [1, 2, 3, 4, 5, 6, 7]}}, runtime=WorkflowRuntime())
    assert result.result == {'responseContent': '', 'output': {'result': [1, 2, 3, 4, 5, 6, 7]}}

