from typing import AsyncIterator

import pytest

from openjiuwen.core.common.constants.constant import END_NODE_STREAM
from openjiuwen.core.component.base import WorkflowComponent
from openjiuwen.core.component.end_comp import End
from openjiuwen.core.component.start_comp import Start
from openjiuwen.core.context_engine.base import Context
from openjiuwen.core.runtime.base import ComponentExecutable, Input, Output
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.runtime.workflow import WorkflowRuntime
from openjiuwen.core.stream.base import OutputSchema, BaseStreamMode
from openjiuwen.core.workflow.base import Workflow
from openjiuwen.core.workflow.workflow_config import ComponentAbility
from tests.unit_tests.core.workflow.mock_nodes import Node1, StreamCompNode

pytestmark = pytest.mark.asyncio


class MockStreamCmp(WorkflowComponent, ComponentExecutable):
    async def stream(self, inputs: Input, runtime: Runtime, context: Context) -> AsyncIterator[Output]:
        yield inputs


async def test_simple_template_workflow():
    # flow1: start -> a -> end
    flow = Workflow()
    flow.set_start_comp("start", Start(
        {"inputs": [{"id": "query", "type": "String", "required": "true", "sourceType": "ref"}]}),
                        inputs_schema={
                            "query": "${a}",
                            "response_node": "${response_mode}",
                            "d": "${b}"})
    flow.add_workflow_comp("a", Node1("a"),
                           inputs_schema={
                               "aa": "${start.d}",
                               "ac": "${start.d}"})
    flow.set_end_comp("end", End({"responseTemplate": "hello:{{end_input}}"}),
                      inputs_schema={
                          "end_input": "${start.d}",
                          "response_mode": "${start.response_node}"})
    flow.add_connection("start", "a")
    flow.add_connection("a", "end")
    res = await flow.invoke({"a": 1, "b": "haha"}, WorkflowRuntime())
    assert res.result == {'output': {}, 'responseContent': 'hello:haha'}


async def test_end_invoke_template():
    flow = Workflow()
    flow.set_start_comp("s", Start(),
                        inputs_schema={"query": "${user_inputs.query}", "content": "${user_inputs.content}"})
    conf = {"responseTemplate": "渲染结果:{{param1}},{{param2}}"}
    flow.set_end_comp("e", End(conf=conf), inputs_schema={"param1": "${s.query}", "param2": "${s.content}"})
    flow.add_connection("s", "e")
    res = await flow.invoke({"user_inputs": {"query": "你好", "content": "杭州"}}, WorkflowRuntime())

    assert res.result == {'responseContent': '渲染结果:你好,杭州', 'output': {}}


async def test_end_invoke_no_template():
    flow = Workflow()
    flow.set_start_comp("s", Start(),
                        inputs_schema={"query": "${user_inputs.query}", "content": "${user_inputs.content}"})
    conf = {}
    flow.set_end_comp("e", End(conf=conf), inputs_schema={"param1": "${s.query}", "param2": "${s.content}"})
    flow.add_connection("s", "e")
    res = await flow.invoke({"user_inputs": {"query": "你好", "content": "杭州"}}, WorkflowRuntime())
    assert res.result == {'output': {'param1': '你好', 'param2': '杭州'}, 'responseContent': ''}


async def test_end_stream_template():
    flow = Workflow()
    flow.set_start_comp("s", Start(),
                        inputs_schema={"query": "${user_inputs.query}", "content": "${user_inputs.content}"})
    conf = {"responseTemplate": "渲染结果:{{param1}},{{param2}}"}
    flow.set_end_comp("e", End(conf=conf), inputs_schema={"param1": "${s.query}", "param2": "${s.content}"},
                      response_mode="streaming")
    flow.add_connection("s", "e")
    result = flow.stream(inputs={"user_inputs": {"query": "你好", "content": "杭州"}}, runtime=WorkflowRuntime(),
                         stream_modes=[BaseStreamMode.OUTPUT])

    expect_result = [OutputSchema(type=END_NODE_STREAM, index=0, payload={'answer': '渲染结果:'}),
                     OutputSchema(type=END_NODE_STREAM, index=1, payload={'answer': '你好'}),
                     OutputSchema(type=END_NODE_STREAM, index=2, payload={'answer': ','}),
                     OutputSchema(type=END_NODE_STREAM, index=3, payload={'answer': '杭州'})]

    streams = []
    async for stream in result:
        print(stream)
        streams.append(stream)

    assert streams == expect_result


async def test_end_stream_no_template():
    flow = Workflow()
    flow.set_start_comp("s", Start(),
                        inputs_schema={"query": "${user_inputs.query}", "content": "${user_inputs.content}"})
    conf = {}
    flow.set_end_comp("e", End(conf=conf), inputs_schema={"param1": "${s.query}", "param2": "${s.content}"},
                      response_mode="streaming")
    flow.add_connection("s", "e")
    result = flow.stream(inputs={"user_inputs": {"query": "你好", "content": "杭州"}}, runtime=WorkflowRuntime(),
                         stream_modes=[BaseStreamMode.OUTPUT])

    expect_result = [
        OutputSchema(type=END_NODE_STREAM, index=0, payload={'output': {'param1': '你好'}}),
        OutputSchema(type=END_NODE_STREAM, index=1, payload={'output': {'param2': '杭州'}}),
    ]

    streams = []
    async for stream in result:
        print(stream)
        streams.append(stream)

    assert streams == expect_result


async def test_end_transform():
    flow = Workflow()
    flow.set_start_comp("s", Start(),
                        inputs_schema={"query": "${user_inputs.query}", "content": "${user_inputs.content}"})
    flow.add_workflow_comp("n", MockStreamCmp(), inputs_schema={"param1": "${s.query}", "param2": "${s.content}"},
                           comp_ability=[ComponentAbility.STREAM], wait_for_all=True)
    conf = {"responseTemplate": "渲染结果:{{param1}},{{param2}}"}
    flow.set_end_comp("e", End(conf=conf), stream_inputs_schema={"param1": "${n.param1}", "param2": "${n.param2}"},
                      response_mode="streaming")
    flow.add_connection("s", "n")
    flow.add_stream_connection("n", "e")
    result = flow.stream(inputs={"user_inputs": {"query": "你好", "content": "杭州"}}, runtime=WorkflowRuntime(),
                         stream_modes=[BaseStreamMode.OUTPUT])
    expect_result = [OutputSchema(type='end node stream', index=0, payload={'answer': '渲染结果:'}),
                     OutputSchema(type='end node stream', index=1, payload={'answer': '你好'}),
                     OutputSchema(type='end node stream', index=2, payload={'answer': ','}),
                     OutputSchema(type='end node stream', index=3, payload={'answer': '杭州'})]

    streams = []
    async for stream in result:
        print(stream)
        streams.append(stream)
    print(streams)

    assert expect_result == streams


async def test_simple_output_schema_workflow():
    # flow1: start -> a -> end
    flow = Workflow()
    flow.set_start_comp("start", Start(
        {"inputs": [{"id": "query", "type": "String", "required": "true", "sourceType": "ref"}]}),
                        inputs_schema={
                            "query": "${a}",
                            "response_node": "${response_mode}",
                            "d": "${b}"})
    flow.add_workflow_comp("a", Node1("a"),
                           inputs_schema={
                               "aa": "${start.d}",
                               "ac": "${start.d}"})
    flow.set_end_comp("end", End(),
                      inputs_schema={
                          "end_input": "${start.d}",
                          "response_mode": "${start.response_node}"},
                      )
    flow.add_connection("start", "a")
    flow.add_connection("a", "end")
    res = await flow.invoke({"a": 1, "b": "haha"}, WorkflowRuntime())
    assert res.result == {'output': {'end_input': 'haha'}, 'responseContent': ''}


async def test_end_stream_workflow():
    flow = Workflow()
    start = Start({"inputs": [{"id": "query", "type": "String", "required": "true", "sourceType": "ref"}]})
    flow.set_start_comp("start", start,
                        inputs_schema={
                            "query": "${a}",
                            "response_node": "${response_mode}",
                            "d": "${a}"})

    flow.add_workflow_comp("a", StreamCompNode("a"), inputs_schema={"value": "${a}"},
                           comp_ability=[ComponentAbility.STREAM], wait_for_all=True)

    flow.set_end_comp("end", End({"responseTemplate": "hello:{{end_input}}"}),
                      inputs_schema={"end_input": "${start.d}"}, response_mode="streaming")
    flow.add_connection("start", "a")
    flow.add_stream_connection("a", "end")

    index = 0
    actual_chunks = []
    expect_chunks = [OutputSchema(type='end node stream', index=0, payload={'answer': 'hello:'}),
                     OutputSchema(type='end node stream', index=1, payload={'answer': 1})]
    async for chunk in flow.stream({"a": 1, "b": "haha"}, WorkflowRuntime(), stream_modes=[BaseStreamMode.OUTPUT]):
        actual_chunks.append(chunk)
        index += 1

    print(actual_chunks)
    assert expect_chunks == actual_chunks


async def test_end_batch_stream_workflow():
    flow = Workflow()
    start = Start({"inputs": [{"id": "query", "type": "String", "required": "true", "sourceType": "ref"}]})
    input_schema = {
        "query": "${a}",
        "response_node": "${response_mode}",
        "d": "${a}"
    }
    flow.set_start_comp("start", start, inputs_schema=input_schema)

    flow.add_workflow_comp("a", StreamCompNode("a"), inputs_schema={"value": "${a}"},
                           comp_ability=[ComponentAbility.STREAM], wait_for_all=True)

    flow.set_end_comp("end", End({"responseTemplate": "hello:{{value}}"}),
                      stream_inputs_schema={"value": "${a.value}"}, inputs_schema={"value": "${a.value}"},
                      response_mode="streaming")
    flow.add_connection("start", "a")
    flow.add_stream_connection("a", "end")

    expect_results = [
        OutputSchema(type='end node stream', index=0, payload={'answer': 'hello:'}),
        OutputSchema(type='end node stream', index=1, payload={'answer': 1}),
        OutputSchema(type='end node stream', index=2, payload={'answer': 2})]

    real_result = []
    async for chunk in flow.stream({"a": 1, "b": "haha"},
                                   WorkflowRuntime(), stream_modes=[BaseStreamMode.OUTPUT]):
        real_result.append(chunk)

    print(real_result)
    assert expect_results == real_result


class MockStreamNode(WorkflowComponent, ComponentExecutable):
    async def stream(self, inputs: Input, runtime: Runtime, context: Context) -> AsyncIterator[Output]:
        yield inputs


async def test_end_no_streaming_no_template():
    workflow = Workflow()
    workflow.set_start_comp("start", Start(), inputs_schema={"a": "${user_input.a}", "b": "${user_input.b}"})
    workflow.set_end_comp("end", End(), stream_inputs_schema={'a': '${stream.a}', 'b': '${stream.b}'})
    workflow.add_workflow_comp("stream", MockStreamNode(), inputs_schema={"a": "${start.a}", "b": "${start.b}"})
    workflow.add_connection("start", "stream")
    workflow.add_stream_connection("stream", "end")

    user_input = {'user_input': {'a': 1, 'b': 2}}
    result = await workflow.invoke(user_input, WorkflowRuntime())
    assert result.result == {'responseContent': '', 'collect_output': [{'a': 1}, {'b': 2}], 'output': None}
