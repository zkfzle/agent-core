import asyncio
import os
import unittest
from typing import Any, AsyncIterator

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.exception.status_code import StatusCode
from jiuwen.core.component.base import WorkflowComponent

from jiuwen.core.component.end_comp import End

from jiuwen.core.component.start_comp import Start
from jiuwen.core.context_engine.base import Context
from jiuwen.core.graph.executable import Executable
from jiuwen.core.runtime.base import ComponentExecutable
from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.runtime.workflow import WorkflowRuntime
from jiuwen.core.stream.base import StreamMode, BaseStreamMode
from jiuwen.core.workflow.base import Workflow, WorkflowOutput, WorkflowChunk
from jiuwen.core.workflow.workflow_config import WorkflowConfig

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
        yield inputs

    def to_executable(self) -> Executable:
        return self


class TestComponentStream(unittest.TestCase):
    def test_no_stream_called(self):
        with self.assertRaises(JiuWenBaseException) as error:
            flow = Workflow(WorkflowConfig(stream_timeout=3))
            flow.set_start_comp("start", Start())
            flow.set_end_comp("end", End(), inputs_schema={}, response_mode="streaming")
            flow.add_workflow_comp("stream", MockStreamNode(), inputs_schema={})
            flow.add_connection("start", "stream")
            flow.add_stream_connection("stream", "end")

            async def run_workflow():
                return await flow.invoke(inputs={"a": "生成markdown回复"}, runtime=WorkflowRuntime())

            asyncio.get_event_loop().run_until_complete(run_workflow())

        assert error.exception.error_code == StatusCode.STREAM_FRAME_TIMEOUT_FAILED.code
        with self.assertRaises(JiuWenBaseException) as error:
            results = []

            async def run_workflow():
                async for chunk in flow.stream(inputs={"a": "生成markdown回复"}, runtime=WorkflowRuntime(),
                                               stream_modes=[BaseStreamMode.OUTPUT]):
                    results.append(chunk)

            asyncio.get_event_loop().run_until_complete(run_workflow())
            for result in results:
                print(result)
        assert error.exception.error_code == StatusCode.STREAM_FRAME_TIMEOUT_FAILED.code



