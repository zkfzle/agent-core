import asyncio
import random

from jiuwen.core.context.context import Context
from jiuwen.core.graph.executable import Input, Output
from jiuwen.core.workflow.base import Workflow
from tests.unit_tests.workflow.test_mock_node import MockNodeBase


class StreamNodeWithTracer(MockNodeBase):
    def __init__(self, node_id: str, datas: list[dict]):
        super().__init__(node_id)
        self._node_id = node_id
        self._datas: list[dict] = datas

    async def invoke(self, inputs: Input, context: Context) -> Output:
        context.state().set_outputs(inputs)
        try:
            await context.tracer().trigger("tracer_workflow", "on_invoke", invoke_id=context.executable_id(),
                                         parent_node_id=context.parent_id(),
                                         on_invoke_data={"on_invoke_data": "mock with" + str(inputs)})
            context.state().update_trace(context.tracer().get_workflow_span(context.executable_id(),
                                                                                    context.parent_id()))

            # 运行时操作

        except Exception as e:
            await context.tracer().trigger("tracer_workflow", "on_invoke", invoke_id=context.executable_id(),
                                         parent_node_id=context.parent_id(),
                                         error=e)
            context.state().update_trace(context.tracer().get_workflow_span(context.executable_id(),
                                                                        context.parent_id()))
            raise e

        await asyncio.sleep(random.randint(0, 5))
        for data in self._datas:
            await asyncio.sleep(1)
            await context.stream_writer_manager().get_custom_writer().write(data)
        print("StreamNode: output = " + str(inputs))
        return inputs

