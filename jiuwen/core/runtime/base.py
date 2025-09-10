from typing import AsyncIterator, TypeVar

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.graph.executable import Executable
from jiuwen.core.runtime.runtime import Runtime, BaseRuntime, NodeRuntime

Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", contravariant=True)


class ComponentExecutable(Executable):

    async def on_invoke(self, inputs: Input, context: BaseRuntime) -> Output:
        if not isinstance(context, NodeRuntime):
            raise JiuWenBaseException(-1, "runtime should be NodeRuntime instance")
        return await self.invoke(inputs, Runtime(context))

    async def on_stream(self, inputs: Input, context: BaseRuntime) -> AsyncIterator[Output]:
        if not isinstance(context, NodeRuntime):
            raise JiuWenBaseException(-1, "runtime should be NodeRuntime instance")
        async for value in self.stream(inputs, Runtime(context)):
            yield value

    async def on_collect(self, inputs: AsyncIterator[Input], context: BaseRuntime) -> Output:
        if not isinstance(context, NodeRuntime):
            raise JiuWenBaseException(-1, "runtime should be NodeRuntime instance")
        return await self.collect(inputs, Runtime(context))

    async def on_transform(self, inputs: AsyncIterator[Input], context: BaseRuntime) -> AsyncIterator[Output]:
        if not isinstance(context, NodeRuntime):
            raise JiuWenBaseException(-1, "runtime should be NodeRuntime instance")
        async for value in self.transform(inputs, Runtime(context)):
            yield value

    async def invoke(self, inputs: Input, runtime: Runtime) -> Output:
        raise JiuWenBaseException(-1, "Invoke is not supported")

    async def stream(self, inputs: Input, runtime: Runtime) -> AsyncIterator[Output]:
        raise JiuWenBaseException(-1, "Stream is not supported")

    async def collect(self, inputs: AsyncIterator[Input], runtime: Runtime) -> Output:
        raise JiuWenBaseException(-1, "Collect is not supported")

    async def transform(self, inputs: AsyncIterator[Input], runtime: Runtime) -> AsyncIterator[Output]:
        raise JiuWenBaseException(-1, "Transform is not supported")
