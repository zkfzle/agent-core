#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import inspect
from typing import AsyncIterator, TypeVar

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.context_engine.base import Context
from openjiuwen.core.graph.executable import Executable
from openjiuwen.core.runtime.runtime import Runtime, BaseRuntime
from openjiuwen.core.runtime.workflow import NodeRuntime
from openjiuwen.core.runtime.wrapper import WrappedNodeRuntime

Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", contravariant=True)


class ComponentExecutable(Executable):

    async def on_invoke(self, inputs: Input, runtime: BaseRuntime) -> Output:
        if not isinstance(runtime, NodeRuntime):
            raise JiuWenBaseException(-1, "runtime should be NodeRuntime instance")
        # Check if the derived class has implemented the invoke method
        current_class = type(self)
        
        # Check if this class has overridden the invoke method
        # Compare the invoke method of the current class with ComponentExecutable's invoke
        if (hasattr(current_class, 'invoke') and 
            current_class.invoke is ComponentExecutable.invoke):
            raise JiuWenBaseException(-1, 
                f"Component ability 'INVOKE' is registered but 'invoke' method is not implemented. "
                f"Please implement the 'invoke' method in your component class '{type(self).__name__}'.")
        
        return await self.invoke(inputs, WrappedNodeRuntime(runtime), runtime.context())

    async def on_stream(self, inputs: Input, runtime: BaseRuntime) -> AsyncIterator[Output]:
        if not isinstance(runtime, NodeRuntime):
            raise JiuWenBaseException(-1, "runtime should be NodeRuntime instance")
        # Check if the derived class has implemented the stream method
        current_class = type(self)
        
        # Check if this class has overridden the stream method
        # Compare the stream method of the current class with ComponentExecutable's stream
        if (hasattr(current_class, 'stream') and 
            current_class.stream is ComponentExecutable.stream):
            raise JiuWenBaseException(-1, 
                f"Component ability 'STREAM' is registered but 'stream' method is not implemented. "
                f"Please implement the 'stream' method in your component class '{type(self).__name__}'.")
        
        async for value in self.stream(inputs, WrappedNodeRuntime(runtime), runtime.context()):
            yield value

    async def on_collect(self, inputs: Input, runtime: BaseRuntime) -> Output:
        if not isinstance(runtime, NodeRuntime):
            raise JiuWenBaseException(-1, "runtime should be NodeRuntime instance")
        # Check if the derived class has implemented the collect method
        current_class = type(self)
        
        # Check if this class has overridden the collect method
        # Compare the collect method of the current class with ComponentExecutable's collect
        if (hasattr(current_class, 'collect') and 
            current_class.collect is ComponentExecutable.collect):
            raise JiuWenBaseException(-1, 
                f"Component ability 'COLLECT' is registered but 'collect' method is not implemented. "
                f"Please implement the 'collect' method in your component class '{type(self).__name__}'.")
        
        return await self.collect(inputs, WrappedNodeRuntime(runtime), runtime.context())

    async def on_transform(self, inputs: Input, runtime: BaseRuntime) -> AsyncIterator[Output]:
        if not isinstance(runtime, NodeRuntime):
            raise JiuWenBaseException(-1, "runtime should be NodeRuntime instance")
        # Check if the derived class has implemented the transform method
        current_class = type(self)
        
        # Check if this class has overridden the transform method
        # Compare the transform method of the current class with ComponentExecutable's transform
        if (hasattr(current_class, 'transform') and 
            current_class.transform is ComponentExecutable.transform):
            raise JiuWenBaseException(-1, 
                f"Component ability 'TRANSFORM' is registered but 'transform' method is not implemented. "
                f"Please implement the 'transform' method in your component class '{type(self).__name__}'.")
        
        # Call the actual transform method
        async for value in self.transform(inputs, WrappedNodeRuntime(runtime), runtime.context()):
            yield value

    async def invoke(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        raise JiuWenBaseException(-1, "Invoke is not supported")

    async def stream(self, inputs: Input, runtime: Runtime, context: Context) -> AsyncIterator[Output]:
        raise JiuWenBaseException(-1, "Stream is not supported")

    async def collect(self, inputs: Input, runtime: Runtime, context: Context) -> Output:
        raise JiuWenBaseException(-1, "Collect is not supported")

    async def transform(self, inputs: Input, runtime: Runtime, context: Context) -> AsyncIterator[Output]:
        raise JiuWenBaseException(-1, "Transform is not supported")
