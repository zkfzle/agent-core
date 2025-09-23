import asyncio
from functools import wraps
from types import MethodType
from jiuwen.core.tracer.data import InvokeType


def decrate_model_with_trace(model, tracer, agent_span):
    model.invoke = MethodType(trace(model.invoke, tracer, agent_span, InvokeType.LLM), model)
    model.ainvoke = MethodType(async_trace(model.ainvoke, tracer, agent_span, InvokeType.LLM), model)
    model.stream = MethodType(trace_stream(model.stream, tracer, agent_span, InvokeType.LLM), model)


def decrate_tool_with_trace(tool, tracer, agent_span):
    tool.invoke = MethodType(trace(tool.invoke, tracer, agent_span, InvokeType.PLUGIN), tool)
    tool.ainvoke = MethodType(async_trace(tool.ainvoke, tracer, agent_span, InvokeType.PLUGIN), tool)


def decrate_workflow_with_trace(workflow, tracer, agent_span):
    workflow.invoke = MethodType(async_trace(workflow.invoke, tracer, agent_span, InvokeType.WORKFLOW), workflow)
    workflow.stream = MethodType(async_trace_stream(workflow.stream, tracer, agent_span, InvokeType.WORKFLOW), workflow)


def trace(func, tracer, agent_span, invoke_type: InvokeType):
    @wraps(func)
    def decorator(*args, **kwargs):
        try:
            span = tracer.tracer_agent_span_manager.create_agent_span(agent_span)
            asyncio.get_event_loop().run_until_complete(
                tracer.trigger("tracer_agent", "on_" + invoke_type.value + "_start", span=span,
                               inputs={"inputs": kwargs.get("inputs", {})},
                               instance_info={"class_name": func}))

            args = args[1:]
            result = func(*args, **kwargs)
            asyncio.get_event_loop().run_until_complete(
                tracer.trigger("tracer_agent", "on_" + invoke_type.value + "_end", span=span,
                               outputs={"outputs": result}))
            return result
        except Exception as error:
            asyncio.get_event_loop().run_until_complete(
                tracer.trigger("trace_agent", "on_" + invoke_type.value + "_error", span=span, error=error))
            raise error

    return decorator


def async_trace(func, tracer, agent_span, invoke_type: InvokeType):
    @wraps(func)
    async def decorator(*args, **kwargs):
        try:
            span = tracer.tracer_agent_span_manager.create_agent_span(agent_span)
            asyncio.get_event_loop().run_until_complete(
                tracer.trigger("tracer_agent", "on_" + invoke_type.value + "_start", span=span,
                               inputs={"inputs": kwargs.get("inputs", {})},
                               instance_info={"class_name": func}))

            args = args[1:]
            result = await func(*args, **kwargs)
            asyncio.get_event_loop().run_until_complete(
                tracer.trigger("tracer_agent", "on_" + invoke_type.value + "_end", span=span,
                               outputs={"outputs": result}))
            return result
        except Exception as error:
            await tracer.trigger("trace_agent", "on_" + invoke_type.value + "_error", span=span, error=error)
            raise error

    return decorator


def trace_stream(func, tracer, agent_span, invoke_type: InvokeType):
    @wraps(func)
    def decorator(*args, **kwargs):
        try:
            span = tracer.tracer_agent_span_manager.create_agent_span(agent_span)
            asyncio.get_event_loop().run_until_complete(
                tracer.trigger("tracer_agent", "on_" + invoke_type.value + "_start", span=span,
                               inputs={"inputs": kwargs.get("inputs", {})},
                               instance_info={"class_name": func}))
            result = func(*args, **kwargs)
            results = []
            if hasattr(result, "__iter__") or hasattr(result, "__getitem__"):
                for item in result:
                    yield item
                    results.append(item)
            else:
                results.append(result)
            asyncio.get_event_loop().run_until_complete(
                tracer.trigger("tracer_agent", "on_" + invoke_type.value + "_end", span=span,
                               outputs={"outputs": result}))
        except Exception as error:
            asyncio.get_event_loop().run_until_complete(
                tracer.trigger("trace_agent", "on_" + invoke_type.value + "_error", span=span, error=error))
            raise error

    return decorator


def async_trace_stream(func, tracer, agent_span, invoke_type: InvokeType):
    @wraps(func)
    async def decorator(*args, **kwargs):
        try:
            span = tracer.tracer_agent_span_manager.create_agent_span(agent_span)
            await tracer.trigger("tracer_agent", "on_" + invoke_type.value + "_start", span=span,
                                 inputs={"inputs": kwargs.get("inputs", {})},
                                 instance_info={"class_name": func})
            result = await func(*args, **kwargs)
            results = []
            if hasattr(result, "__aiter__") or hasattr(result, "__anext__"):
                async for item in result:
                    yield item
                    results.append(item)
            else:
                results.append(result)
            await tracer.trigger("tracer_agent", "on_" + invoke_type.value + "_end", span=span,
                                 outputs={"outputs": result})
        except Exception as error:
            await tracer.trigger("trace_agent", "on_" + invoke_type.value + "_error", span=span, error=error)
            raise error

    return decorator
