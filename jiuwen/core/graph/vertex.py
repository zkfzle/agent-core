#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
import asyncio
from typing import Any, Optional, AsyncIterator

from jiuwen.core.common.constants.component import SUB_WORKFLOW_COMPONENT
from jiuwen.core.common.constants.constant import INTERACTIVE_INPUT, END_NODE_STREAM, INPUTS_KEY, CONFIG_KEY
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.logging import logger
from jiuwen.core.component.end_comp import End
from jiuwen.core.graph.atomic_node import AsyncAtomicNode
from jiuwen.core.graph.executable import Executable, Output
from jiuwen.core.graph.graph_state import GraphState
from jiuwen.core.runtime.runtime import BaseRuntime, NodeRuntime
from jiuwen.core.runtime.utils import get_by_schema
from jiuwen.core.tracer.workflow_tracer import trace_inputs, trace_outputs
from jiuwen.core.workflow.workflow_config import ComponentAbility


class Vertex(AsyncAtomicNode):
    def __init__(self, node_id: str, executable: Executable = None):
        self._node_id = node_id
        self._executable = executable
        self._runtime: NodeRuntime = None
        # if stream_call is available, call should wait for it
        self._stream_done = asyncio.Event()
        self._stream_called = False

    def init(self, runtime: BaseRuntime) -> bool:
        self._runtime = NodeRuntime(runtime, self._node_id)
        return True

    async def _run_executable(self, ability: ComponentAbility, is_subgraph: bool = False, config: Any = None):
        if ability == ComponentAbility.INVOKE:
            batch_inputs = await self._pre_invoke()
            if is_subgraph:
                batch_inputs = {INPUTS_KEY: batch_inputs, CONFIG_KEY: config}
            results = await self._executable.on_invoke(batch_inputs, runtime=self._runtime)
            await self._post_invoke(results)
        elif ability == ComponentAbility.STREAM:
            batch_inputs = await self._pre_invoke()
            if is_subgraph:
                batch_inputs = {INPUTS_KEY: batch_inputs, CONFIG_KEY: config}
            result_iter = self._executable.on_stream(batch_inputs, runtime=self._runtime)
            await self._post_stream(result_iter)
        elif ability == ComponentAbility.COLLECT:
            collect_iter = self._pre_stream(ability)
            batch_output = await self._executable.on_collect(collect_iter, self._runtime)
            await self._post_invoke(batch_output)
        elif ability == ComponentAbility.TRANSFORM:
            transform_iter = self._pre_stream(ability)
            output_iter = self._executable.on_transform(transform_iter, self._runtime)
            await self._post_stream(output_iter)
        else:
            logger.error(f"error ComponentAbility: {ability.name}")

    async def __call__(self, state: GraphState, config) -> Output:
        if self._executable.post_commit():
            await self.atomic_invoke(config=config, runtime=self._runtime)
        else:
            await self.call(config)
        return {"source_node_id": [self._node_id]}

    async def _atomic_invoke(self, **kwargs) -> Any:
        return await self.call(kwargs.get("config", None))

    async def _pre_invoke(self) -> Optional[dict]:
        inputs_transformer = self._runtime.config().get_input_transformer(self._node_id)
        if inputs_transformer is None:
            inputs_schema = self._runtime.config().get_inputs_schema(self._node_id)
            inputs = self._runtime.state().get_inputs(inputs_schema)
        else:
            inputs = self._runtime.state().get_inputs_by_transformer(inputs_transformer)
        if self._runtime.tracer() is not None:
            await self.__trace_inputs__(inputs)
        return inputs

    async def _post_invoke(self, results: Optional[dict]) -> Any:
        output_transformer = self._runtime.config().get_output_transformer(self._node_id)
        if output_transformer is None:
            output_schema = self._runtime.config().get_outputs_schema(self._node_id)
            results = get_by_schema(output_schema, results) if output_schema else results
        else:
            results = output_transformer(results)
        self._runtime.state().set_outputs(results)
        if self._runtime.tracer() is not None:
            await self.__trace_outputs__(results)

        self.__clear_interactive__()
        return results

    async def _pre_stream(self, ability: ComponentAbility) -> AsyncIterator[dict]:
        queue_manager = self._runtime.queue_manager()
        workflow_config = self._runtime.config().get_workflow_config()
        inputs_transformer = workflow_config.comp_stream_configs[self._node_id].inputs_transformer
        inputs_schema = workflow_config.comp_stream_configs[self._node_id].inputs_schema
        async for message in queue_manager.consume(self._node_id, ability):
            # message 是{id: content}
            if inputs_transformer is None:
                inputs = queue_manager.stream_transform.get_by_default_transformer(message, inputs_schema) \
                    if inputs_schema else message
            else:
                inputs = queue_manager.stream_transform.get_by_defined_transformer(message, inputs_transformer)
            yield inputs

    async def _post_stream(self, results_iter: AsyncIterator) -> None:
        queue_manager = self._runtime.queue_manager()
        workflow_config = self._runtime.config().get_workflow_config()
        output_transformer = workflow_config.comp_stream_configs[self._node_id].outputs_transformer
        output_schema = workflow_config.comp_stream_configs[self._node_id].outputs_schema
        end_stream_index = 0
        async for chunk in results_iter:
            if output_transformer is None:
                message = queue_manager.stream_transform.get_by_default_transformer(chunk, output_schema) \
                    if output_schema else chunk
            else:
                message = queue_manager.stream_transform.get_by_defined_transformer(chunk, output_transformer)
            await self._process_chunk(end_stream_index, message)
        await queue_manager.end_message(self._node_id)

    async def _process_chunk(self, end_stream_index: int, message: Any) -> None:
        end_node = isinstance(self._executable, End)
        sub_graph = self._runtime.parent_id() != ''
        if end_node and not sub_graph:
            message_stream_data = {
                "type": END_NODE_STREAM,
                "index": ++end_stream_index,
                "payload": message
            }
            await self._runtime.stream_writer_manager().get_output_writer().write(message_stream_data)
        elif end_node and sub_graph:
            await self._runtime.queue_manager().sub_workflow_stream.send(message)
        else:
            await self._runtime.queue_manager().produce(self._node_id, message)

    def __clear_interactive__(self) -> None:
        if self._runtime.state().get(INTERACTIVE_INPUT):
            self._runtime.state().update({INTERACTIVE_INPUT: None})

    async def __trace_inputs__(self, inputs: Optional[dict]) -> None:
        if self._executable.skip_trace():
            return
        # TODO 组件信息
        await trace_inputs(self._runtime, inputs)

        if self._executable.component_type() == SUB_WORKFLOW_COMPONENT:
            self._runtime.tracer().register_workflow_span_manager(self._runtime.executable_id())

    async def call(self, config: Any = None):
        if self._runtime is None or self._executable is None:
            raise JiuWenBaseException(1, "vertex is not initialized, node is is " + self._node_id)

        is_subgraph = self._executable.graph_invoker()

        try:
            workflow_config = self._runtime.config().get_workflow_config()
            component_ability = workflow_config.comp_abilities.get(self._node_id)
            component_ability = component_ability if component_ability else [ComponentAbility.INVOKE]
            call_ability = [ability for ability in component_ability if
                            ability in [ComponentAbility.INVOKE, ComponentAbility.STREAM]]
            for ability in call_ability:
                await self._run_executable(ability, is_subgraph, config)

        except JiuWenBaseException as e:
            raise JiuWenBaseException(e.error_code, "failed to invoke, caused by " + e.message)

        # 仅当 stream_call 被调用时才等待
        if self._stream_called:
            await self._stream_done.wait()
        logger.debug("node [%s] call finished", self._node_id)

    async def stream_call(self):
        self._stream_called = True  # 标记 stream_call 已被调用
        self._stream_done.clear()  # 清除之前的完成状态

        if self._runtime is None or self._runtime.queue_manager() is None:
            raise JiuWenBaseException(1, "queue manager is not initialized")

        try:
            workflow_config = self._runtime.config().get_workflow_config()
            component_ability = workflow_config.comp_abilities.get(self._node_id)
            call_ability = [ability for ability in component_ability if
                            ability in [ComponentAbility.COLLECT, ComponentAbility.TRANSFORM]]
            for ability in call_ability:
                await self._run_executable(ability)
        except JiuWenBaseException as e:
            raise JiuWenBaseException(e.error_code, "failed to stream, caused by " + e.message)
        finally:
            self._stream_done.set()  # 标记完成
            logger.info("end to stream call, node %s", self._node_id)

    async def __trace_outputs__(self, outputs: Optional[dict] = None) -> None:
        if self._executable.skip_trace():
            return
        await trace_outputs(self._runtime, outputs)
