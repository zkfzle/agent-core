#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
import asyncio
from typing import Self, Dict, Any, Union, AsyncIterator

from pydantic import BaseModel

from jiuwen.core.common.constants.constant import INTERACTION
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.common.logging.base import logger
from jiuwen.core.component.base import WorkflowComponent
from jiuwen.core.component.end_comp import End
from jiuwen.core.component.start_comp import Start
from jiuwen.core.context.config import CompIOConfig, Transformer
from jiuwen.core.context.context import Context
from jiuwen.core.context.mq_manager import MessageQueueManager
from jiuwen.core.graph.base import Graph, Router, INPUTS_KEY, CONFIG_KEY, ExecutableGraph
from jiuwen.core.graph.executable import Executable, Input, Output
from jiuwen.core.stream.base import StreamMode, BaseStreamMode
from jiuwen.core.stream.emitter import StreamEmitter
from jiuwen.core.stream.manager import StreamWriterManager
from jiuwen.core.stream.writer import OutputSchema
from jiuwen.core.stream_actor.base import StreamActor
from jiuwen.core.tracer.tracer import Tracer
from jiuwen.core.workflow.workflow_config import WorkflowConfig, ComponentAbility


class WorkflowOutput(BaseModel):
    result: str


class WorkflowChunk(BaseModel):
    chunk_id: str
    payload: str
    metadata: Dict[str, Any]
    is_final: bool


class BaseWorkFlow:
    def __init__(self, workflow_config: WorkflowConfig, new_graph: Graph):
        self._graph = new_graph
        self._workflow_config = workflow_config
        self._stream_actor = StreamActor()

    def config(self):
        return self._workflow_config

    def add_workflow_comp(
            self,
            comp_id: str,
            workflow_comp: Union[Executable, WorkflowComponent],
            *,
            wait_for_all: bool = False,
            inputs_schema: dict = None,
            outputs_schema: dict = None,
            inputs_transformer: Transformer = None,
            outputs_transformer: Transformer = None,
            stream_inputs_schema: dict = None,
            stream_outputs_schema: dict = None,
            stream_inputs_transformer: Transformer = None,
            stream_outputs_transformer: Transformer = None,
            comp_ability: list[ComponentAbility] = None,
            response_mode: str = None
    ) -> Self:
        if not isinstance(workflow_comp, WorkflowComponent):
            workflow_comp = self._convert_to_component(workflow_comp)
        workflow_comp.add_component(graph=self._graph, node_id=comp_id, wait_for_all=wait_for_all)
        self._workflow_config.comp_configs[comp_id] = CompIOConfig(inputs_schema=inputs_schema,
                                                                   outputs_schema=outputs_schema,
                                                                   inputs_transformer=inputs_transformer,
                                                                   outputs_transformer=outputs_transformer)
        self._workflow_config.comp_stream_configs[comp_id] = CompIOConfig(inputs_schema=stream_inputs_schema,
                                                                          outputs_schema=stream_outputs_schema,
                                                                          inputs_transformer=stream_inputs_transformer,
                                                                          outputs_transformer=stream_outputs_transformer)
        self._workflow_config.comp_abilities[
            comp_id] = comp_ability if comp_ability is not None else [ComponentAbility.INVOKE]
        for ability in self._workflow_config.comp_abilities[comp_id]:
            if ability in [ComponentAbility.STREAM, ComponentAbility.TRANSFORM, ComponentAbility.COLLECT]:
                if not wait_for_all:
                    raise JiuWenBaseException(-1, "stream components need to wait for all")
        if response_mode is not None:
            if "streaming" == response_mode:
                self._workflow_config.comp_abilities[
                    comp_id] =  [ComponentAbility.STREAM, ComponentAbility.TRANSFORM]
            else:
                self._workflow_config.comp_abilities[comp_id] = [ComponentAbility.INVOKE]
        return self

    def start_comp(
            self,
            start_comp_id: str,
    ) -> Self:
        self._graph.start_node(start_comp_id)
        return self

    def end_comp(
            self,
            end_comp_id: str,
    ) -> Self:
        self._graph.end_node(end_comp_id)
        return self

    def add_connection(self, src_comp_id: str, target_comp_id: str) -> Self:
        self._graph.add_edge(src_comp_id, target_comp_id)
        return self

    def add_stream_connection(self, src_comp_id: str, target_comp_id: str) -> Self:
        self._graph.add_edge(src_comp_id, target_comp_id)
        stream_executables = self._graph.get_nodes()
        self._stream_actor.add_stream_consumer(stream_executables[target_comp_id], target_comp_id)
        if target_comp_id not in self._workflow_config.stream_edges:
            self._workflow_config.stream_edges[src_comp_id] = [target_comp_id]
        else:
            self._workflow_config.stream_edges[src_comp_id].append(target_comp_id)
        return self

    def add_conditional_connection(self, src_comp_id: str, router: Router) -> Self:
        self._graph.add_conditional_edges(source_node_id=src_comp_id, router=router)
        return self

    def compile(self, context: Context) -> ExecutableGraph:
        context.config().set_workflow_config(self._workflow_config)
        return self._graph.compile(context)


class Workflow(BaseWorkFlow):
    def __init__(self, workflow_config: WorkflowConfig, graph: Graph):
        super().__init__(workflow_config, graph)
        self._end_comp_id: str = ""

    def set_start_comp(
            self,
            start_comp_id: str,
            component: Start,
            inputs_schema: dict = None,
            outputs_schema: dict = None,
            inputs_transformer: Transformer = None,
            outputs_transformer: Transformer = None
    ) -> Self:
        self.add_workflow_comp(start_comp_id, component, wait_for_all=False, inputs_schema=inputs_schema,
                               outputs_schema=outputs_schema,
                               inputs_transformer=inputs_transformer,
                               outputs_transformer=outputs_transformer)
        self.start_comp(start_comp_id)
        return self

    def set_end_comp(
            self,
            end_comp_id: str,
            component: End,
            inputs_schema: dict = None,
            outputs_schema: dict = None,
            inputs_transformer: Transformer = None,
            outputs_transformer: Transformer = None,
            stream_inputs_schema: dict = None,
            stream_outputs_schema: dict = None,
            stream_inputs_transformer: Transformer = None,
            stream_outputs_transformer: Transformer = None,
            response_mode: str = None,
    ) -> Self:
        self.add_workflow_comp(end_comp_id, component, wait_for_all=False, inputs_schema=inputs_schema,
                               outputs_schema=outputs_schema,
                               inputs_transformer=inputs_transformer,
                               outputs_transformer=outputs_transformer,
                               stream_inputs_schema=stream_inputs_schema,
                               stream_outputs_schema=stream_outputs_schema,
                               stream_inputs_transformer=stream_inputs_transformer,
                               stream_outputs_transformer=stream_outputs_transformer,
                               response_mode=response_mode
                               )
        self.end_comp(end_comp_id)
        self._end_comp_id = end_comp_id
        return self

    async def sub_invoke(self, inputs: Input, context: Context, config: Any = None) -> Output:
        logger.info("begin to sub_invoke, input=%s", inputs)
        context.config().set_workflow_config(self._workflow_config)
        compiled_graph = self._graph.compile(context)
        await compiled_graph.invoke({INPUTS_KEY: inputs, CONFIG_KEY: config}, context)
        results = context.state().get_inputs(self._end_comp_id)
        logger.info("end to sub_invoke, results=%s", results)
        return results

    async def invoke(self, inputs: Input, context: Context) -> Output:
        logger.info("begin to invoke, input=%s", inputs)
        chunks = []
        async for chunk in self.stream(inputs, context, stream_modes=[BaseStreamMode.OUTPUT]):
            chunks.append(chunk)

        is_interaction = False
        for chunk in chunks:
            if isinstance(chunk, OutputSchema) and chunk.type == INTERACTION:
                is_interaction = True
                break
        results = [chunk.model_dump() for chunk in chunks] if is_interaction else context.state().get_outputs(
            self._end_comp_id)
        logger.info("end to invoke, results=%s", results)
        return results

    async def stream(
            self,
            inputs: Input,
            context: Context,
            stream_modes: list[StreamMode] = None
    ) -> AsyncIterator[WorkflowChunk]:
        mq_manager = MessageQueueManager(self._workflow_config.stream_edges, self._workflow_config.comp_abilities,
                                         False)
        context.set_queue_manager(mq_manager)
        context.set_stream_writer_manager(StreamWriterManager(stream_emitter=StreamEmitter(), modes=stream_modes))
        if context.tracer() is None and (stream_modes is None or BaseStreamMode.TRACE in stream_modes):
            tracer = Tracer()
            tracer.init(context.stream_writer_manager(), context.callback_manager())
            context.set_tracer(tracer)
        compiled_graph = self.compile(context)
        self._stream_actor.init(context)
        async def stream_process():
            try:
                await self._stream_actor.run()
                await compiled_graph.invoke({INPUTS_KEY: inputs, CONFIG_KEY: None}, context)
            finally:
                await context.stream_writer_manager().stream_emitter.close()

        task = asyncio.create_task(stream_process())
        async for chunk in context.stream_writer_manager().stream_output():
            yield chunk

        try:
            await task
        except Exception:
            raise

    def _convert_to_component(self, executable: Executable) -> WorkflowComponent:
        pass
