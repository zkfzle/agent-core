#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import asyncio
import inspect
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import Self, Any, Union, AsyncIterator, List

from pydantic import BaseModel

from openjiuwen.core.common.configs.env_constant import WORKFLOW_DRAWABLE
from openjiuwen.core.common.constants.constant import INTERACTION
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.logging import logger
from openjiuwen.core.component.base import WorkflowComponent
from openjiuwen.core.component.branch_router import BranchRouter
from openjiuwen.core.component.end_comp import End
from openjiuwen.core.context_engine.base import Context
from openjiuwen.core.graph.base import Graph, Router, INPUTS_KEY, CONFIG_KEY, ExecutableGraph
from openjiuwen.core.graph.executable import Executable, Input, Output
from openjiuwen.core.runtime.constants import WORKFLOW_INVOKE_TIMEOUT, WORKFLOW_STREAM_TIMEOUT, \
    WORKFLOW_STREAM_FRAME_TIMEOUT
from openjiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from openjiuwen.core.runtime.runtime import BaseRuntime, ProxyRuntime
from openjiuwen.core.runtime.state import Transformer
from openjiuwen.core.runtime.utils import NESTED_PATH_SPLIT
from openjiuwen.core.runtime.workflow import WorkflowRuntime, SubWorkflowRuntime, NodeRuntime
from openjiuwen.core.runtime.wrapper import RouterRuntime
from openjiuwen.core.stream.base import StreamMode, BaseStreamMode, OutputSchema, CustomSchema, TraceSchema
from openjiuwen.core.stream.emitter import StreamEmitter
from openjiuwen.core.stream.manager import StreamWriterManager
from openjiuwen.core.stream_actor.base import StreamGraph
from openjiuwen.core.stream_actor.manager import ActorManager
from openjiuwen.core.tracer.tracer import Tracer
from openjiuwen.core.tracer.workflow_tracer import workflow_trace_inputs, workflow_trace_outputs
from openjiuwen.core.utils.llm.messages import ToolInfo, Function, Parameters
from openjiuwen.core.workflow.workflow_config import WorkflowConfig, ComponentAbility, \
    NodeSpec, CompIOConfig, WorkflowInputsSchema, WorkflowMetadata
from openjiuwen.graph.pregel.graph import PregelGraph
from openjiuwen.graph.visualization.drawable import Drawable


class WorkflowExecutionState(Enum):
    COMPLETED = "COMPLETED"
    INPUT_REQUIRED = "INPUT_REQUIRED"


class WorkflowOutput(BaseModel):
    result: Any
    state: WorkflowExecutionState


WorkflowChunk = Union[OutputSchema, CustomSchema, TraceSchema]


class BaseWorkFlow:
    def __init__(self, workflow_config: WorkflowConfig = None, new_graph: Graph = None):
        self._graph = new_graph if new_graph else PregelGraph()
        self._workflow_config = workflow_config if workflow_config else WorkflowConfig()
        if not self._workflow_config.metadata:
            self._workflow_config.metadata = WorkflowMetadata()
        self._workflow_spec = self._workflow_config.spec
        self._stream_actor = StreamGraph()
        self._runtime = ProxyRuntime()
        self._drawable = None
        if os.environ.get(WORKFLOW_DRAWABLE, "false").lower() == "true":
            self._drawable = Drawable()

    def config(self):
        return self._workflow_config

    def add_workflow_comp(
            self,
            comp_id: str,
            workflow_comp: Union[Executable, WorkflowComponent],
            *,
            wait_for_all: bool = None,
            inputs_schema: dict = None,
            outputs_schema: dict = None,
            inputs_transformer: Transformer = None,
            outputs_transformer: Transformer = None,
            stream_inputs_schema: dict = None,
            stream_outputs_schema: dict = None,
            stream_inputs_transformer: Transformer = None,
            stream_outputs_transformer: Transformer = None,
            comp_ability: list[ComponentAbility] = None
    ) -> Self:
        if not isinstance(workflow_comp, WorkflowComponent):
            workflow_comp = self._convert_to_component(workflow_comp)
        node_spec = NodeSpec(
            io_config=CompIOConfig(inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                   inputs_transformer=inputs_transformer, outputs_transformer=outputs_transformer),
            stream_io_configs=CompIOConfig(inputs_schema=stream_inputs_schema, outputs_schema=stream_outputs_schema,
                                           inputs_transformer=stream_inputs_transformer,
                                           outputs_transformer=stream_outputs_transformer),
            abilities=comp_ability if comp_ability is not None else [])

        for ability in node_spec.abilities:
            if ability in [ComponentAbility.STREAM, ComponentAbility.TRANSFORM, ComponentAbility.COLLECT]:
                if wait_for_all is None:
                    wait_for_all = True
                if not wait_for_all:
                    raise JiuWenBaseException(-1, "stream components need to wait for all")
        self._workflow_spec.comp_configs[comp_id] = node_spec
        if wait_for_all is None:
            wait_for_all = False
        workflow_comp.add_component(graph=self._graph, node_id=comp_id, wait_for_all=wait_for_all)

        if self._drawable:
            self._drawable.add_node(comp_id, workflow_comp)
        return self

    def start_comp(
            self,
            start_comp_id: str,
    ) -> Self:
        self._graph.start_node(start_comp_id)

        if self._drawable:
            self._drawable.set_start_node(start_comp_id)
        return self

    def end_comp(
            self,
            end_comp_id: str,
    ) -> Self:
        self._graph.end_node(end_comp_id)

        if self._drawable:
            self._drawable.set_end_node(end_comp_id)
        return self

    def add_connection(self, src_comp_id: str, target_comp_id: str) -> Self:
        self._graph.add_edge(src_comp_id, target_comp_id)
        if src_comp_id not in self._workflow_spec.edges:
            self._workflow_spec.edges[src_comp_id] = [target_comp_id]
        else:
            self._workflow_spec.edges[src_comp_id].append(target_comp_id)

        if self._drawable:
            self._drawable.add_edge(src_comp_id, target_comp_id)
        return self

    def add_stream_connection(self, src_comp_id: str, target_comp_id: str) -> Self:
        self._graph.add_edge(src_comp_id, target_comp_id)
        stream_executables = self._graph.get_nodes()
        self._stream_actor.add_stream_consumer(stream_executables[target_comp_id], target_comp_id)
        if src_comp_id not in self._workflow_spec.stream_edges:
            self._workflow_spec.stream_edges[src_comp_id] = [target_comp_id]
        else:
            self._workflow_spec.stream_edges[src_comp_id].append(target_comp_id)

        if self._drawable:
            self._drawable.add_edge(src_comp_id, target_comp_id, False, True)
        return self

    def add_conditional_connection(self, src_comp_id: str, router: Router) -> Self:
        if isinstance(router, BranchRouter):
            router.set_runtime(self._runtime)
            self._graph.add_conditional_edges(source_node_id=src_comp_id, router=router)
        else:
            def new_router(state):
                sig = inspect.signature(router)
                if 'runtime' in sig.parameters:
                    return router(runtime=RouterRuntime(self._runtime))
                else:
                    return router()

            self._graph.add_conditional_edges(source_node_id=src_comp_id, router=new_router)

        if self._drawable:
            self._drawable.add_edge(source=src_comp_id, conditional=True, data=router)
        return self

    def compile(self, runtime: BaseRuntime) -> ExecutableGraph:
        if isinstance(runtime, WorkflowRuntime):
            runtime.set_workflow_id(self._workflow_config.metadata.id)
        self._auto_complete_abilities()
        runtime.config().add_workflow_config(self._workflow_config.metadata.id, self._workflow_config)

        if isinstance(runtime, SubWorkflowRuntime):
            main_workflow_config = runtime.config().get_workflow_config(
                runtime.main_workflow_id())
            if main_workflow_config is None:
                raise JiuWenBaseException(StatusCode.SUB_WORKFLOW_COMPONENT_RUNNING_ERROR.code,
                                          StatusCode.SUB_WORKFLOW_COMPONENT_RUNNING_ERROR.errmsg.format(
                                              detail=f"main workflow config is not exit,"
                                                     f" main workflow_id={runtime.main_workflow_id()}"))
            if runtime.workflow_nesting_depth() > main_workflow_config.workflow_max_nesting_depth:
                raise JiuWenBaseException(StatusCode.SUB_WORKFLOW_COMPONENT_RUNNING_ERROR.code,
                                          StatusCode.SUB_WORKFLOW_COMPONENT_RUNNING_ERROR.errmsg.format(
                                              detail=f"workflow nesting hierarchy is too big, must <= "
                                                     f"{main_workflow_config.workflow_max_nesting_depth}"))
        self._runtime.set_runtime(runtime)
        return self._graph.compile(runtime)

    @property
    def drawable(self):
        return self._drawable

    def to_mermaid(self, title: str = "", expand_subgraph: int | bool = False, enable_animation: bool = False) -> str:
        if self._drawable:
            return self._drawable.to_mermaid(title=title, expand_subgraph=expand_subgraph,
                                             enable_animation=enable_animation)
        return ""

    def to_mermaid_png(self, title: str = "", expand_subgraph: int | bool = False) -> bytes:
        if self._drawable:
            return self._drawable.to_mermaid_png(title=title, expand_subgraph=expand_subgraph)
        return b""

    def to_mermaid_svg(self, title: str = "", expand_subgraph: int | bool = False) -> bytes:
        if self._drawable:
            return self._drawable.to_mermaid_svg(title=title, expand_subgraph=expand_subgraph)
        return b""

    def _auto_complete_abilities(self):
        conf = self._workflow_spec.comp_configs
        source_map = self._workflow_spec.edges
        target_map = self._source_to_target_map(source_map)
        source_stream_map = self._workflow_spec.stream_edges
        target_stream_map = self._source_to_target_map(source_stream_map)

        user_provided_abilities = {}
        for comp_id, comp_conf in conf.items():
            user_provided_abilities[comp_id] = len(comp_conf.abilities) > 0

        for source in source_stream_map:
            if not user_provided_abilities[source]:
                if source in target_map:
                    self._add_ability(conf, source, ComponentAbility.STREAM)
                if source in target_stream_map:
                    self._add_ability(conf, source, ComponentAbility.TRANSFORM)
        for target in target_stream_map:
            if not user_provided_abilities[target]:
                if target in source_map:
                    self._add_ability(conf, target, ComponentAbility.COLLECT)

    @staticmethod
    def _add_ability(conf: dict[str, NodeSpec], comp: str, ability: ComponentAbility):
        if ability not in conf[comp].abilities:
            conf[comp].abilities.append(ability)

    @staticmethod
    def _source_to_target_map(source_map: dict[str, list[str]]):
        target_map = {}
        for source, targets in source_map.items():
            for target in targets:
                if target not in target_map:
                    target_map[target] = []
                target_map[target].append(source)
        return target_map


class WorkflowExecutable(ABC):
    @abstractmethod
    async def invoke(self, inputs, runtime: BaseRuntime, context: Context = None) -> WorkflowOutput:
        pass

    @abstractmethod
    async def sub_invoke(self, inputs, runtime: BaseRuntime, config: Any = None) -> WorkflowOutput:
        pass

    @abstractmethod
    async def stream(
            self,
            inputs,
            runtime: BaseRuntime,
            context: Context = None,
            stream_modes: list[StreamMode] = None
    ) -> AsyncIterator[WorkflowChunk]:
        pass

    def get_tool_info(self) -> ToolInfo:
        pass


class Workflow(BaseWorkFlow, WorkflowExecutable):
    def __init__(self, workflow_config: WorkflowConfig = None):
        super().__init__(workflow_config, PregelGraph())
        self.tool_info = self._convert_to_tool_info(self._workflow_config.workflow_inputs_schema)
        self._end_comp_id: str = ""
        self._end_comp = None
        self._is_streaming = False
        self.inputs_schema = self._convert_to_tool_info(self._workflow_config.workflow_inputs_schema)

    def _convert_to_tool_info(self, inputs_schema: WorkflowInputsSchema) -> ToolInfo:
        parameters = Parameters(
            type=inputs_schema.type,
            properties=inputs_schema.properties,
            required=inputs_schema.required
        )
        function = Function(
            name=self._workflow_config.metadata.name,
            parameters=parameters,
            description=self._workflow_config.metadata.description,
        )
        return ToolInfo(function=function)

    def set_start_comp(
            self,
            start_comp_id: str,
            component: Union[Executable, WorkflowComponent],
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
            component: Union[Executable, WorkflowComponent],
            inputs_schema: dict = None,
            outputs_schema: dict = None,
            inputs_transformer: Transformer = None,
            outputs_transformer: Transformer = None,
            stream_inputs_schema: dict = None,
            stream_outputs_schema: dict = None,
            stream_inputs_transformer: Transformer = None,
            stream_outputs_transformer: Transformer = None,
            response_mode: str = None
    ) -> Self:
        wait_for_all = False
        if response_mode is not None and "streaming" == response_mode:
            comp_ability = [ComponentAbility.STREAM]
            self._is_streaming = True
            if stream_inputs_schema is not None or stream_inputs_transformer is not None:
                comp_ability.append(ComponentAbility.TRANSFORM)
                if isinstance(component, End):
                    component.set_mix()
            wait_for_all = True
        else:
            comp_ability = [ComponentAbility.INVOKE]
            if stream_inputs_schema is not None or stream_inputs_transformer is not None:
                comp_ability.append(ComponentAbility.COLLECT)
                if isinstance(component, End):
                    component.set_mix()
                wait_for_all = True
        self.add_workflow_comp(end_comp_id, component, wait_for_all=wait_for_all, inputs_schema=inputs_schema,
                               comp_ability=comp_ability,
                               outputs_schema=outputs_schema,
                               inputs_transformer=inputs_transformer,
                               outputs_transformer=outputs_transformer,
                               stream_inputs_schema=stream_inputs_schema,
                               stream_outputs_schema=stream_outputs_schema,
                               stream_inputs_transformer=stream_inputs_transformer,
                               stream_outputs_transformer=stream_outputs_transformer
                               )
        self.end_comp(end_comp_id)
        self._end_comp_id = end_comp_id
        self._end_comp = component
        return self

    async def sub_invoke(self, inputs: Input, runtime: BaseRuntime, config: Any = None) -> Output:
        logger.info(f"begin to sub_invoke, inputs: {inputs}")
        actor_manager, sub_workflow_runtime = self._prepare_sub_workflow_runtime(runtime)

        compiled_graph = self.compile(sub_workflow_runtime)
        await compiled_graph.invoke({INPUTS_KEY: inputs, CONFIG_KEY: config}, runtime)
        if self._is_streaming:
            messages = []
            while True:
                frame = await actor_manager.sub_workflow_stream().receive(
                    runtime.config().get_env(WORKFLOW_STREAM_TIMEOUT))
                if frame is None:
                    logger.warning("no frame received")
                    continue
                if frame == StreamEmitter.END_FRAME:
                    logger.info("received end frame of sub_invoke")
                    break
                messages.append(frame)
            if messages:
                logger.debug(f"sub workflow messages: {messages}")
                return dict(stream=messages)

        node_runtime = NodeRuntime(runtime, self._end_comp_id)
        output_key = self._end_comp_id
        if isinstance(self._end_comp, End):
            output_key = self._end_comp_id + NESTED_PATH_SPLIT + "output"
        results = node_runtime.state().get_outputs(output_key)
        logger.info(f"end to sub_invoke, result: {results}")
        return results

    async def sub_stream(self, inputs: Input, runtime: BaseRuntime, config: Any = None) -> AsyncIterator[Output]:
        logger.info(f"begin to sub_stream, input: {inputs}")
        actor_manager, sub_workflow_runtime = self._prepare_sub_workflow_runtime(runtime)

        compiled_graph = self.compile(sub_workflow_runtime)
        await compiled_graph.invoke({INPUTS_KEY: inputs, CONFIG_KEY: config}, runtime)
        if self._is_streaming:
            frame_count = 0
            stream_timeout = runtime.config().get_env(WORKFLOW_STREAM_TIMEOUT)
            while True:
                logger.debug(f"waiting for frame {frame_count} with timeout {stream_timeout}")
                frame = await actor_manager.sub_workflow_stream().receive(stream_timeout)
                if frame is None:
                    logger.warning("no frame received")
                    continue
                if frame == StreamEmitter.END_FRAME:
                    logger.info(f"received end frame of sub_stream after {frame_count} frames")
                    break
                frame_count += 1
                logger.debug(f"yielding frame {frame_count}: {frame}")
                yield frame

    async def invoke(self, inputs: Input, runtime: BaseRuntime, context: Context = None) -> WorkflowOutput:
        async def _invoke_task():
            logger.info(f"begin to invoke, input: {inputs}")
            chunks = []
            async for chunk in self.stream(inputs, runtime, context=context, stream_modes=[BaseStreamMode.OUTPUT]):
                chunks.append(chunk)

            is_interaction = False
            for chunk in chunks:
                if isinstance(chunk, OutputSchema) and chunk.type == INTERACTION:
                    is_interaction = True
                    break
            if is_interaction:
                output = WorkflowOutput(result=[chunk for chunk in chunks],
                                        state=WorkflowExecutionState.INPUT_REQUIRED)
            else:
                if self._is_streaming:
                    result = chunks
                else:
                    result = runtime.state().get_outputs(self._end_comp_id)
                output = WorkflowOutput(result=result, state=WorkflowExecutionState.COMPLETED)
            logger.info("end to invoke, results=%s", output)
            return output

        invoke_timeout = runtime.config().get_env(WORKFLOW_INVOKE_TIMEOUT)
        return await self._execute_with_timeout(_invoke_task, invoke_timeout, StatusCode.WORKFLOW_INVOKE_TIMEOUT)

    async def stream(
            self,
            inputs: Input,
            runtime: BaseRuntime,
            context: Context = None,
            stream_modes: list[StreamMode] = None
    ) -> AsyncIterator[WorkflowChunk]:
        self._validate_and_init_runtime(runtime, stream_modes, context)
        # workflow start tracer info
        await workflow_trace_inputs(runtime, inputs)
        timeout = runtime.config().get_env(WORKFLOW_STREAM_TIMEOUT)
        frame_timeout = runtime.config().get_env(WORKFLOW_STREAM_FRAME_TIMEOUT)
        frame_timeout = min(frame_timeout, self._workflow_config.stream_timeout) \
            if frame_timeout and frame_timeout > 0 else self._workflow_config.stream_timeout
        if timeout is not None and 0 < timeout <= frame_timeout:
            frame_timeout = timeout
        runtime.config().set_envs({WORKFLOW_STREAM_FRAME_TIMEOUT: frame_timeout})

        async def stream_process():
            compiled_graph = self.compile(runtime)
            try:
                await compiled_graph.invoke({INPUTS_KEY: inputs, CONFIG_KEY: None}, runtime)
            finally:
                # workflow end tracer info
                outputs = runtime.state().get_outputs(self._end_comp_id)
                await workflow_trace_outputs(runtime, outputs)
                await runtime.stream_writer_manager().stream_emitter().close()

        task = asyncio.create_task(
            self._execute_with_timeout(stream_process, timeout, StatusCode.WORKFLOW_STREAM_TIMEOUT))

        interaction_chuck_list = []
        async for chunk in runtime.stream_writer_manager().stream_output(frame_timeout):
            yield chunk
            if isinstance(chunk, OutputSchema) and chunk.type == INTERACTION:
                interaction_chuck_list.append(chunk)

        results = runtime.state().get_outputs(self._end_comp_id)
        if results:
            yield OutputSchema(type="workflow_final", index=0, payload=results)
            self._add_messages_to_context(inputs, results, context)
        elif interaction_chuck_list:
            self._add_messages_to_context(inputs, interaction_chuck_list, context)

        try:
            await task
        except Exception as e:
            raise e

    async def _execute_with_timeout(self, task, timeout, status_code):
        try:
            return await asyncio.wait_for(task(), timeout=timeout if (timeout and timeout > 0) else None)
        except asyncio.TimeoutError:
            raise JiuWenBaseException(status_code.code, status_code.errmsg.format(timeout=timeout))

    def _validate_and_init_runtime(self, runtime: BaseRuntime, stream_modes: list[StreamMode], context: Context):
        if isinstance(runtime, WorkflowRuntime):
            runtime.set_workflow_id(self._workflow_config.metadata.id)
            if context:
                runtime._context = context
        mq_manager = ActorManager(self._workflow_spec, self._stream_actor, sub_graph=False, runtime=runtime)
        runtime.set_actor_manager(mq_manager)
        runtime.set_stream_writer_manager(StreamWriterManager(stream_emitter=StreamEmitter(), modes=stream_modes))
        if runtime.tracer() is None and (stream_modes is None or BaseStreamMode.TRACE in stream_modes):
            tracer = Tracer()
            tracer.init(runtime.stream_writer_manager(), runtime.callback_manager())
            runtime.set_tracer(tracer)

    def _prepare_sub_workflow_runtime(self, runtime: BaseRuntime):
        """
        Prepare common components for sub workflow execution.
        
        Args:
            runtime: The base runtime
            
        Returns:
            tuple: (actor_manager, sub_workflow_runtime)
        """
        actor_manager = ActorManager(self._workflow_spec, self._stream_actor, sub_graph=True, runtime=runtime)
        sub_workflow_runtime = SubWorkflowRuntime(
            runtime,
            workflow_id=self._workflow_config.metadata.id,
            actor_manager=actor_manager
        )
        return actor_manager, sub_workflow_runtime

    def _convert_to_component(self, executable: Executable) -> WorkflowComponent:
        pass

    def get_tool_info(self) -> ToolInfo:
        return self.tool_info

    @staticmethod
    def _add_messages_to_context(inputs, results: Union[dict, List[OutputSchema]], context):
        if context is None:
            return

        user_messages = []
        if isinstance(inputs, dict):
            user_messages.append({"role": "user", "content": inputs.get("query", "")})
        elif isinstance(inputs, InteractiveInput):
            sorted_user_feedback = OrderedDict(inputs.user_inputs)
            user_feedback = "\n".join([feedback for _, feedback in sorted_user_feedback.items()])
            user_messages.append({"role": "user", "content": user_feedback})

        assistant_messages = []
        if isinstance(results, dict):
            workflow_result = results.get("responseContent") or results.get("output")
            assistant_messages.append({"role": "assistant", "content": workflow_result})
        elif isinstance(results, list):
            sorted_user_feedback = OrderedDict()
            for item in results:
                if isinstance(item, OutputSchema):
                    sorted_user_feedback.update({item.payload.id: item.payload.value})
            questions = "\n".join([question for _, question in sorted_user_feedback.items()])
            assistant_messages.append({"role": "assistant", "content": questions})

        context.batch_add_messages(user_messages + assistant_messages)
