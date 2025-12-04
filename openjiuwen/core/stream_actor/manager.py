#!/usr/bin/env python
# coding: utf-8
# Copyright c) Huawei Technologies Co. Ltd. 2025-2025. All rights reserved.

import asyncio
from typing import Dict, Any

from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.logging import logger
from openjiuwen.core.runtime.constants import STREAM_INPUT_GEN_TIMEOUT_KEY
from openjiuwen.core.runtime.state import Transformer
from openjiuwen.core.runtime.utils import get_by_schema
from openjiuwen.core.stream.emitter import AsyncStreamQueue, StreamEmitter
from openjiuwen.core.stream_actor.base import StreamActor, StreamGraph
from openjiuwen.core.workflow.workflow_config import ComponentAbility, WorkflowSpec


class StreamTransform:
    def get_by_defined_transformer(self, origin_message: dict, transformer: Transformer) -> dict:
        return transformer(origin_message)

    def get_by_default_transformer(self, origin_message: dict, stream_inputs_schema: dict) -> dict:
        return get_by_schema(stream_inputs_schema, origin_message)


class ActorManager:
    def __init__(self, workflow_spec: WorkflowSpec, graph: StreamGraph, sub_graph: bool, runtime):
        self._stream_edges = workflow_spec.stream_edges
        self._streams: Dict[str, StreamActor] = {}
        self._streams_transform = StreamTransform()

        consumer_dict = _build_reverse_graph(self._stream_edges)
        for consumer_id, producer_ids in consumer_dict.items():
            consumer_stream_ability = [ability for ability in workflow_spec.comp_configs[consumer_id].abilities if
                                       ability in [ComponentAbility.COLLECT, ComponentAbility.TRANSFORM]]
            self._streams[consumer_id] = StreamActor(consumer_id, graph.get_node(consumer_id),
                                                     consumer_stream_ability, producer_ids,
                                                     stream_generator_timeout=runtime.config().get_env(
                                                         STREAM_INPUT_GEN_TIMEOUT_KEY))

        self._sub_graph = sub_graph
        self._sub_workflow_stream = AsyncStreamQueue(maxsize=10 * 1024) if sub_graph else None

        # Producer stream ability count management
        self._producer_stream_ability_counts: Dict[str, int] = {}
        self._producer_completed_counts: Dict[str, int] = {}
        self._producer_locks: Dict[str, asyncio.Lock] = {}
        self._init_producer_stream_ability_counts(workflow_spec)

    def _init_producer_stream_ability_counts(self, workflow_spec: WorkflowSpec) -> None:
        """Initialize producer stream output ability counts."""
        stream_output_abilities = {ComponentAbility.STREAM, ComponentAbility.TRANSFORM}

        for producer_id in self._stream_edges.keys():
            node_spec = workflow_spec.comp_configs.get(producer_id)
            if not node_spec or not node_spec.abilities:
                continue

            count = 0
            for ability in node_spec.abilities:
                if ability in stream_output_abilities:
                    count += 1

            if count > 1:  # Only multi-ability nodes need counting
                self._producer_stream_ability_counts[producer_id] = count
                self._producer_completed_counts[producer_id] = 0
                self._producer_locks[producer_id] = asyncio.Lock()
                logger.debug(
                    f"producer [{producer_id}] registered with {count} stream output abilities"
                )

    def sub_workflow_stream(self) -> AsyncStreamQueue:
        if not self._sub_graph:
            raise JiuWenBaseException(
                error_code=StatusCode.WORKFLOW_MESSAGE_QUEUE_MANAGER_ERROR.code,
                message=f"only sub graph has sub_workflow_stream")
        return self._sub_workflow_stream

    def _get_actor(self, consumer_id: str) -> StreamActor:
        return self._streams[consumer_id]

    @property
    def stream_transform(self):
        return self._streams_transform

    async def produce(self, producer_id: str, message_content: Any):
        consumer_ids = self._stream_edges.get(producer_id)
        if consumer_ids:
            for consumer_id in consumer_ids:
                actor = self._get_actor(consumer_id)
                logger.debug(f"send message to consumer [{consumer_id}] actor from producer [{producer_id}]")
                await actor.send({producer_id: message_content})

    async def end_message(self, producer_id: str):
        end_message_content = f"END_{producer_id}"
        await self.produce(producer_id, end_message_content)

    async def notify_stream_ability_done(
        self,
        producer_id: str,
        is_end_node: bool,
        is_sub_graph: bool
    ) -> None:
        """
        Notify that a producer's stream ability has completed.
        Only sends end message when ALL stream abilities are done.

        For sub-workflow end nodes (is_end_node=True and is_sub_graph=True):
            Send END_FRAME for each stream output ability completion.
            This maintains backward compatibility with sub_stream.

        For non-end nodes with multiple stream output abilities:
            Only send end_message when ALL stream output abilities are completed.
        """
        # Sub-workflow end nodes: send END_FRAME immediately for each ability
        if is_end_node and is_sub_graph:
            await self.sub_workflow_stream().send(StreamEmitter.END_FRAME)
            return

        # Single stream ability nodes: send end message directly
        if producer_id not in self._producer_stream_ability_counts:
            await self.end_message(producer_id)
            return

        # Multi-stream ability nodes: count management
        async with self._producer_locks[producer_id]:
            self._producer_completed_counts[producer_id] += 1
            completed = self._producer_completed_counts[producer_id]
            total = self._producer_stream_ability_counts[producer_id]

            logger.debug(
                f"producer [{producer_id}] completed "
                f"{completed}/{total} stream output abilities"
            )

            if completed < total:
                return

            # Reset counter for potential node reuse (e.g., in loop workflows)
            self._producer_completed_counts[producer_id] = 0

            logger.debug(
                f"producer [{producer_id}] all stream abilities done, sending end message"
            )
            await self.end_message(producer_id)

    async def consume(self, consumer_id: str, ability: ComponentAbility, schema: dict) -> dict:
        actor = self._get_actor(consumer_id)
        return await actor.generator(ability, schema)


def _build_reverse_graph(graph):
    reverse_graph = {}

    for source, targets in graph.items():
        for target in targets:
            if target not in reverse_graph:
                reverse_graph[target] = []
            reverse_graph[target].append(source)

    return reverse_graph
