#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from __future__ import annotations

from typing import Any, Iterator, Sequence, AsyncIterator

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id
)
from langgraph.checkpoint.memory import InMemorySaver

from jiuwen.core.common.constants.constant import INTERACTIVE_INPUT
from jiuwen.core.runtime.runtime import NodeRuntime
from jiuwen.core.graph.interrupt.interactive_input import InteractiveInput
from jiuwen.graph.checkpoint.base import BaseCheckpointer

STATE_KEY = "state"
STATE_UPDATES_KEY = "state_updates"


class InMemoryCheckpointer(BaseCheckpointer[str]):

    def __init__(self):
        super().__init__()

        # (thread ID, checkpoint ns, checkpoint ID, io_state KEY) -> (value type, value dumped bytes)
        self.state_blobs: dict[
            tuple[
                str, str, str, str
            ],
            tuple[str, bytes],
        ] = {}

        # (thread ID, checkpoint ns, checkpoint ID, io_state_updates KEY) -> (value type, value dumped bytes)
        self.state_updates_blobs: dict[
            tuple[
                str, str, str, str
            ],
            tuple[str, bytes]
        ] = {}

        self.in_mem_saver = InMemorySaver()

    def recover(self, config: RunnableConfig):
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id is None and (checkpoints := self.in_mem_saver.storage[thread_id][checkpoint_ns]):
            checkpoint_id = max(checkpoints.keys())
        if checkpoint_id is None:
            return

        if (state_blob := self.state_blobs.get((thread_id, checkpoint_ns, checkpoint_id, STATE_KEY))) and \
                state_blob[0] != "empty":
            state = self.serde.loads_typed(state_blob)
            self.ctx.state().set_state(state)

        if isinstance(self.input, InteractiveInput):
            if self.input.raw_inputs is not None:
                self.ctx.state().update_and_commit_workflow_state({INTERACTIVE_INPUT: self.input.raw_inputs})
            else:
                for node_id, user_input in self.input.user_inputs.items():
                    exe_ctx = NodeRuntime(self.ctx, node_id)
                    interactive_input = exe_ctx.state().get(INTERACTIVE_INPUT)
                    if isinstance(interactive_input, list):
                        interactive_input.append(user_input)
                        exe_ctx.state().update({INTERACTIVE_INPUT: interactive_input})
                        continue
                    exe_ctx.state().update({INTERACTIVE_INPUT: [user_input]})
                self.ctx.state().commit()

        if state_updates_blob := self.state_updates_blobs.get(
                (thread_id, checkpoint_ns, checkpoint_id, STATE_UPDATES_KEY)):
            state_updates = self.serde.loads_typed(state_updates_blob)
            self.ctx.state().set_updates(state_updates)

    def save(self, config: RunnableConfig):
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id is None and (checkpoints := self.in_mem_saver.storage[thread_id][checkpoint_ns]):
            checkpoint_id = max(checkpoints.keys())
        if checkpoint_id is None:
            return

        if self.ctx:
            state = self.ctx.state().get_state()
            if state_blob := self.serde.dumps_typed(state):
                self.state_blobs[(thread_id, checkpoint_ns, checkpoint_id, STATE_KEY)] = state_blob

            updates = self.ctx.state().get_updates()
            if updates_blob := self.serde.dumps_typed(updates):
                self.state_updates_blobs[
                    (thread_id, checkpoint_ns, checkpoint_id, STATE_UPDATES_KEY)] = updates_blob

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.in_mem_saver.get_tuple(config=config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        return self.in_mem_saver.list(config=config, filter=filter, before=before, limit=limit)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.in_mem_saver.put(config=config, checkpoint=checkpoint, metadata=metadata, new_versions=new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.in_mem_saver.put_writes(config=config, writes=writes, task_id=task_id, task_path=task_path)

    def delete_thread(
        self,
        thread_id: str,
    ) -> None:
        self.in_mem_saver.delete_thread(thread_id=thread_id)
        for key in list(self.state_blobs.keys()):
            if key[0] == thread_id:
                del self.state_blobs[key]

        for key in list(self.state_updates_blobs.keys()):
            if key[0] == thread_id:
                del self.state_updates_blobs[key]

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await self.in_mem_saver.aget_tuple(config=config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        return self.in_mem_saver.alist(config=config, filter=filter, before=before, limit=limit)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await self.in_mem_saver.aput(config=config, checkpoint=checkpoint, metadata=metadata, new_versions=new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return await self.in_mem_saver.aput_writes(config=config, writes=writes, task_id=task_id, task_path=task_path)

    async def adelete_thread(
        self,
        thread_id: str,
    ) -> None:
        return self.delete_thread(thread_id=thread_id)

    def get_next_version(self, current: str | None, channel: None) -> str:
        return self.in_mem_saver.get_next_version(current=current, channel=channel)

default_inmemory_checkpointer = InMemoryCheckpointer()
