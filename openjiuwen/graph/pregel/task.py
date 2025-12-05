#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved."""

from __future__ import annotations

import asyncio
import inspect
from typing import Dict, Optional, List

from openjiuwen.graph.checkpoint.base import PendingNode
from openjiuwen.graph.pregel.config import PregelConfig, InnerPregelConfig, \
    create_inner_config
from openjiuwen.graph.pregel.constants import GraphInterrupt, TASK_STATUS_INTERRUPT, TASK_STATUS_ERROR, PARENT_NS, NS
from openjiuwen.graph.pregel.messages import Message
from openjiuwen.graph.pregel.nodes import PregelNode


class TaskExecutorPool:
    def __init__(self, config: PregelConfig):
        self.config = config
        self.routed_messages: List[Message] = []
        self.failed: Dict[str, PendingNode] = {}
        self._tasks: Dict[asyncio.Task, PregelNode] = {}

    def submit(self, node: PregelNode) -> None:
        """Submit node's execution task"""
        async_task = asyncio.create_task(NodeTask(node, self.config).run())
        self._tasks[async_task] = node

    def _commit_failure(self, node: PregelNode, exc: Exception):
        """Record failed node and exception"""
        name = node.name
        if name not in self.failed:
            status = TASK_STATUS_INTERRUPT if isinstance(exc, GraphInterrupt) else TASK_STATUS_ERROR
            self.failed[name] = PendingNode(node_name=name, status=status, exception=[exc])

    async def wait_all(self) -> None:
        """Wait and process all tasks with FIRST_EXCEPTION semantics"""
        if not self._tasks:
            return

        tasks = list(self._tasks.keys())

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

        # Collect done results & exceptions
        first_exc: Optional[Exception] = None

        for t in done:
            node = self._tasks.pop(t)

            exc = t.exception()
            if exc is None:
                # Success - collect messages
                msgs: List[Message] = t.result()
                self.routed_messages.extend(msgs)
            else:
                # Failure
                self._commit_failure(node, exc)
                if first_exc is None:
                    first_exc = exc

        # Cancel remaining tasks
        for t in pending:
            node = self._tasks.pop(t)
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)
            # Cancellation is not a failure, only mark in pending nodes
            self._commit_failure(node, asyncio.CancelledError())

        if first_exc:
            raise first_exc

    def clear(self):
        self.routed_messages.clear()
        self.failed.clear()
        self._tasks.clear()


class NodeTask:
    def __init__(self, node: PregelNode, config: PregelConfig):
        self.node = node
        self.config = config
        self.messages: List[Message] = []

    async def run(self) -> List[Message]:
        func = self.node.func
        target_func = func.__call__ if hasattr(func, "__call__") else func
        sig = inspect.signature(func)
        kwargs = {}
        if 'config' in sig.parameters:
            # Create namespace for current node
            inner_config: InnerPregelConfig = create_inner_config(self.config)
            current_parent_ns = inner_config.get(PARENT_NS)
            current_node_name = self.node.name
            if current_parent_ns:
                new_full_ns = f"{current_parent_ns}:{current_node_name}"
                inner_config[NS] = new_full_ns
                inner_config[PARENT_NS] = new_full_ns
            kwargs['config'] = inner_config
        if 'state' in sig.parameters:
            kwargs['state'] = None

        if inspect.iscoroutinefunction(func) or asyncio.iscoroutinefunction(target_func):
            await target_func(**kwargs)
        else:
            target_func(**kwargs)

        # Route messages
        self.messages = []
        for r in self.node.routers:
            # Assume IRouter.route is an async method
            msgs = await r.dispatch(source_node=self.node.name)
            self.messages.extend(msgs)

        return self.messages
