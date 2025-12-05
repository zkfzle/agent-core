#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    List,
    Dict,
    Optional
)

from openjiuwen.graph.pregel.messages import Message


@dataclass
class PendingNode:
    node_name: str
    status: str
    exception: list[Exception] = None


@dataclass
class Checkpoint:
    ns: str
    step: int
    channel_values: Dict[str, Any]
    pending_buffer: List[Message]
    pending_node: Dict[str, PendingNode]

class CheckpointerSaver(ABC):
    @abstractmethod
    async def get(self, session_id: str, ns: str) -> Optional[Checkpoint]: ...

    @abstractmethod
    async def save(self, session_id: str, ns: str, checkpoint: Checkpoint) -> None: ...

    @abstractmethod
    async def delete(self, session_id: str, ns : str | None = None) -> None: ...


def create_checkpoint(
        ns: str,
        step: int,
        channel_snapshot: Dict[str, Any],
        *,
        pending_buffer: Optional[List[Message]] = None,
        pending_node: Optional[Dict[str, PendingNode]] = None,
) -> Checkpoint:
    return Checkpoint(
        ns=ns,
        step=step,
        channel_values=channel_snapshot,
        pending_buffer=pending_buffer or [],
        pending_node=pending_node or {},
    )
