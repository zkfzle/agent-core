#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved."""

from collections import defaultdict
from typing import Dict, Optional

from openjiuwen.graph.checkpoint.base import (
    CheckpointerSaver,
    Checkpoint
)


class MemoryCheckpointSaver(CheckpointerSaver):
    def __init__(self) -> None:
        # 存储最新 checkpoint： { session_id: { ns: Checkpoint } }
        self.store_ck: defaultdict[str, Dict[str, Checkpoint]] = defaultdict(dict)

    async def get(self, session_id: str, ns: str) -> Optional[Checkpoint]:
        return self.store_ck.get(session_id, {}).get(ns)

    async def save(self, session_id: str, ns: str, checkpoint: Checkpoint) -> None:
        # store the checkpoint object directly
        self.store_ck[session_id][ns] = checkpoint

    async def delete(self, session_id: str, ns : str | None = None) -> None:
        if session_id not in self.store_ck:
            return  # Conversation ID doesn't exist, nothing to delete

        if ns is None:
            # Delete all namespaces for this session_id
            del self.store_ck[session_id]
        else:
            # Delete specific namespace under session_id
            if ns in self.store_ck[session_id]:
                del self.store_ck[session_id][ns]

            # If session_id becomes empty after deletion, clean it up
            if not self.store_ck[session_id]:
                del self.store_ck[session_id]
