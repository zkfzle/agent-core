#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved."""

from typing import Optional

from openjiuwen.core.runtime.runtime import BaseRuntime
from openjiuwen.graph.checkpoint.base import (
    CheckpointerSaver,
    Checkpoint
)


class GraphCheckpointer(CheckpointerSaver):
    def __init__(self, runtime: BaseRuntime, saver: CheckpointerSaver):
        self._saver = saver
        self.ctx = runtime

    def reset(self, runtime: BaseRuntime):
        self.ctx = runtime

    async def get(self, session_id: str, ns: str) -> Optional[Checkpoint]:
        return await self._saver.get(session_id, ns)

    async def save(self, session_id: str, ns: str, checkpoint: Checkpoint) -> None:
        return await self._saver.save(session_id, ns, checkpoint)

    async def delete(self, session_id: str, ns: str | None = None) -> None:
        return await self._saver.delete(session_id, ns)
