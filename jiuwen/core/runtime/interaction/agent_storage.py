#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from jiuwen.core.runtime.interaction.interactive_input import InteractiveInput
from jiuwen.core.runtime.runtime import BaseRuntime
from jiuwen.core.runtime.interaction.storage import Storage


class AgentStorage(Storage):
    def __init__(self):
        self.state_blobs: dict[
            str,
            tuple[str, bytes],
        ] = {}

        self.serde: SerializerProtocol = JsonPlusSerializer()

    def save(self, runtime: BaseRuntime):
        session_id = runtime.session_id()
        state = runtime.state().get_state()
        if state_blob := self.serde.dumps_typed(state):
            self.state_blobs[session_id] = state_blob

    def recover(self, runtime: BaseRuntime, inputs: InteractiveInput = None):
        session_id = runtime.session_id()
        state_blob = self.state_blobs.get(session_id)
        if state_blob is None:
            return
        state = self.serde.loads_typed(state_blob)
        runtime.state().set_state(state)

    def clear(self, session_id: str):
        self.state_blobs.pop(session_id, None)
