#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from __future__ import annotations

import asyncio
from typing import Any, Union

from langgraph.errors import GraphInterrupt
from langgraph.types import Interrupt
from pydantic import BaseModel

from jiuwen.core.common.constants.constant import INTERACTION
from jiuwen.core.common.constants.constant import INTERACTIVE_INPUT
from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.runtime.runtime import Runtime, BaseRuntime
from jiuwen.core.stream.writer import OutputSchema


class InteractionOutput(BaseModel):
    id: str
    value: Any


class Interaction(object):
    def __init__(self, runtime: Union[Runtime, BaseRuntime]):
        if isinstance(runtime, Runtime):
            self.ctx = runtime.base()
        elif isinstance(runtime, BaseRuntime):
            self.ctx = runtime
        else:
            raise JiuWenBaseException(-1, "wrong type runtime")
        self.idx = 0
        self.node_id = self.ctx.executable_id()
        self.interactive_inputs = None

        if workflow_interactive_input := self.ctx.state().get_workflow_state(INTERACTIVE_INPUT):
            self.interactive_inputs = [workflow_interactive_input]
            self.ctx.state().update_and_commit_workflow_state({INTERACTIVE_INPUT: None})

        interactive_inputs = self.ctx.state().get(INTERACTIVE_INPUT)
        if isinstance(interactive_inputs, list):
            if self.interactive_inputs:
                self.interactive_inputs += interactive_inputs
            else:
                self.interactive_inputs = interactive_inputs

        if self.interactive_inputs:
            self.ctx.state().update({INTERACTIVE_INPUT: self.interactive_inputs})

        self.latest_interactive_inputs = None
        if self.interactive_inputs:
            self.latest_interactive_inputs = self.interactive_inputs[-1]

    def _get_next_interactive_input(self) -> Any | None:
        if self.interactive_inputs and self.idx < len(self.interactive_inputs):
            res = self.interactive_inputs[self.idx]
            self.idx += 1
            return res
        return None

    async def wait_user_inputs(self, value: Any) -> Any:
        if res := self._get_next_interactive_input():
            return res
        self.ctx.state().commit_cmp()
        payload = InteractionOutput(id=self.node_id, value=value)
        if self.ctx.stream_writer_manager():
            output_writer = self.ctx.stream_writer_manager().get_output_writer()
            await output_writer.write(OutputSchema(type=INTERACTION, index=self.idx, payload=payload))

        raise GraphInterrupt((Interrupt(
            value=OutputSchema(type=INTERACTION, index=self.idx, payload=payload)),))

    def user_latest_input(self, value: Any) -> Any:
        if res := self.latest_interactive_inputs:
            self.latest_interactive_inputs = None
            return res
        if self.ctx.stream_writer_manager:
            output_writer = self.ctx.stream_writer_manager().get_output_writer()
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    output_writer.write(OutputSchema(type=INTERACTION, index=self.idx, payload=(self.node_id, value))))
            else:
                loop.run_until_complete(
                    output_writer.write(OutputSchema(type=INTERACTION, index=self.idx, payload=(self.node_id, value))))

        raise GraphInterrupt((Interrupt(
            value=OutputSchema(type=INTERACTION, index=self.idx, payload=(self.node_id, value)), resumable=True,
            ns=self.node_id),))
