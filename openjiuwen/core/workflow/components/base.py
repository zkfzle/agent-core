#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from typing import AsyncIterator, TypeVar

from openjiuwen.core.graph.base import Graph
from openjiuwen.core.common.exception.exception import JiuWenBaseException
from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.context_engine import ModelContext
from openjiuwen.core.graph.executable import Executable
from openjiuwen.core.session import Session, BaseSession
from openjiuwen.core.session import NodeSession
from openjiuwen.core.session import WrappedNodeSession


@dataclass
class WorkflowComponentMetadata:
    node_id: str
    node_type: str
    node_name: str


@dataclass
class ComponentConfig:
    metadata: Optional[WorkflowComponentMetadata] = field(default=None)


@dataclass
class ComponentState:
    comp_id: str
    status: Enum


Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", contravariant=True)


class ComponentExecutable(Executable):

    async def on_invoke(self, inputs: Input, session: BaseSession) -> Output:
        if not isinstance(session, NodeSession):
            raise JiuWenBaseException(StatusCode.SESSION_COMPONENT_INVALID_SESSION_TYPE.code,
                                      StatusCode.SESSION_COMPONENT_INVALID_SESSION_TYPE.errmsg)

        current_class = type(self)
        # Check if the attribute exists, is callable, and is not the base implementation
        if (hasattr(current_class, 'invoke') and
                callable(getattr(current_class, 'invoke')) and
                current_class.invoke is ComponentExecutable.invoke):
            raise JiuWenBaseException(StatusCode.SESSION_COMPONENT_ABILITY_NOT_IMPLEMENTED.code,
                                      StatusCode.SESSION_COMPONENT_ABILITY_NOT_IMPLEMENTED.errmsg.format(
                                          ability='INVOKE', method='invoke', class_name=type(self).__name__))

        return await self.invoke(inputs, WrappedNodeSession(session), session.context())

    async def on_stream(self, inputs: Input, session: BaseSession) -> AsyncIterator[Output]:
        if not isinstance(session, NodeSession):
            raise JiuWenBaseException(StatusCode.SESSION_COMPONENT_INVALID_SESSION_TYPE.code,
                                      StatusCode.SESSION_COMPONENT_INVALID_SESSION_TYPE.errmsg)

        current_class = type(self)

        # Check if the attribute exists, is callable, and is not the base implementation
        if (hasattr(current_class, 'stream') and
                callable(getattr(current_class, 'stream')) and
                current_class.stream is ComponentExecutable.stream):
            raise JiuWenBaseException(StatusCode.SESSION_COMPONENT_ABILITY_NOT_IMPLEMENTED.code,
                                      StatusCode.SESSION_COMPONENT_ABILITY_NOT_IMPLEMENTED.errmsg.format(
                                          ability='STREAM', method='stream', class_name=type(self).__name__))

        async for value in self.stream(inputs, WrappedNodeSession(session), session.context()):
            yield value

    async def on_collect(self, inputs: Input, session: BaseSession) -> Output:
        if not isinstance(session, NodeSession):
            raise JiuWenBaseException(StatusCode.SESSION_COMPONENT_INVALID_SESSION_TYPE.code,
                                      StatusCode.SESSION_COMPONENT_INVALID_SESSION_TYPE.errmsg)

        current_class = type(self)

        # Check if the attribute exists, is callable, and is not the base implementation
        if (hasattr(current_class, 'collect') and
                callable(getattr(current_class, 'collect')) and
                current_class.collect is ComponentExecutable.collect):
            raise JiuWenBaseException(StatusCode.SESSION_COMPONENT_ABILITY_NOT_IMPLEMENTED.code,
                                      StatusCode.SESSION_COMPONENT_ABILITY_NOT_IMPLEMENTED.errmsg.format(
                                          ability='COLLECT', method='collect', class_name=type(self).__name__))

        return await self.collect(inputs, WrappedNodeSession(session, True), session.context())

    async def on_transform(self, inputs: Input, session: BaseSession) -> AsyncIterator[Output]:
        if not isinstance(session, NodeSession):
            raise JiuWenBaseException(StatusCode.SESSION_COMPONENT_INVALID_SESSION_TYPE.code,
                                      StatusCode.SESSION_COMPONENT_INVALID_SESSION_TYPE.errmsg)

        current_class = type(self)

        # Check if the attribute exists, is callable, and is not the base implementation
        if (hasattr(current_class, 'transform') and
                callable(getattr(current_class, 'transform')) and
                current_class.transform is ComponentExecutable.transform):
            raise JiuWenBaseException(StatusCode.SESSION_COMPONENT_ABILITY_NOT_IMPLEMENTED.code,
                                      StatusCode.SESSION_COMPONENT_ABILITY_NOT_IMPLEMENTED.errmsg.format(
                                          ability='TRANSFORM', method='transform', class_name=type(self).__name__))

        async for value in self.transform(inputs, WrappedNodeSession(session, True), session.context()):
            yield value

    async def invoke(self, inputs: Input, session: Session, context: ModelContext) -> Output:
        raise JiuWenBaseException(StatusCode.SESSION_COMPONENT_ABILITY_NOT_SUPPORTED.code,
                                  StatusCode.SESSION_COMPONENT_ABILITY_NOT_SUPPORTED.errmsg.format(ability='Invoke'))

    async def stream(self, inputs: Input, session: Session, context: ModelContext) -> AsyncIterator[Output]:
        raise JiuWenBaseException(StatusCode.SESSION_COMPONENT_ABILITY_NOT_SUPPORTED.code,
                                  StatusCode.SESSION_COMPONENT_ABILITY_NOT_SUPPORTED.errmsg.format(ability='Stream'))

    async def collect(self, inputs: Input, session: Session, context: ModelContext) -> Output:
        raise JiuWenBaseException(StatusCode.SESSION_COMPONENT_ABILITY_NOT_SUPPORTED.code,
                                  StatusCode.SESSION_COMPONENT_ABILITY_NOT_SUPPORTED.errmsg.format(ability='Collect'))

    async def transform(self, inputs: Input, session: Session, context: ModelContext) -> AsyncIterator[Output]:
        raise JiuWenBaseException(StatusCode.SESSION_COMPONENT_ABILITY_NOT_SUPPORTED.code,
                                  StatusCode.SESSION_COMPONENT_ABILITY_NOT_SUPPORTED.errmsg.format(ability='Transform'))


class WorkflowComponent(ABC):

    def add_component(self, graph: Graph, node_id: str, wait_for_all: bool = False) -> None:
        graph.add_node(node_id, self.to_executable(), wait_for_all=wait_for_all)

    def to_executable(self) -> Executable:
        if isinstance(self, Executable):
            return self
        raise JiuWenBaseException(
            StatusCode.COMPONENT_NOT_EXECUTABLE_ERROR.code, "workflow component should implement Executable"
        )


class SimpleComponent(ComponentExecutable, WorkflowComponent):
    ...
