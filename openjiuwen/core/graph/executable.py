# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import TypeVar, Generic, AsyncIterator, Any

from openjiuwen.core.common.exception.errors import build_error
from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.session import BaseSession

Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", covariant=True)


class Executable(Generic[Input, Output]):
    async def on_invoke(self, inputs: Input, session: BaseSession, **kwargs) -> Output:
        raise build_error(StatusCode.METHOD_NOT_IMPLEMENTED, method="on_invoke", class_name=type(self).__name__)

    async def on_stream(self, inputs: Input, session: BaseSession, **kwargs) -> AsyncIterator[Output]:
        raise build_error(StatusCode.METHOD_NOT_IMPLEMENTED, method="on_stream", class_name=type(self).__name__)

    async def on_collect(self, inputs: Input, session: BaseSession, **kwargs) -> Output:
        raise build_error(StatusCode.METHOD_NOT_IMPLEMENTED, method="on_collect", class_name=type(self).__name__)

    async def on_transform(self, inputs: Input, session: BaseSession, **kwargs) -> AsyncIterator[Output]:
        raise build_error(StatusCode.METHOD_NOT_IMPLEMENTED, method="on_transform", class_name=type(self).__name__)

    def skip_trace(self) -> bool:
        return False

    def graph_invoker(self) -> bool:
        return False

    def post_commit(self) -> bool:
        return True

    def component_type(self) -> str:
        return ""


GeneralExecutor = Executable[dict[str, Any], dict[str, Any]]
