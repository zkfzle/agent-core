#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Union, Optional, Callable, Self

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.context.utils import update_dict, get_by_schema


class ReadableStateLike(ABC):
    @abstractmethod
    def get(self, key: Union[str, list, dict]) -> Optional[Any]:
        pass

    @abstractmethod
    def get_by_prefix(self, key: Union[str, list, dict], nested_prefix: str) -> Optional[Any]:
        pass


Transformer = Callable[[ReadableStateLike], Any]


class StateLike(ReadableStateLike):
    @abstractmethod
    def update(self, data: dict) -> None:
        pass

    @abstractmethod
    def get_by_transformer(self, transformer: Transformer) -> Optional[Any]:
        pass

    @abstractmethod
    def get_state(self) -> dict:
        pass

    @abstractmethod
    def set_state(self, state: dict) -> None:
        pass


class CommitState(StateLike):
    @abstractmethod
    def update_by_id(self, node_id: str, data: dict) -> None:
        pass

    @abstractmethod
    def commit(self) -> None:
        pass

    @abstractmethod
    def rollback(self, node_id: str) -> None:
        pass

    @abstractmethod
    def get_updates(self) -> dict:
        pass

    @abstractmethod
    def set_updates(self, updates: dict):
        pass


IO_STATE_KEY = "io_state"
IO_STATE_UPDATES_KEY = "io_state_updates"
GLOBAL_STATE_KEY = "global_state"
GLOBAL_STATE_UPDATES_KEY = "global_state_updates"
COMP_STATE_KEY = "comp_state"
COMP_STATE_UPDATES_KEY = "comp_state_updates"
DEFAULT_NODE_ID = "default"


class State(ABC):
    def __init__(
            self,
            io_state: CommitState,
            global_state: CommitState,
            comp_state: CommitState,
            trace_state: dict = {},
            node_id: str = DEFAULT_NODE_ID
    ):
        self._io_state = io_state
        self._global_state = global_state
        self._trace_state = trace_state
        self._comp_state = comp_state
        self._node_id = node_id

    def get(self, key: Union[str, list, dict]) -> Optional[Any]:
        if self._global_state is None:
            return None
        value = self._global_state.get(key)
        if value is None:
            return self._io_state.get(key)
        return value

    def update(self, data: dict) -> None:
        if self._global_state is None:
            return
        self._global_state.update_by_id(self._node_id, data)

    def update_io(self, data: dict) -> None:
        if self._io_state is None:
            return
        self._io_state.update_by_id(self._node_id, data)

    def get_io(self, key: Union[str, list, dict]) -> Optional[Any]:
        if self._io_state is None:
            return
        return self._io_state.get(key)

    def update_trace(self, span):
        self._trace_state.update({self._node_id: span})

    def update_comp(self, data: dict) -> None:
        if self._comp_state is None:
            return
        self._comp_state.update_by_id(self._node_id, {self._node_id: data})

    def get_comp(self, key: Union[str, list, dict]) -> Optional[Any]:
        if self._comp_state is None:
            return
        return self._comp_state.get_by_prefix(key, self._node_id)

    def set_user_inputs(self, inputs: Any) -> None:
        if self._io_state is None or inputs is None:
            return
        self._io_state.update_by_id(self._node_id, inputs)
        self._global_state.update_by_id(self._node_id, inputs)
        self.commit()

    def get_inputs_by_transformer(self, transformer: Callable) -> dict:
        if self._io_state is None:
            return {}
        return self._io_state.get_by_transformer(transformer)

    def get_outputs(self, node_id: str) -> Any:
        if self._io_state is None:
            return {}
        return self._io_state.get(node_id)

    def set_outputs(self, node_id: str, outputs: dict) -> None:
        if self._io_state is None or outputs is None:
            return
        return self._io_state.update_by_id(node_id, {node_id: outputs})

    def create_node_state(self, node_id: str) -> Self:
        return State(io_state=self._io_state, global_state=self._global_state, comp_state=self._comp_state,
                     trace_state=self._trace_state, node_id=node_id)

    def commit(self) -> None:
        self._io_state.commit()
        self._comp_state.commit()
        self._global_state.commit()

    def rollback(self) -> None:
        self._comp_state.rollback(self._node_id)
        self._io_state.rollback(self._node_id)
        self._global_state.rollback(self._node_id)

    def get_state(self) -> dict:
        return {
            IO_STATE_KEY: self._io_state.get_state(),
            GLOBAL_STATE_KEY: self._global_state.get_state(),
            COMP_STATE_KEY: self._comp_state.get_state(),
        }

    def set_state(self, state: dict) -> None:
        self._io_state.set_state(state.get(IO_STATE_KEY))
        self._global_state.set_state(state.get(GLOBAL_STATE_KEY))
        self._comp_state.set_state(state.get(COMP_STATE_KEY))

    def get_updates(self) -> dict:
        return {
            IO_STATE_UPDATES_KEY: self._io_state.get_updates(),
            GLOBAL_STATE_UPDATES_KEY: self._global_state.get_updates(),
            COMP_STATE_UPDATES_KEY: self._comp_state.get_updates(),
        }

    def set_updates(self, updates: dict) -> None:
        self._io_state.set_updates(updates.get(IO_STATE_UPDATES_KEY))
        self._global_state.set_updates(updates.get(GLOBAL_STATE_UPDATES_KEY))
        self._comp_state.set_updates(updates.get(COMP_STATE_UPDATES_KEY))


class InMemoryStateLike(StateLike):
    def __init__(self):
        self._state: dict = dict()

    def get(self, key: Union[str, list, dict]) -> Optional[Any]:
        return get_by_schema(key, self._state)

    def get_by_prefix(self, key: Union[str, list, dict], nested_prefix: str) -> Optional[Any]:
        return get_by_schema(key, self._state, nested_prefix)

    def get_by_transformer(self, transformer: Callable) -> Optional[Any]:
        return transformer(self._state)

    def update(self, data: dict) -> None:
        update_dict(data, self._state)

    def get_state(self) -> dict:
        return deepcopy(self._state)

    def set_state(self, state: dict) -> None:
        if state:
            self._state = state


class InMemoryCommitState(CommitState):
    def __init__(self, state: StateLike = None):
        self._state = state if state else InMemoryStateLike()
        self._updates: dict[str, list[dict]] = dict()

    def update(self, data: dict) -> None:
        raise JiuWenBaseException(-1, "commit state update must support node_id")

    def update_by_id(self, node_id: str, data: dict) -> None:
        if node_id is None:
            raise JiuWenBaseException(1, "can not update state by none node_id")
        if node_id not in self._updates:
            self._updates[node_id] = []
        self._updates[node_id].append(data)

    def commit(self) -> None:
        for key, updates in self._updates.items():
            for update in updates:
                self._state.update(update)
        self._updates.clear()

    def rollback(self, node_id: str) -> None:
        self._updates[node_id] = []

    def get_by_transformer(self, transformer: Transformer) -> Optional[Any]:
        return transformer(self._state)

    def get(self, key: Union[str, list, dict]) -> Optional[Any]:
        return self._state.get(key)

    def get_by_prefix(self, key: Union[str, list, dict], nested_prefix: str) -> Optional[Any]:
        return self._state.get_by_prefix(key, nested_prefix)

    def get_updates(self) -> dict:
        return self._updates

    def set_updates(self, updates: dict):
        if updates:
            self._updates = updates

    def get_state(self) -> dict:
        return self._state.get_state()

    def set_state(self, state: dict) -> None:
        self._state.set_state(state)


class InMemoryState(State):
    def __init__(self, global_state: CommitState = InMemoryCommitState()):
        super().__init__(io_state=InMemoryCommitState(),
                         global_state=global_state,
                         trace_state=dict(),
                         comp_state=InMemoryCommitState())
