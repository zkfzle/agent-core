#!/usr/bin/python3.10
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Union, Optional, Callable, Self

from jiuwen.core.common.exception.exception import JiuWenBaseException
from jiuwen.core.runtime.utils import update_dict, get_by_schema


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


class CommitStateLike(StateLike):
    @abstractmethod
    def update_by_id(self, node_id: str, data: dict) -> None:
        pass

    @abstractmethod
    def commit(self, node_id: str = None) -> None:
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
WORKFLOW_STATE_KEY = "workflow_state"
COMP_STATE_UPDATES_KEY = "comp_state_updates"
WORKFLOW_STATE_UPDATES_KEY = "workflow_state_updates"
DEFAULT_NODE_ID = "default"
DEFAULT_WORKFLOW_ID = "workflow"


class State(ABC):
    def __init__(
            self,
            io_state: CommitStateLike,
            global_state: CommitStateLike,
            comp_state: CommitStateLike,
            workflow_state: CommitStateLike,
            trace_state: dict = {},
            parent_id: str = '',
            node_id: str = DEFAULT_NODE_ID
    ):
        self._io_state = io_state
        self._global_state = global_state
        self._trace_state = trace_state
        self._comp_state = comp_state
        self._workflow_state = workflow_state
        self._parent_id = parent_id
        self._node_id = node_id

    def get_global(self, key: Union[str, list, dict]) -> Optional[Any]:
        if self._global_state is None or key is None:
            return None
        result = self._global_state.get(key)
        if result is None:
            return self._io_state.get_by_prefix(key, self._parent_id)
        return result

    def update_global(self, data: dict) -> None:
        if self._global_state is None or data is None:
            return
        self._global_state.update_by_id(self._node_id, data)

    def update_trace(self, span):
        self._trace_state.update({self._node_id: span})

    def update(self, data: dict) -> None:
        if self._comp_state is None:
            return
        self._comp_state.update_by_id(self._node_id, {self._node_id: data})

    def get(self, key: Union[str, list, dict] = None) -> Optional[Any]:
        if self._comp_state is None:
            return None
        if key is None:
            return self._comp_state.get(self._node_id)
        result = self._comp_state.get_by_prefix(key, self._node_id)
        return result

    def commit_cmp(self):
        self._comp_state.commit(self._node_id)
        self._io_state.commit(self._node_id)

class CommitState(State):
    def __init__(self, io_state: CommitStateLike,
                 global_state: CommitStateLike,
                 comp_state: CommitStateLike,
                 workflow_state: CommitStateLike,
                 trace_state: dict = {},
                 parent_id: str = '',
                 node_id: str = DEFAULT_NODE_ID):
        super().__init__(io_state=io_state, global_state=global_state, comp_state=comp_state, trace_state=trace_state,
                         workflow_state=workflow_state, parent_id=parent_id, node_id=node_id)

    def get_workflow_state(self, key: Union[str, list, dict]) -> Optional[Any]:
        if self._workflow_state is None or key is None:
            return None
        return self._workflow_state.get(key)

    def update_and_commit_workflow_state(self, data: dict):
        self._workflow_state.update_by_id(DEFAULT_WORKFLOW_ID, data)
        self._workflow_state.commit()

    def set_outputs(self, data: dict) -> None:
        if self._io_state is None or data is None:
            return
        self._io_state.update_by_id(self._node_id, {self._node_id: data})

    def get_inputs(self, schema: Union[str, list, dict] = None) -> Optional[Any]:
        if self._io_state is None:
            return None
        if schema is None:
            return self._io_state.get(self._node_id)
        result = self._io_state.get_by_prefix(schema, self._parent_id)
        return result

    def get_outputs(self, node_id) -> Optional[Any]:
        if self._io_state is None:
            return None
        return self._io_state.get_by_prefix(node_id, self._parent_id)

    def get_inputs_by_transformer(self, transformer: Callable) -> dict:
        if self._io_state is None:
            return {}
        return self._io_state.get_by_transformer(transformer)

    def commit_user_inputs(self, inputs: Any) -> None:
        if self._io_state is None or inputs is None:
            return
        self._io_state.update_by_id(self._node_id,
                                    {self._node_id: inputs} if self._node_id != DEFAULT_NODE_ID else inputs)
        self._global_state.update_by_id(self._node_id, inputs)
        self.commit()

    def commit(self) -> None:
        self._io_state.commit()
        self._comp_state.commit()
        self._global_state.commit()
        self._workflow_state.commit()

    def rollback(self) -> None:
        self._comp_state.rollback(self._node_id)
        self._io_state.rollback(self._node_id)
        self._global_state.rollback(self._node_id)
        self._workflow_state.rollback(self._node_id)

    def get_state(self) -> dict:
        return {
            IO_STATE_KEY: self._io_state.get_state(),
            GLOBAL_STATE_KEY: self._global_state.get_state(),
            COMP_STATE_KEY: self._comp_state.get_state(),
            WORKFLOW_STATE_KEY: self._workflow_state.get_state()
        }

    def set_state(self, state: dict) -> None:
        self._io_state.set_state(state.get(IO_STATE_KEY))
        self._global_state.set_state(state.get(GLOBAL_STATE_KEY))
        self._comp_state.set_state(state.get(COMP_STATE_KEY))
        self._workflow_state.set_state(state.get(WORKFLOW_STATE_KEY))

    def get_updates(self) -> dict:
        return {
            IO_STATE_UPDATES_KEY: self._io_state.get_updates(),
            GLOBAL_STATE_UPDATES_KEY: self._global_state.get_updates(),
            COMP_STATE_UPDATES_KEY: self._comp_state.get_updates(),
            WORKFLOW_STATE_UPDATES_KEY: self._workflow_state.get_updates()
        }

    def set_updates(self, updates: dict) -> None:
        self._io_state.set_updates(updates.get(IO_STATE_UPDATES_KEY))
        self._global_state.set_updates(updates.get(GLOBAL_STATE_UPDATES_KEY))
        self._comp_state.set_updates(updates.get(COMP_STATE_UPDATES_KEY))
        self._workflow_state.set_updates(updates.get(WORKFLOW_STATE_UPDATES_KEY))

    def create_node_state(self, node_id: str, parent_id: str) -> Self:
        return CommitState(io_state=self._io_state, global_state=self._global_state,
                           comp_state=self._comp_state, workflow_state=self._workflow_state,
                           trace_state=self._trace_state, node_id=node_id, parent_id=parent_id)


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


class InMemoryCommitState(CommitStateLike):
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

    def commit(self, node_id: str = None) -> None:
        if node_id is None:
            for key, updates in self._updates.items():
                for update in updates:
                    self._state.update(update)
            self._updates.clear()
        else:
            node_updates = self._updates.get(node_id)
            if not node_updates:
                return
            for update in node_updates:
                self._state.update(update)
            self._updates[node_id] = []

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


class InMemoryState(CommitState):
    def __init__(self, global_state: CommitStateLike = None):
        super().__init__(io_state=InMemoryCommitState(),
                         global_state=global_state if global_state is not None else InMemoryCommitState(),
                         workflow_state=InMemoryCommitState(),
                         trace_state=dict(),
                         comp_state=InMemoryCommitState())
