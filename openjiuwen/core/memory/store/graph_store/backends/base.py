# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from threading import Lock
from typing import Dict, Optional, Type

from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.config.graph.config import GraphConfig

from .graph_backend import GraphBackend


class GraphBackendFactory:
    """Factory class to assemble database backend objects for graph memory"""

    class_map: Dict[str, Type[GraphBackend]] = dict()
    __thread_lock: Lock = Lock()

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Please do not instantiate GraphBackendFactory")

    @classmethod
    def register_backend(cls, name: str, backend: Type[GraphBackend], force: bool = False):
        """Register vector database backend for storing / representing graph memory.

        Args:
            name (str): Name for the new database backend.
            backend (Type[GraphBackend]): Class for the new database backend.
            force (bool, optional): Whether to force register. Defaults to False.

        Raises:
            KeyError: 1) name is empty or 2) name already registered and force=False.
            NotImplementedError: backend did not implement the GraphBackend Protocol.
        """
        with cls.__thread_lock:
            if not name:
                raise KeyError("Backend name cannot be registered as an empty value.")
            if name in cls.class_map and not force:
                raise KeyError(f"Entry [{name}] -> {cls.class_map[name]} already exists.")
            if not isinstance(backend, GraphBackend):
                err_msg = f"{name} did not implement GraphBackend Protocol!"
                if not force:
                    raise NotImplementedError(err_msg)
                logger.warning(err_msg)
            cls.class_map[name] = backend
            logger.info("[图谱记忆]数据库后端注册成功：%s", name)

    @classmethod
    def from_config(cls, config: GraphConfig, backend_name: Optional[str] = None, **kwargs) -> GraphBackend:
        """Fetch a GraphBackend instance by configuration file.

        Args:
            config (GraphConfig): Database configuration.
            backend_name (Optional[str], optional): If not None, overwrites database backend choice in config. \
                Defaults to None.

        Raises:
            KeyError: The database backend choice has not been registered in GraphBackendFactory.

        Returns:
            GraphBackend: instance of database backend for graph memory.
        """
        with cls.__thread_lock:
            name = backend_name or config.backend
            backend_cls = cls.class_map.get(name)
            if backend_cls:
                return backend_cls.from_config(config, **kwargs)
            raise KeyError(f"Backend type [{name}] does not exist.")
