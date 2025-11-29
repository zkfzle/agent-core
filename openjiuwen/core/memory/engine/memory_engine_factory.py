#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import threading
from typing import Optional

from sqlalchemy import Engine
from openjiuwen.core.memory.store.base_kv_store import BaseKVStore
from openjiuwen.core.memory.engine.memory_engine_base import MemoryEngineBase
from openjiuwen.core.memory.config.config import Config
from openjiuwen.core.memory.engine.memory_engine import MemoryEngine
from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.store.base_semantic_store import BaseSemanticStore
from openjiuwen.core.utils.llm.base import BaseModelClient



def new(
    config: Config,
    llm_base: BaseModelClient | None = None,
    semantic_recall_instance: BaseSemanticStore | None = None,
    kv_store_instance: BaseKVStore | None = None,
    db_engine_instance: Engine | None = None
) -> MemoryEngineBase | None:
    try:
        mem_engine = MemoryEngine(config, llm_base)
        mem_engine.init_mem_store(
            semantic_db_instance=semantic_recall_instance,
            kv_db_instance=kv_store_instance,
            db_engine_instance=db_engine_instance
        )
        return mem_engine
    except Exception as e:
        logger.error(f"Failed to create MemoryEngine: {e}")
        return None


_memengine_singleton_instance: Optional[MemoryEngine] = None
_memengine_singleton_lock = threading.Lock()
_kv_db_instance: Optional[BaseKVStore] = None
_semantic_recall_instance: Optional[BaseSemanticStore] = None
_db_engine_instance: Optional[Engine] = None
_llm_base: Optional[BaseModelClient] = None

def register_kv_db(kv_db_instance: BaseKVStore):
    global _kv_db_instance
    if issubclass(kv_db_instance.__class__, BaseKVStore):
        _kv_db_instance = kv_db_instance
    else:
        logger.error("kv db must be subclass of BaseKVStore")

def register_semantic_recall(semantic_recall_instance: BaseSemanticStore):
    global _semantic_recall_instance
    if issubclass(semantic_recall_instance.__class__, BaseSemanticStore):
        _semantic_recall_instance = semantic_recall_instance
    else:
        logger.error("semantic recall instance must be subclass of BaseSemanticStore")

def register_relation_db(db_engine_instance: Engine):
    global _db_engine_instance
    if issubclass(db_engine_instance.__class__, Engine):
        _db_engine_instance = db_engine_instance
    else:
        logger.error("db engine instance must be subclass of Engine")

def register_llm(llm_base: BaseModelClient):
    global _llm_base
    if issubclass(llm_base.__class__, BaseModelClient):
        _llm_base = llm_base
    else:
        logger.error("llm base must be subclass of BaseModelClient")

def get_memengine_instance(config: Config) -> MemoryEngineBase | None:
    global _memengine_singleton_instance
    if _memengine_singleton_instance is None:
        with _memengine_singleton_lock:
            if _memengine_singleton_instance is None:
                if _kv_db_instance is None:
                    logger.error("Failed to new memory engine, you need register kv store")
                    return None
                _memengine_singleton_instance = new(
                    config=config,
                    semantic_recall_instance=_semantic_recall_instance,
                    kv_store_instance=_kv_db_instance,
                    db_engine_instance=_db_engine_instance,
                    llm_base=_llm_base
                )
    return _memengine_singleton_instance
