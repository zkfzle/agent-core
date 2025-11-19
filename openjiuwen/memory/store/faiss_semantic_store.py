#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import threading
from pathlib import Path
from typing import List
from collections import defaultdict
import re
import faiss
import numpy as np

from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.common.base import generate_idx_name
from openjiuwen.memory.store.faiss_semantic_utils import SearchType, TimeUtil
from openjiuwen.core.memory.store.base_semantic_store import BaseSemanticStore, SearchHit
from openjiuwen.memory.store.embedding_model import EmbeddingModel


def match_index_name(match_list: List[str], cur_index: str) -> bool:
    cur_parts = re.split(r'\^', cur_index)
    if len(cur_parts) != 5 and len(cur_parts) != 4:
        logger.warning(f"find")
        return False
    result = True
    for match_part, cur_part in zip(match_list, cur_parts):
        if match_part == "*" or match_part == cur_part:
            continue
        else:
            result = False
            break
    return result

def convert_faiss_result(distance: np.ndarray, ids: np.ndarray) -> List[List[SearchHit]]:
    result = []
    for i in range(distance.shape[0]):
        hits = [
            SearchHit(id=str(ids[i][j]), distance=float(distance[i][j]))
            for j in range(distance.shape[1]) if int(ids[i][j]) != -1
        ]
        result.append(hits)
    return result

class FaissSemanticStore(BaseSemanticStore):
    vector_persist_interval = 1 * 24 * 60 * 60

    def __init__(self, vector_store_dir: str, model_name_or_path: str):

        self.normalize_L2 = True
        self.search_type = SearchType.COSINE
        self.mod_cnt = defaultdict(int)
        self.index_store: dict[str, faiss.Index] = {}
        self.suffix = ".faiss"
        path = Path(vector_store_dir)
        self.fold_path = str(path.resolve())

        self.instance_lock = threading.RLock()
        self.index_locks: defaultdict[str, threading.RLock] = defaultdict(threading.RLock)
        self.timer = TimeUtil(interval=self.vector_persist_interval, callback=self.__persist_all)
        self.timer.start()
        self.closed = False
        self.embedding_model = EmbeddingModel(model_name_or_path)

    def __with_lock(self, index_name: str):
        return self.index_locks[index_name]

    def add(self, mem: List[str], memory_id: List[str], user_id: str, app_id: str, mem_type: str | None = None) -> None:
        index_name = generate_idx_name(usr_id=user_id, app_id=app_id, mem_type=mem_type)
        embeddings = self.embedding_model.encode(texts=mem)
        dimension = len(embeddings[0])
        if len(memory_id) != len(embeddings):
            raise ValueError(f"ids and embeddings must have same length, len of mem_id={len(memory_id)}, "
                             f"len of embeddings={len(embeddings)}")
        with self.__with_lock(index_name):
            index = self.__get_faiss_index(index_name, dimension, True)
            vectors_arr = np.array(embeddings, dtype=np.float32)
            ids_arr = np.array(memory_id, dtype=np.int64)
            if self.normalize_L2:
                faiss.normalize_L2(vectors_arr)
            index.add_with_ids(vectors_arr, ids_arr)
            self.mod_cnt[index_name] += 1
            self.__flush(index_name)

    def remove(self, ids: List[str], user_id: str, app_id: str, mem_type: str | None = None) -> None:
        index_name = generate_idx_name(usr_id=user_id, app_id=app_id, mem_type=mem_type)
        cur_index = self.__get_faiss_index(index_name)
        if cur_index is None:
            return
        # need thread safe
        with self.__with_lock(index_name):
            ids_arr = np.array(ids, dtype=np.int64)
            cur_index.remove_ids(ids_arr)
            self.mod_cnt[index_name] += 1
            self.__flush(index_name)

    def search(self, query: List[str], user_id: str, app_id: str, mem_type: str | None = None, top_k: int = 5):
        index_name = generate_idx_name(usr_id=user_id, app_id=app_id, mem_type=mem_type)
        embedding = self.embedding_model.encode(texts=query)
        with self.__with_lock(index_name):
            cur_index = self.__get_faiss_index(index_name)
            if cur_index is None:
                return []
            search_vectors_arr = np.array(embedding, dtype=np.float32)
            if self.normalize_L2:
                faiss.normalize_L2(search_vectors_arr)
            distance, ids = cur_index.search(search_vectors_arr, top_k)
            search_result = convert_faiss_result(distance, ids)[0]
            return search_result

    def __flush(self, index_name: str) -> None:
        cur_index = self.__get_faiss_index(index_name)
        path = Path(self.fold_path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(cur_index, str(path / f"{index_name}{self.suffix}"))

    def __get_faiss_index(self, index_name: str, dimension: int = -1, create_if_absent: bool = False) -> faiss.Index:
        if index_name in self.index_store:
            return self.index_store[index_name]
        path = Path(self.fold_path)
        index: faiss.Index | None = None
        try:
            index = faiss.read_index(str(path / f"{index_name}{self.suffix}"))
        except Exception:
            logger.debug("")
        if index is None and create_if_absent:
            base_index = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIDMap2(base_index)
        if index is not None:
            self.index_store[index_name] = index
        return index

    def __persist_all(self):
        for index_name, index in self.index_store.items():
            with self.__with_lock(index_name):
                self.__flush(index_name)

    def delete_index_by_match(self, match_str: str) -> None:
        match_list = re.split(r'\^', match_str)
        if len(match_list) != 4:
            logger.error(f"invalid match_str: {match_str}")
            return
        del_list = [k for k, v in self.index_store.items() if match_index_name(match_list, k)]
        for k in del_list:
            with self.__with_lock(k):
                del self.index_store[k]

        for filename in Path(self.fold_path).rglob(f"*{self.suffix}"):
            if match_index_name(match_list, filename.stem):
                filename.unlink()


