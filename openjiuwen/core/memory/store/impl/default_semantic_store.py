#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import threading
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
import faiss
import numpy as np
import requests
from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.store.impl.faiss_semantic_utils import SearchType, TimeUtil
from openjiuwen.core.memory.store.base_semantic_store import BaseSemanticStore
from urllib.parse import urljoin


def convert_faiss_result(distance: np.ndarray, ids: np.ndarray) -> List[List[Tuple[str, float]]]:
    result = []
    for i in range(distance.shape[0]):
        hits = [
            (str(ids[i][j]), float(distance[i][j]))
            for j in range(distance.shape[1]) if int(ids[i][j]) != -1
        ]
        result.append(hits)
    return result


class DefaultSemanticStore(BaseSemanticStore):
    vector_persist_interval = 1 * 24 * 60 * 60

    def __init__(self, vector_store_dir: str, embedding_addr: str, embedding_dims: int):
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
        self.embedding_addr = embedding_addr
        self.embedding_dims = embedding_dims

    def __with_lock(self, index_name: str):
        return self.index_locks[index_name]

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        embedding_addr: embedding server addr, example: "http://127.0.0.1:8000"
        texts: List[str], text list
        return: List[List[float]], embedding list
        """
        url = urljoin(self.embedding_addr, "/embedding")
        payload = {"texts": texts}
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if "embeddings" not in data:
                raise ValueError(f"response missing 'embeddings': {data}")
            embs = data["embeddings"]
            if len(embs[0]) != self.embedding_dims:
                raise ValueError(
                    f"embeddings dimension mismatch: expected {self.embedding_dims}, got {len(embs[0])}"
                )
            return embs
        except Exception as e:
            logger.error(f"[get_embeddings] request failed: {e}")
            return None

    async def add_docs(self, docs: List[Tuple[str, str]], table_name: str) -> bool:
        memory_ids, memories = zip(*docs)
        memory_ids = list(memory_ids)
        memories = list(memories)
        embeddings = self._get_embeddings(texts=memories)
        dimension = len(embeddings[0])
        if len(memory_ids) != len(embeddings):
            raise ValueError(f"memory_ids and embeddings must have same length, len of docs={len(memory_ids)}, "
                             f"len of embeddings={len(embeddings)}")
        with self.__with_lock(table_name):
            index = self.__get_faiss_index(table_name, dimension, True)
            vectors_arr = np.array(embeddings, dtype=np.float32)
            ids_arr = np.array(memory_ids, dtype=np.int64)
            if self.normalize_L2:
                faiss.normalize_L2(vectors_arr)
            index.add_with_ids(vectors_arr, ids_arr)
            self.mod_cnt[table_name] += 1
            self.__flush(table_name)
        return True

    async def delete_docs(self, ids: List[str], table_name: str) -> bool:
        cur_index = self.__get_faiss_index(table_name)
        if cur_index is None:
            return True
        # need thread safe
        with self.__with_lock(table_name):
            ids_arr = np.array(ids, dtype=np.int64)
            cur_index.remove_ids(ids_arr)
            self.mod_cnt[table_name] += 1
            self.__flush(table_name)
        return True

    async def search(self, query: str, table_name: str, top_k: int) -> List[Tuple[str, float]]:
        embedding = self._get_embeddings(texts=[query])
        with self.__with_lock(table_name):
            cur_index = self.__get_faiss_index(table_name)
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
        for index_name, _ in self.index_store.items():
            with self.__with_lock(index_name):
                self.__flush(index_name)

    async def delete_table(self, table_name: str) -> bool:
        if table_name not in self.index_store:
            return True
        with self.__with_lock(table_name):
            del self.index_store[table_name]

        for filename in Path(self.fold_path).rglob(f"*{self.suffix}"):
            if table_name == filename.stem:
                filename.unlink()
        return True
