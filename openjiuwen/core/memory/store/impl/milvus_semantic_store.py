# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import List, Tuple, Any
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, connections, utility
from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.store.base_semantic_store import BaseSemanticStore


MEMORY_ID_LENGTH = 36


def convert_milvus_result(results) -> List[List[Tuple[str, float]]]:
    final_results = []
    for hits_per_query in results:
        hits = []
        for hit in hits_per_query:
            memory_id = hit.entity.get("memory_id")  # 获取字段
            distance = hit.distance
            hits.append((memory_id, distance))
        final_results.append(hits)
    return final_results


class MilvusSemanticStore(BaseSemanticStore):
    def __init__(self, milvus_host: str, milvus_port: str, token: str | None,
                 embed_model: Any, embedding_dims: int):
        self.embed_model = embed_model
        self.embedding_dims = embedding_dims
        self.token = token
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.time_out = 3
        self.collections = {}

    def _ensure_connection(self):
        connections.connect(
            host=self.milvus_host,
            port=self.milvus_port,
            alias="default",
            token=self.token,
            timeout=self.time_out
        )

    def get_collection(self, collection_name: str) -> Collection:
        self._ensure_connection()
        if collection_name in self.collections:
            return self.collections[collection_name]
        if not utility.has_collection(collection_name):
            logger.info(f"Collection {collection_name} not found, creating...")
            fields = [
                FieldSchema(name="memory_id", dtype=DataType.VARCHAR,
                            is_primary=True, max_length=MEMORY_ID_LENGTH),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,
                            dim=self.embedding_dims),
            ]
            schema = CollectionSchema(fields, description="embedding collection")
            collection = Collection(name=collection_name, schema=schema, using="default")
            index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}
            collection.create_index("embedding", index_params)
            logger.info(f"Index created for collection {collection_name}")
        else:
            logger.info(f"milvus collection already exists: {collection_name}")
            collection = Collection(name=collection_name, using="default")
        collection.load()
        self.collections[collection_name] = collection
        return collection

    async def add_docs(self, docs: List[Tuple[str, str]], table_name: str) -> bool:
        memory_ids, memories = zip(*docs)
        memory_ids = list(memory_ids)
        memories = list(memories)
        embeddings = await self.embed_model.embed_queries(texts=memories)
        if len(memory_ids) != len(embeddings):
            raise ValueError(f"memory_ids and embeddings must have same length")
        collection = self.get_collection(collection_name=table_name)
        vectors_arr = [
            [float(x) for x in vec]
            for vec in embeddings
        ]
        collection.insert([
            memory_ids,
            vectors_arr
        ], timeout=self.time_out)
        return True

    async def delete_docs(self, ids: List[str], table_name: str) -> bool:
        self._ensure_connection()
        if not utility.has_collection(table_name, using="default"):
            logger.debug(f"Milvus Collection {table_name} does not exist, skip delete vector")
            return True
        collection = self.get_collection(table_name)
        ids_str = ", ".join(f'"{i}"' for i in ids)
        expr = f'memory_id in [{ids_str}]'
        collection.delete(expr, timeout=self.time_out)
        return True

    async def search(self, query: str, table_name: str, top_k: int) -> List[Tuple[str, float]]:
        collection = self.get_collection(table_name)
        query_vector = await self.embed_model.embed_queries(texts=[query])
        results = collection.search(
            data=query_vector,
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=top_k,
            timeout=self.time_out
        )
        parsed_results = convert_milvus_result(results)
        return parsed_results[0] if parsed_results else []

    async def delete_table(self, table_name: str) -> bool:
        self._ensure_connection()
        if not utility.has_collection(table_name, using="default"):
            logger.debug(f"Milvus Collection {table_name} does not exist, skip delete collection")
            return True
        utility.drop_collection(table_name, timeout=self.time_out)
        self.collections.pop(table_name, None)
        return True
