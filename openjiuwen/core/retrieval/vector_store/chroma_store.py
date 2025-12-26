# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
ChromaDB 向量存储实现

支持向量搜索、稀疏搜索（文本匹配）和混合搜索。
"""
import uuid
import asyncio
import json
from typing import Any, List, Optional
import chromadb

from openjiuwen.core.common.logging import logger
from openjiuwen.core.retrieval.vector_store.base import VectorStore
from openjiuwen.core.retrieval.common.retrieval_result import SearchResult, RetrievalResult
from openjiuwen.core.retrieval.common.config import VectorStoreConfig
from openjiuwen.core.retrieval.utils.fusion import rrf_fusion


class ChromaVectorStore(VectorStore):
    """ChromaDB 向量存储实现"""

    def __init__(
        self,
        config: VectorStoreConfig,
        chroma_path: str,
        text_field: str = "content",
        vector_field: str = "embedding",
        sparse_vector_field: str = "sparse_vector",
        metadata_field: str = "metadata",
        doc_id_field: str = "document_id",
        **kwargs: Any,
    ):
        """
        初始化 ChromaDB 向量存储（持久化模式）
        
        Args:
            config: 向量存储配置
            chroma_path: ChromaDB 持久化路径（必需）
            text_field: 文本字段名
            vector_field: 向量字段名
            sparse_vector_field: 稀疏向量字段名（ChromaDB 中作为元数据存储）
            metadata_field: 元数据字段名
            doc_id_field: 文档ID字段名
        
        Raises:
            ValueError: 如果 chroma_path 未提供或为空
        """
        # 校验 chroma_path
        if not chroma_path or not chroma_path.strip():
            raise ValueError("chroma_path is required and cannot be empty")
        
        self.config = config
        self.collection_name = config.collection_name
        self.chroma_path = chroma_path
        self.text_field = text_field
        self.vector_field = vector_field
        self.sparse_vector_field = sparse_vector_field
        self.metadata_field = metadata_field
        self.doc_id_field = doc_id_field
        
        # 初始化 ChromaDB 持久化客户端
        self._client = chromadb.PersistentClient(path=chroma_path)
        
        # 获取或创建集合
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine" if config.distance_metric == "cosine" else "l2"
            }
        )

    @property
    def client(self):
        """获取 ChromaDB 客户端"""
        return self._client

    @property
    def collection(self):
        """获取 ChromaDB 集合"""
        return self._collection

    async def add(
        self,
        data: dict | List[dict],
        batch_size: int | None = 128,
        **kwargs: Any,
    ) -> None:
        """添加向量数据"""
        if batch_size is None or batch_size <= 0:
            batch_size = 128

        if isinstance(data, dict):
            data = [data]

        processed = 0
        total = len(data)
        cache: list[dict] = []
        
        for doc in data:
            cache.append(doc)
            if len(cache) >= batch_size:
                nodes = cache[:batch_size]
                cache = []
                await self._add_batch(nodes)
                processed += len(nodes)
                if processed % 100 == 0:
                    logger.info(
                        "Written %d/%d records to %s",
                        processed,
                        total,
                        self.collection_name,
                    )
        
        if cache:
            await self._add_batch(cache)
            processed += len(cache)
        
        logger.info(
            "Writing completed, total %d/%d records to %s",
            processed,
            total,
            self.collection_name,
        )

    async def _add_batch(self, nodes: List[dict]) -> None:
        """批量添加数据到 ChromaDB"""
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for node in nodes:
            # 提取向量
            embedding = node.get(self.vector_field, [])
            if not embedding:
                # 如果没有向量，生成一个警告但继续处理（某些情况下可能允许）
                logger.warning(f"Node has no embedding, skipping: {node.get('id', 'unknown')}")
                continue
            
            # 提取 ID
            node_id = str(node.get("id", node.get("pk", "")))
            if not node_id:
                node_id = str(uuid.uuid4())
            ids.append(node_id)
            embeddings.append(embedding)
            
            # 提取文本
            text = node.get(self.text_field, "")
            documents.append(text)
            
            # 构建元数据
            metadata = {}
            # 复制原始元数据
            if self.metadata_field in node:
                raw_metadata = node[self.metadata_field]
                if isinstance(raw_metadata, dict):
                    metadata.update(raw_metadata)
                elif isinstance(raw_metadata, str):
                    try:
                        metadata.update(json.loads(raw_metadata))
                    except Exception:
                        pass
            
            # 添加其他字段到元数据
            if self.doc_id_field in node:
                metadata[self.doc_id_field] = str(node[self.doc_id_field])
            if "chunk_id" in node:
                metadata["chunk_id"] = str(node["chunk_id"])
            if self.sparse_vector_field in node:
                # 稀疏向量作为元数据存储（ChromaDB 不直接支持稀疏向量）
                sparse_vec = node[self.sparse_vector_field]
                if isinstance(sparse_vec, (list, dict)):
                    metadata[self.sparse_vector_field] = json.dumps(sparse_vec)
            
            metadatas.append(metadata)
        
        # 如果没有有效数据，直接返回
        if not ids:
            return
        
        # 重新获取集合，确保使用最新的集合引用
        collection = await asyncio.to_thread(
            self._client.get_collection,
            name=self.collection_name,
        )
        
        # 添加到 ChromaDB
        await asyncio.to_thread(
            collection.add,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """向量搜索"""
        # 重新获取集合，确保使用最新的集合引用
        collection = await asyncio.to_thread(
            self._client.get_collection,
            name=self.collection_name,
        )
        
        # 构建 where 过滤条件
        where = None
        if filters:
            where = {}
            for key, value in filters.items():
                if isinstance(value, str):
                    where[key] = value
                else:
                    where[key] = value
        
        # 执行搜索
        results = await asyncio.to_thread(
            collection.query,
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
        )
        
        return self._chroma_result_to_search_results(results, mode="vector")

    async def sparse_search(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """稀疏搜索（文本匹配）"""
        # 重新获取集合，确保使用最新的集合引用
        collection = await asyncio.to_thread(
            self._client.get_collection,
            name=self.collection_name,
        )
        
        # ChromaDB 不直接支持 BM25，使用文本查询作为替代
        # 构建 where 过滤条件
        where = None
        if filters:
            where = {}
            for key, value in filters.items():
                if isinstance(value, str):
                    where[key] = value
                else:
                    where[key] = value
        
        try:
            # 使用文本查询（ChromaDB 的文本搜索基于 TF-IDF）
            results = await asyncio.to_thread(
                collection.query,
                query_texts=[query_text],
                n_results=top_k,
                where=where,
            )
            
            if results and results.get("ids") and len(results["ids"][0]) > 0:
                return self._chroma_result_to_search_results(results, mode="sparse")
            return []
        except Exception as e:
            logger.warning(f"Text search failed: {e}")
            return []

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: Optional[List[float]] = None,
        top_k: int = 5,
        alpha: float = 0.5,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """混合搜索（文本检索 + 向量检索）"""
        # 构建 where 过滤条件
        where = None
        if filters:
            where = {}
            for key, value in filters.items():
                if isinstance(value, str):
                    where[key] = value
                else:
                    where[key] = value
        
        try:
            # 分别执行向量搜索和文本搜索
            tasks = []
            
            if query_vector is not None:
                task_vector = asyncio.create_task(
                    self.search(query_vector, top_k * 2, filters)
                )
                tasks.append(("vector", task_vector))
            
            task_text = asyncio.create_task(
                self.sparse_search(query_text, top_k * 2, filters)
            )
            tasks.append(("text", task_text))
            
            # 等待所有任务完成
            results_dict = {}
            for mode, task in tasks:
                try:
                    results_dict[mode] = await task
                except Exception as e:
                    logger.warning(f"{mode} search failed in hybrid search: {e}")
                    results_dict[mode] = []
            
            # 融合结果
            results_list = [r for r in results_dict.values() if r]
            if not results_list:
                return []
            
            # 将 SearchResult 转换为 RetrievalResult 以使用 rrf_fusion
            # 同时保存 ID 映射以便后续恢复
            retrieval_results_list = []
            id_mapping = {}  # text -> id 映射
            
            for search_results in results_list:
                retrieval_results = []
                for sr in search_results:
                    # 保存 ID 映射
                    id_mapping[sr.text] = sr.id
                    # 确保 ID 在元数据中
                    metadata = sr.metadata.copy()
                    metadata["id"] = sr.id
                    retrieval_results.append(
                        RetrievalResult(
                            text=sr.text,
                            score=sr.score,
                            metadata=metadata,
                            doc_id=sr.metadata.get(self.doc_id_field),
                            chunk_id=sr.metadata.get("chunk_id"),
                        )
                    )
                retrieval_results_list.append(retrieval_results)
            
            # 使用 RRF 融合
            fused_retrieval_results = rrf_fusion(retrieval_results_list, k=60)
            
            # 将 RetrievalResult 转换回 SearchResult
            fused_results = []
            for rr in fused_retrieval_results[:top_k]:
                # 从元数据或映射中恢复 ID
                result_id = rr.metadata.get("id") or id_mapping.get(rr.text, str(hash(rr.text)))
                # 从元数据中移除临时添加的 id 字段
                metadata = rr.metadata.copy()
                metadata.pop("id", None)
                search_result = SearchResult(
                    id=result_id,
                    text=rr.text,
                    score=rr.score,
                    metadata=metadata,
                )
                fused_results.append(search_result)
            
            return fused_results
        except Exception as e:
            logger.warning(
                f"Hybrid search failed: {e}"
            )
            return []

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """删除向量"""
        try:
            # 重新获取集合，确保使用最新的集合引用
            collection = await asyncio.to_thread(
                self._client.get_collection,
                name=self.collection_name,
            )
            
            if ids:
                # 通过 ID 删除
                await asyncio.to_thread(
                    collection.delete,
                    ids=ids,
                )
                return True
            elif filter_expr:
                # ChromaDB 不支持复杂的 filter_expr，需要先查询再删除
                # 这里简化处理，只支持简单的 where 条件
                logger.warning(
                    "ChromaDB does not support complex filter expressions for deletion. "
                    "Please use ids parameter instead."
                )
                return False
            else:
                logger.warning("Either ids or filter_expr must be provided")
                return False
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False

    def _chroma_result_to_search_results(
        self,
        results: dict,
        mode: str,
    ) -> List[SearchResult]:
        """将 ChromaDB 搜索结果转换为 SearchResult 列表"""
        search_results = []
        
        if not results or "ids" not in results or not results["ids"]:
            return search_results
        
        ids_list = results["ids"][0] if results["ids"] else []
        documents_list = results.get("documents", [[]])[0] if results.get("documents") else []
        metadatas_list = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances_list = results.get("distances", [[]])[0] if results.get("distances") else []
        
        for idx, result_id in enumerate(ids_list):
            # 提取字段
            text = documents_list[idx] if idx < len(documents_list) else ""
            metadata = metadatas_list[idx] if idx < len(metadatas_list) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            
            # 处理稀疏向量元数据
            if self.sparse_vector_field in metadata:
                try:
                    sparse_vec = json.loads(metadata[self.sparse_vector_field])
                    metadata[self.sparse_vector_field] = sparse_vec
                except Exception:
                    pass
            
            # 计算分数
            raw_score = distances_list[idx] if idx < len(distances_list) else None
            raw_score_val = float(raw_score) if raw_score is not None else None
            raw_score_scaled: Optional[float] = None
            final_score: float = 0.0
            
            if mode == "vector":
                # ChromaDB 返回的是距离，需要转换为相似度分数
                if raw_score_val is not None:
                    # 对于 cosine 距离，相似度 = 1 - 距离
                    # 对于 L2 距离，需要归一化
                    if self.config.distance_metric == "cosine":
                        raw_score_scaled = 1.0 - raw_score_val
                    else:
                        # L2 距离，简单归一化（假设最大距离为 2）
                        raw_score_scaled = max(0.0, 1.0 - raw_score_val / 2.0)
                    final_score = raw_score_scaled
            elif mode == "sparse":
                # 文本搜索的分数（ChromaDB 可能返回相似度分数或距离）
                # 如果没有距离信息，使用默认分数
                if raw_score_val is not None:
                    # 如果是距离，转换为相似度
                    if raw_score_val <= 1.0:
                        final_score = 1.0 - raw_score_val
                    else:
                        final_score = raw_score_val
                else:
                    # 没有分数信息，使用默认值
                    final_score = 0.5
            else:  # hybrid 或其他
                if raw_score_val is not None:
                    # 对于混合搜索，分数已经在融合时计算
                    final_score = raw_score_val
                else:
                    final_score = 0.0
            
            metadata.setdefault("raw_score", raw_score_val)
            if raw_score_scaled is not None:
                metadata.setdefault("raw_score_scaled", raw_score_scaled)
            
            search_result = SearchResult(
                id=str(result_id),
                text=text,
                score=final_score,
                metadata=metadata,
            )
            search_results.append(search_result)
        
        return search_results

    def close(self) -> None:
        """关闭向量存储"""
        # ChromaDB 客户端通常不需要显式关闭
        # 但如果是持久化客户端，可以重置
        if hasattr(self, "_client") and self._client is not None:
            try:
                # ChromaDB 客户端没有 close 方法，但可以重置
                pass
            except Exception as e:
                logger.warning(f"Failed to close ChromaDB client: {e}")

