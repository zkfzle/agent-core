#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import asyncio
import tempfile
import unittest
from typing import List

from openjiuwen.core.memory.store.impl.chroma_semantic_store import ChromaSemanticStore


class MockEmbedModel:

    def __init__(self):
        self.embeddings = {
            "我叫张明": [0.1, 0.9, 0.2, 0.8],
            "张明": [0.12, 0.88, 0.22, 0.78],
            "打篮球": [0.8, 0.2, 0.9, 0.1],
            "我喜欢运动": [0.75, 0.25, 0.85, 0.15],
            "今天天气好": [0.3, 0.4, 0.5, 0.6],
        }

    async def embed_queries(self, texts: List[str]) -> List[List[float]]:
        result = []
        for text in texts:
            if text in self.embeddings:
                result.append(self.embeddings[text])
            else:
                import random
                result.append([random.random() for _ in range(4)])
        return result


class TestChromaSemanticStore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.embed_model = MockEmbedModel()
        self.store = ChromaSemanticStore(
            persist_directory=self.temp_dir,
            embed_model=self.embed_model,
        )
        self.test_table = "test_table"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_docs_success(self):
        docs = [
            ("doc1", "我叫张明"),
            ("doc2", "打篮球"),
            ("doc3", "今天天气好")
        ]
        result = asyncio.run(self.store.add_docs(docs, self.test_table))
        self.assertTrue(result)

    def test_search_basic(self):
        docs = [
            ("doc1", "我叫张明"),
            ("doc2", "打篮球"),
            ("doc3", "今天天气好")
        ]
        asyncio.run(self.store.add_docs(docs, self.test_table))

        results = asyncio.run(self.store.search("张明", self.test_table, top_k=2))

        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)  # top_k=2

        for result in results:
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            self.assertIsInstance(result[0], str)  # ID
            self.assertIsInstance(result[1], float)  # Similarity score

    def test_search_with_top_k_limit(self):
        docs = [
            (f"doc{i}", f"文档内容 {i}") for i in range(10)
        ]
        asyncio.run(self.store.add_docs(docs, self.test_table))

        results = asyncio.run(self.store.search("文档", self.test_table, top_k=3))

        self.assertLessEqual(len(results), 3)

    def test_delete_docs(self):
        docs = [
            ("doc1", "我叫张明"),
            ("doc2", "打篮球"),
            ("doc3", "今天天气好")
        ]
        asyncio.run(self.store.add_docs(docs, self.test_table))

        delete_result = asyncio.run(self.store.delete_docs(["doc1", "doc2"], self.test_table))

        self.assertTrue(delete_result)

        search_results = asyncio.run(self.store.search("张明", self.test_table, top_k=5))
        self.assertIsInstance(search_results, list)

    def test_delete_table(self):
        docs = [("doc1", "测试内容")]
        asyncio.run(self.store.add_docs(docs, self.test_table))

        search_results = asyncio.run(self.store.search("测试", self.test_table, top_k=5))
        initial_count = len(search_results)
        self.assertTrue(initial_count > 0)

        delete_result = asyncio.run(self.store.delete_table(self.test_table))

        self.assertTrue(delete_result)
        search_results_after = asyncio.run(self.store.search("测试", self.test_table, top_k=5))
        self.assertEqual(len(search_results_after), 0)

    def test_similarity_scoring(self):
        docs = [
            ("similar_doc", "我叫张明"),
            ("dissimilar_doc", "打篮球")
        ]
        asyncio.run(self.store.add_docs(docs, self.test_table))

        results = asyncio.run(self.store.search("张明", self.test_table, top_k=5))
        result = {}
        for doc_id, similarity in results:
            result[doc_id] = similarity
        self.assertTrue(result["similar_doc"] > result["dissimilar_doc"])

    def test_empty_search(self):
        results = asyncio.run(self.store.search("不存在的内容", self.test_table, top_k=5))
        self.assertEqual(results, [])

    def test_delete_docs_empty_list(self):
        result = asyncio.run(self.store.delete_docs([], self.test_table))
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
