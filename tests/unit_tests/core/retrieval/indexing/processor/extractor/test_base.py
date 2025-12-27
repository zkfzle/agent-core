# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
提取器抽象基类测试用例
"""
import pytest
from unittest.mock import AsyncMock

from openjiuwen.core.retrieval.indexing.processor.extractor.base import Extractor
from openjiuwen.core.retrieval.common.document import TextChunk
from openjiuwen.core.retrieval.common.triple import Triple


class ConcreteExtractor(Extractor):
    """具体提取器实现，用于测试抽象基类"""

    async def extract(self, chunks, **kwargs):
        triples = []
        for chunk in chunks:
            # 简单的提取逻辑：从文本中提取一些三元组
            if "knows" in chunk.text:
                triples.append(Triple(
                    subject="Alice",
                    predicate="knows",
                    object="Bob",
                    metadata={"doc_id": chunk.doc_id},
                ))
        return triples


class TestExtractor:
    """提取器抽象基类测试"""

    @pytest.mark.asyncio
    async def test_extract(self):
        """测试提取"""
        extractor = ConcreteExtractor()
        chunks = [
            TextChunk(id_="1", text="Alice knows Bob", doc_id="doc_1"),
            TextChunk(id_="2", text="Some other text", doc_id="doc_1"),
        ]
        triples = await extractor.extract(chunks)
        assert len(triples) == 1
        assert triples[0].subject == "Alice"
        assert triples[0].predicate == "knows"
        assert triples[0].object == "Bob"

    @pytest.mark.asyncio
    async def test_process(self):
        """测试处理（实现 Processor 接口）"""
        extractor = ConcreteExtractor()
        chunks = [
            TextChunk(id_="1", text="Alice knows Bob", doc_id="doc_1"),
        ]
        triples = await extractor.process(chunks)
        assert len(triples) == 1

    def test_cannot_instantiate_abstract_class(self):
        """测试不能直接实例化抽象类"""
        with pytest.raises(TypeError):
            Extractor()

