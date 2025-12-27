# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
文档解析器抽象基类测试用例
"""
import pytest
from unittest.mock import AsyncMock

from openjiuwen.core.retrieval.indexing.processor.parser.base import Parser
from openjiuwen.core.retrieval.common.document import Document


class ConcreteParser(Parser):
    """具体解析器实现，用于测试抽象基类"""

    async def _parse(self, file_path: str):
        return f"Content from {file_path}"

    def supports(self, doc: str) -> bool:
        return doc.endswith(".test")


class TestParser:
    """文档解析器抽象基类测试"""

    @pytest.mark.asyncio
    async def test_parse_success(self):
        """测试解析成功"""
        parser = ConcreteParser()
        documents = await parser.parse("test.txt", doc_id="doc_1")
        assert len(documents) == 1
        assert documents[0].id_ == "doc_1"
        assert "Content from test.txt" in documents[0].text

    @pytest.mark.asyncio
    async def test_parse_empty_content(self):
        """测试解析空内容"""
        class EmptyParser(Parser):
            async def _parse(self, file_path: str):
                return None

            def supports(self, doc: str) -> bool:
                return True

        parser = EmptyParser()
        documents = await parser.parse("test.txt")
        assert len(documents) == 0

    @pytest.mark.asyncio
    async def test_parse_with_kwargs(self):
        """测试解析时传递额外参数"""
        parser = ConcreteParser()
        documents = await parser.parse("test.txt", doc_id="doc_1", file_name="test")
        assert len(documents) == 1

    @pytest.mark.asyncio
    async def test_lazy_parse(self):
        """测试懒加载解析"""
        parser = ConcreteParser()
        docs = []
        async for doc in parser.lazy_parse("test.txt", doc_id="doc_1"):
            docs.append(doc)
        assert len(docs) == 1
        assert docs[0].id_ == "doc_1"

    @pytest.mark.asyncio
    async def test_process(self):
        """测试处理（实现 Processor 接口）"""
        parser = ConcreteParser()
        result = await parser.process("test.txt", doc_id="doc_1")
        assert len(result) == 1

    def test_supports(self):
        """测试支持检查"""
        parser = ConcreteParser()
        assert parser.supports("file.test") is True
        assert parser.supports("file.txt") is False

