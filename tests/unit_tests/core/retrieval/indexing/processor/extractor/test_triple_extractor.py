# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
三元组提取器测试用例
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from openjiuwen.core.retrieval.indexing.processor.extractor.triple_extractor import TripleExtractor
from openjiuwen.core.retrieval.common.document import TextChunk
from openjiuwen.core.retrieval.common.triple import Triple


@pytest.fixture
def mock_llm_client():
    """创建模拟 LLM 客户端"""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_completion():
    """创建模拟完成对象"""
    completion = MagicMock()
    completion.content = json.dumps({
        "triples": [
            ["Alice", "knows", "Bob"],
            ["Bob", "works_at", "Company"],
        ]
    })
    return completion


class TestTripleExtractor:
    """三元组提取器测试"""

    def test_init(self, mock_llm_client):
        """测试初始化"""
        extractor = TripleExtractor(
            llm_client=mock_llm_client,
            model_name="test-model",
            temperature=0.0,
            max_concurrent=10,
        )
        assert extractor.llm_client == mock_llm_client
        assert extractor.model_name == "test-model"
        assert extractor.temperature == 0.0
        assert extractor.limiter._value == 10

    def test_init_with_defaults(self, mock_llm_client):
        """测试使用默认值初始化"""
        extractor = TripleExtractor(
            llm_client=mock_llm_client,
            model_name="test-model",
        )
        assert extractor.temperature == 0.0
        assert extractor.limiter._value == 50  # 默认值

    @pytest.mark.asyncio
    async def test_extract_multiple_chunks(self, mock_llm_client, mock_completion):
        """测试提取多个块"""
        mock_llm_client.ainvoke = AsyncMock(return_value=mock_completion)

        extractor = TripleExtractor(
            llm_client=mock_llm_client,
            model_name="test-model",
            max_concurrent=2,
        )
        chunks = [
            TextChunk(id_="1", text="Alice knows Bob", doc_id="doc_1"),
            TextChunk(id_="2", text="Charlie knows David", doc_id="doc_1"),
        ]
        triples = await extractor.extract(chunks)
        # 应该为每个块提取三元组
        assert mock_llm_client.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_extract_with_exception(self, mock_llm_client):
        """测试提取时发生异常"""
        mock_llm_client.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        extractor = TripleExtractor(
            llm_client=mock_llm_client,
            model_name="test-model",
        )
        chunks = [
            TextChunk(id_="1", text="Alice knows Bob", doc_id="doc_1"),
        ]
        # 应该捕获异常并返回空列表
        triples = await extractor.extract(chunks)
        assert len(triples) == 0

    @pytest.mark.asyncio
    async def test_extract_invalid_json(self, mock_llm_client):
        """测试无效的 JSON 响应"""
        mock_completion = MagicMock()
        mock_completion.content = "Invalid JSON response"
        mock_llm_client.ainvoke = AsyncMock(return_value=mock_completion)

        extractor = TripleExtractor(
            llm_client=mock_llm_client,
            model_name="test-model",
        )
        chunks = [
            TextChunk(id_="1", text="Alice knows Bob", doc_id="doc_1"),
        ]
        # 应该处理 JSON 解析错误
        triples = await extractor.extract(chunks)
        # 可能返回空列表或部分结果
        assert isinstance(triples, list)

    @pytest.mark.asyncio
    async def test_extract_empty_chunks(self, mock_llm_client):
        """测试提取空块列表"""
        extractor = TripleExtractor(
            llm_client=mock_llm_client,
            model_name="test-model",
        )
        triples = await extractor.extract([])
        assert len(triples) == 0
        mock_llm_client.ainvoke.assert_not_called()

    def test_build_prompt(self, mock_llm_client):
        """测试构建提示词"""
        extractor = TripleExtractor(
            llm_client=mock_llm_client,
            model_name="test-model",
        )
        prompt = extractor._build_prompt("Test passage", "Test Title")
        assert "Test passage" in prompt
        assert "Test Title" in prompt
        assert "triples" in prompt.lower() or "triple" in prompt.lower()

    def test_parse_triples_invalid_json(self, mock_llm_client):
        """测试解析无效的 JSON"""
        extractor = TripleExtractor(
            llm_client=mock_llm_client,
            model_name="test-model",
        )
        triples = extractor._parse_triples("Invalid JSON", "doc_1")
        assert len(triples) == 0

    def test_parse_triples_missing_triples_key(self, mock_llm_client):
        """测试缺少 triples 键的 JSON"""
        extractor = TripleExtractor(
            llm_client=mock_llm_client,
            model_name="test-model",
        )
        json_str = json.dumps({"other_key": "value"})
        triples = extractor._parse_triples(json_str, "doc_1")
        assert len(triples) == 0

