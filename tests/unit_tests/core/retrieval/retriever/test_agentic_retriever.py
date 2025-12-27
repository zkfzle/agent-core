# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Agentic 检索器测试用例
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from openjiuwen.core.retrieval.retriever.agentic_retriever import AgenticRetriever
from openjiuwen.core.retrieval.retriever.graph_retriever import GraphRetriever
from openjiuwen.core.retrieval.common.retrieval_result import RetrievalResult


@pytest.fixture
def mock_graph_retriever():
    """创建模拟图检索器"""
    retriever = AsyncMock(spec=GraphRetriever)
    retriever.retrieve = AsyncMock(return_value=[
        RetrievalResult(text="Result 1", score=0.9),
        RetrievalResult(text="Result 2", score=0.8),
    ])
    retriever.index_type = "hybrid"
    return retriever


@pytest.fixture
def mock_llm_client():
    """创建模拟 LLM 客户端"""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_llm_response():
    """创建模拟 LLM 响应"""
    response = MagicMock()
    response.content = "rewritten query"
    return response


class TestAgenticRetriever:
    """Agentic 检索器测试"""

    def test_init_success(self, mock_graph_retriever, mock_llm_client):
        """测试初始化成功"""
        retriever = AgenticRetriever(
            graph_retriever=mock_graph_retriever,
            llm_client=mock_llm_client,
            llm_model_name="test-model",
            max_iter=3,
            agent_topk=15,
        )
        assert retriever.graph_retriever == mock_graph_retriever
        assert retriever.llm == mock_llm_client
        assert retriever.llm_model_name == "test-model"
        assert retriever.max_iter == 3
        assert retriever.agent_topk == 15

    def test_init_with_defaults(self, mock_graph_retriever, mock_llm_client):
        """测试使用默认值初始化"""
        retriever = AgenticRetriever(
            graph_retriever=mock_graph_retriever,
            llm_client=mock_llm_client,
        )
        assert retriever.max_iter == 3
        assert retriever.agent_topk == 15

    def test_init_without_graph_retriever(self, mock_llm_client):
        """测试没有图检索器时初始化"""
        with pytest.raises(ValueError, match="graph_retriever is required"):
            AgenticRetriever(graph_retriever=None, llm_client=mock_llm_client)

    def test_init_without_llm_client(self, mock_graph_retriever):
        """测试没有 LLM 客户端时初始化"""
        with pytest.raises(ValueError, match="llm_client is required"):
            AgenticRetriever(graph_retriever=mock_graph_retriever, llm_client=None)

    @pytest.mark.asyncio
    async def test_retrieve_success_single_iteration(self, mock_graph_retriever, mock_llm_client):
        """测试检索成功（单次迭代）"""
        mock_graph_retriever.retrieve = AsyncMock(return_value=[
            RetrievalResult(text="Result 1", score=0.9),
        ])
        
        mock_response = MagicMock()
        mock_response.content = "original query"  # 改写后与原查询相同，应该停止
        mock_llm_client.ainvoke = AsyncMock(return_value=mock_response)
        
        retriever = AgenticRetriever(
            graph_retriever=mock_graph_retriever,
            llm_client=mock_llm_client,
            max_iter=3,
        )
        
        results = await retriever.retrieve("original query", top_k=5)
        assert len(results) == 1
        # 因为改写后与原查询相同，应该只迭代一次
        assert mock_graph_retriever.retrieve.call_count == 1

    @pytest.mark.asyncio
    async def test_retrieve_multiple_iterations(self, mock_graph_retriever, mock_llm_client):
        """测试检索（多次迭代）"""
        mock_graph_retriever.retrieve = AsyncMock(return_value=[
            RetrievalResult(text="Result 1", score=0.9),
        ])
        
        # 第一次改写返回新查询，第二次返回相同查询
        mock_responses = [
            MagicMock(content="rewritten query 1"),
            MagicMock(content="rewritten query 1"),  # 与原查询不同，但第二次相同
        ]
        mock_llm_client.ainvoke = AsyncMock(side_effect=mock_responses)
        
        retriever = AgenticRetriever(
            graph_retriever=mock_graph_retriever,
            llm_client=mock_llm_client,
            max_iter=3,
        )
        
        results = await retriever.retrieve("original query", top_k=5)
        assert len(results) == 1
        # 应该迭代 2 次（第一次 + 改写后一次）
        assert mock_graph_retriever.retrieve.call_count == 2

    @pytest.mark.asyncio
    async def test_retrieve_max_iterations(self, mock_graph_retriever, mock_llm_client):
        """测试检索达到最大迭代次数"""
        mock_graph_retriever.retrieve = AsyncMock(return_value=[
            RetrievalResult(text="Result 1", score=0.9),
        ])
        
        # 每次都返回不同的改写查询
        mock_responses = [
            MagicMock(content=f"rewritten query {i}")
            for i in range(3)
        ]
        mock_llm_client.ainvoke = AsyncMock(side_effect=mock_responses)
        
        retriever = AgenticRetriever(
            graph_retriever=mock_graph_retriever,
            llm_client=mock_llm_client,
            max_iter=3,
        )
        
        results = await retriever.retrieve("original query", top_k=5)
        assert len(results) == 1
        # 应该达到最大迭代次数
        assert mock_graph_retriever.retrieve.call_count == 3

    @pytest.mark.asyncio
    async def test_retrieve_without_top_k(self, mock_graph_retriever, mock_llm_client):
        """测试检索时缺少 top_k"""
        retriever = AgenticRetriever(
            graph_retriever=mock_graph_retriever,
            llm_client=mock_llm_client,
        )
        
        with pytest.raises(ValueError, match="top_k is required"):
            await retriever.retrieve("test query", top_k=None)

    @pytest.mark.asyncio
    async def test_retrieve_with_custom_mode(self, mock_graph_retriever, mock_llm_client):
        """测试使用自定义模式检索"""
        mock_graph_retriever.retrieve = AsyncMock(return_value=[
            RetrievalResult(text="Result 1", score=0.9),
        ])
        
        mock_response = MagicMock()
        mock_response.content = "original query"
        mock_llm_client.ainvoke = AsyncMock(return_value=mock_response)
        
        retriever = AgenticRetriever(
            graph_retriever=mock_graph_retriever,
            llm_client=mock_llm_client,
        )
        
        results = await retriever.retrieve("test query", top_k=5, mode="vector")
        assert len(results) == 1
        # 验证使用了指定的模式
        call_kwargs = mock_graph_retriever.retrieve.call_args[1]
        assert call_kwargs["mode"] == "vector"

    @pytest.mark.asyncio
    async def test_retrieve_with_score_threshold(self, mock_graph_retriever, mock_llm_client):
        """测试使用分数阈值检索"""
        mock_graph_retriever.retrieve = AsyncMock(return_value=[
            RetrievalResult(text="Result 1", score=0.9),
        ])
        
        mock_response = MagicMock()
        mock_response.content = "original query"
        mock_llm_client.ainvoke = AsyncMock(return_value=mock_response)
        
        retriever = AgenticRetriever(
            graph_retriever=mock_graph_retriever,
            llm_client=mock_llm_client,
        )
        
        results = await retriever.retrieve("test query", top_k=5, score_threshold=0.8)
        assert len(results) == 1
        # 验证传递了分数阈值
        call_kwargs = mock_graph_retriever.retrieve.call_args[1]
        assert call_kwargs["score_threshold"] == 0.8

    @pytest.mark.asyncio
    async def test_retrieve_fusion_multiple_results(self, mock_graph_retriever, mock_llm_client):
        """测试融合多个检索结果"""
        # 模拟多次迭代返回不同的结果
        mock_results = [
            [RetrievalResult(text="Result 1", score=0.9)],
            [RetrievalResult(text="Result 2", score=0.8)],
        ]
        mock_graph_retriever.retrieve = AsyncMock(side_effect=mock_results)
        
        mock_responses = [
            MagicMock(content="rewritten query"),
            MagicMock(content="rewritten query"),  # 第二次相同，停止
        ]
        mock_llm_client.ainvoke = AsyncMock(side_effect=mock_responses)
        
        retriever = AgenticRetriever(
            graph_retriever=mock_graph_retriever,
            llm_client=mock_llm_client,
            max_iter=3,
        )
        
        results = await retriever.retrieve("test query", top_k=5)
        # 应该融合多个结果
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_batch_retrieve(self, mock_graph_retriever, mock_llm_client):
        """测试批量检索"""
        mock_graph_retriever.retrieve = AsyncMock(return_value=[
            RetrievalResult(text="Result 1", score=0.9),
        ])
        
        mock_response = MagicMock()
        mock_response.content = "original query"
        mock_llm_client.ainvoke = AsyncMock(return_value=mock_response)
        
        retriever = AgenticRetriever(
            graph_retriever=mock_graph_retriever,
            llm_client=mock_llm_client,
        )
        
        queries = ["query 1", "query 2"]
        results_list = await retriever.batch_retrieve(queries, top_k=5)
        assert len(results_list) == 2
        assert len(results_list[0]) == 1
        assert len(results_list[1]) == 1

    @pytest.mark.asyncio
    async def test_close(self, mock_graph_retriever, mock_llm_client):
        """测试关闭检索器"""
        mock_graph_retriever.close = AsyncMock()
        
        retriever = AgenticRetriever(
            graph_retriever=mock_graph_retriever,
            llm_client=mock_llm_client,
        )
        
        await retriever.close()
        mock_graph_retriever.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_sync_close(self, mock_graph_retriever, mock_llm_client):
        """测试关闭检索器（同步 close 方法）"""
        mock_graph_retriever.close = MagicMock()
        
        retriever = AgenticRetriever(
            graph_retriever=mock_graph_retriever,
            llm_client=mock_llm_client,
        )
        
        await retriever.close()
        mock_graph_retriever.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_close_method(self, mock_llm_client):
        """测试关闭没有 close 方法的图检索器"""
        mock_graph = MagicMock()
        del mock_graph.close  # 移除 close 方法
        
        retriever = AgenticRetriever(
            graph_retriever=mock_graph,
            llm_client=mock_llm_client,
        )
        # 不应该抛出异常
        await retriever.close()

