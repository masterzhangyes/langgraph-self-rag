"""
测试重排序器 - Cross-Encoder 和 LLM 重排序
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.documents import Document


class TestReranker:
    """重排序器单元测试"""

    @pytest.fixture
    def reranker(self):
        from services.reranker import Reranker
        r = Reranker()
        r._cross_encoder = None
        return r

    @pytest.fixture
    def sample_docs_with_scores(self, sample_documents):
        return [(sample_documents[0], 0.9), (sample_documents[1], 0.7), (sample_documents[2], 0.5)]

    @pytest.mark.asyncio
    async def test_rerank_method_none(self, reranker, sample_docs_with_scores):
        """测试 method='none' 直接返回不重排"""
        result = await reranker.rerank("Python", sample_docs_with_scores, method="none", top_k=2)
        assert len(result) == 2
        assert result[0][0].metadata["source"] == "python_intro.pdf"

    @pytest.mark.asyncio
    async def test_rerank_empty_docs(self, reranker):
        """测试空文档列表"""
        result = await reranker.rerank("query", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_llm_rerank_output_format(self, reranker, sample_docs_with_scores):
        """测试 LLM 重排序输出格式（mock LLM）

        注意: MagicMock 实例会被 LangChain 判定为 callable 并包装成
        RunnableLambda（调用 mock(...) 而非 mock.ainvoke），因此这里直接
        提供一个异步 callable 模拟 LLM，按文档顺序返回分数。
        """
        score_queue = ["8.5", "6.0", "3.0"]

        async def fake_llm(_input) -> str:
            return score_queue.pop(0)

        result = await reranker.llm_rerank("Python", sample_docs_with_scores, fake_llm)
        assert len(result) == 3
        assert result[0][1] > result[-1][1]  # 降序排列

    def test_cross_encoder_load_failure(self, reranker, sample_docs_with_scores):
        """测试 Cross-Encoder 加载失败时的降级行为"""
        with patch.object(reranker, '_load_cross_encoder', side_effect=Exception("GPU not available")):
            reranker._cross_encoder = False
            result = reranker.cross_encoder_rerank("Python", sample_docs_with_scores)
            assert len(result) == len(sample_docs_with_scores)
