"""
测试 RAG 评估模块 - Faithfulness / Answer Relevancy / Context Relevancy
"""
import pytest
from unittest.mock import AsyncMock, MagicMock


class TestRAGEvaluator:
    """RAG 评估器单元测试"""

    @pytest.fixture
    def mock_llm(self):
        """mock LLM，返回固定分数"""
        llm = MagicMock()
        async def _ainvoke(input_data):
            return MagicMock(content="0.85")
        llm.ainvoke = AsyncMock(side_effect=_ainvoke)
        return llm

    @pytest.fixture
    def evaluator(self, mock_llm):
        from services.evaluation import RAGEvaluator
        return RAGEvaluator(mock_llm)

    @pytest.mark.asyncio
    async def test_faithfulness_with_context(self, evaluator):
        """测试忠实度评估 - 有上下文"""
        score = await evaluator.evaluate_faithfulness(
            "Python是AI领域最流行的语言。",
            ["Python 广泛应用于 AI 和数据科学领域。"]
        )
        assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_faithfulness_empty_context(self, evaluator):
        """测试忠实度评估 - 空上下文返回 0"""
        score = await evaluator.evaluate_faithfulness("some answer", [])
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_answer_relevancy(self, evaluator):
        """测试答案相关性评估"""
        score = await evaluator.evaluate_answer_relevancy(
            "什么是Python？", "Python是一门编程语言。"
        )
        assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_context_relevancy(self, evaluator):
        """测试上下文相关性评估"""
        score = await evaluator.evaluate_context_relevancy(
            "什么是RAG？",
            ["RAG 结合了检索和生成技术。", "Python 是一门编程语言。"]
        )
        assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_full_evaluation_structure(self, evaluator):
        """测试完整评估返回结构"""
        result = await evaluator.evaluate(
            question="什么是RAG？",
            answer="RAG是检索增强生成。",
            contexts=["RAG 结合了检索和生成技术。"],
            ground_truth="RAG是Retrieval-Augmented Generation的缩写。",
        )
        assert "question" in result
        assert "answer" in result
        assert "faithfulness" in result
        assert "answer_relevancy" in result
        assert "context_relevancy" in result
        assert "latency_ms" in result
        assert "ground_truth" in result

    @pytest.mark.asyncio
    async def test_batch_evaluate_aggregation(self, evaluator):
        """测试批量评估聚合结果（提供 rag_fn 模拟真实 API 调用）"""

        async def fake_rag_fn(q: str):
            return f"关于{q}的回答。", [f"{q} 相关上下文。"]

        result = await evaluator.batch_evaluate(
            questions=["Q1", "Q2", "Q3"],
            ground_truths=["A1", "A2", "A3"],
            rag_fn=fake_rag_fn,
        )
        assert "results" in result
        assert "avg_faithfulness" in result
        assert "avg_answer_relevancy" in result
        assert "avg_context_relevancy" in result
        assert len(result["results"]) == 3
        # 每个 question 都经过 rag_fn 生成过回答
        assert all(r["answer"].startswith("关于") for r in result["results"])
        # mock LLM 恒定返回 0.85，平均值应落在有效区间
        assert result["avg_answer_relevancy"] is not None
        assert 0 <= result["avg_answer_relevancy"] <= 1
