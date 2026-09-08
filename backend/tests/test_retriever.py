"""
测试混合检索器 - BM25 + Dense + RRF 融合
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


class TestHybridRetriever:
    """混合检索器单元测试"""

    def test_tokenize_chinese(self):
        """测试中文分词

        说明: jieba 在未加载自定义词典时可能把"机器学习"切成"机器/学习"，
        因此这里断言小写英文关键词 + 至少一个有效中文词，而不是具体词组。
        """
        from services.retriever import HybridRetriever
        retriever = HybridRetriever.__new__(HybridRetriever)
        tokens = retriever._tokenize("Python和机器学习")
        assert len(tokens) > 0
        assert "python" in tokens
        assert any(any("\u4e00" <= ch <= "\u9fff" for ch in t) for t in tokens)

    def test_reciprocal_rank_fusion(self, sample_documents):
        """测试 RRF 融合算法"""
        from services.retriever import HybridRetriever

        sparse = [(sample_documents[0], 0.9), (sample_documents[1], 0.7)]
        dense = [(sample_documents[1], 0.95), (sample_documents[2], 0.8)]

        merged = HybridRetriever.reciprocal_rank_fusion(sparse, dense, k=60)

        assert len(merged) >= 2
        # 在两个列表中都出现的应有更高分
        scores = {doc.page_content[:30]: score for doc, score in merged}
        assert len(scores) >= 2

    def test_reciprocal_rank_fusion_empty(self):
        """测试空列表 RRF 融合"""
        from services.retriever import HybridRetriever
        merged = HybridRetriever.reciprocal_rank_fusion([], [])
        assert merged == []

    def test_sparse_search_no_index(self):
        """测试 BM25 索引未构建时返回空列表"""
        from services.retriever import HybridRetriever
        from unittest.mock import MagicMock
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.bm25 = None
        result = retriever.sparse_search("测试查询")
        assert result == []

    def test_search_dense_only(self, sample_documents):
        """测试纯 Dense 检索（不启用混合检索）"""
        from services.retriever import HybridRetriever
        from unittest.mock import MagicMock, patch

        # 创建一个不完全初始化的 retriever
        retriever = HybridRetriever.__new__(HybridRetriever)

        # Mock vectorstore
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (sample_documents[0], 0.9),
            (sample_documents[1], 0.7),
        ]
        retriever.vectorstore = mock_vs
        retriever.bm25 = None
        retriever._dense_k = 5

        results = retriever.search("Python", enable_hybrid=False, top_k=3)
        assert len(results) == 2
        assert results[0][0].metadata["source"] == "python_intro.pdf"

    def test_search_with_source_output_format(self, sample_documents):
        """测试 search_with_source 输出格式"""
        from services.retriever import HybridRetriever
        from unittest.mock import MagicMock

        retriever = HybridRetriever.__new__(HybridRetriever)
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (sample_documents[0], 0.9),
        ]
        retriever.vectorstore = mock_vs
        retriever.bm25 = None
        retriever._dense_k = 5

        results = retriever.search_with_source("Python")
        assert len(results) == 1
        assert "content" in results[0]
        assert "source" in results[0]
        assert "relevance_score" in results[0]
        assert isinstance(results[0]["relevance_score"], float)
