"""
混合检索器 - BM25 关键词检索 + Chroma 向量检索 + RRF 融合
"""
import os
import json
import pickle
import logging
import hashlib
from typing import List, Tuple, Optional

from langchain_core.documents import Document

from core.config import settings

logger = logging.getLogger("qa.retriever")


class HybridRetriever:
    """
    混合检索器

    检索策略:
    1. Sparse (BM25): 关键词匹配，擅长精确词匹配
    2. Dense (Chroma): 语义匹配，擅长同义词/概念匹配
    3. RRF 融合: 排名倒数求和 fusion
    """

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.bm25 = None
        self._dense_k = settings.RETRIEVAL_TOP_K

    # ──────── BM25 构建 / 缓存 ────────

    def _tokenize(self, text: str) -> list:
        """中文分词（jieba）"""
        import jieba
        return list(jieba.cut(text.lower()))

    def _get_cache_path(self) -> str:
        """获取 BM25 索引缓存路径"""
        kb_dir = self.vectorstore._persist_directory if hasattr(self.vectorstore, '_persist_directory') else ""
        cache_key = hashlib.md5(kb_dir.encode()).hexdigest()[:12]
        return os.path.join(kb_dir, f"bm25_index_{cache_key}.pkl")

    def build_bm25_index(self, docs: List[Document] = None):
        """从全量文档构建 BM25 索引"""
        if docs is None:
            # 从 Chroma 获取全量文档
            docs = self.vectorstore.get()["documents"]
            docs = [Document(page_content=doc) for doc in docs]

        if not docs:
            logger.warning("No documents to build BM25 index")
            return

        from rank_bm25 import BM25Okapi
        tokenized = [self._tokenize(doc.page_content) for doc in docs]
        self.bm25 = BM25Okapi(tokenized)

        # 缓存到磁盘
        cache_path = self._get_cache_path()
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(self.bm25, f)
        logger.info(f"BM25 index built ({len(docs)} docs) → {cache_path}")

    def load_bm25_index(self) -> bool:
        """从缓存加载 BM25 索引"""
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            return False
        with open(cache_path, "rb") as f:
            self.bm25 = pickle.load(f)
        logger.info(f"BM25 loaded from cache: {cache_path}")
        return True

    # ──────── 检索 ────────

    def sparse_search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """BM25 关键词检索"""
        if self.bm25 is None:
            return []

        k = k or self._dense_k
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 归一化到 [0, 1]
        max_score = max(scores) if scores else 1.0
        if max_score > 0:
            scores = [s / max_score for s in scores]

        # 获取所有文档
        docs = self.vectorstore.get()["documents"]
        results = [
            (Document(page_content=docs[i]), float(scores[i]))
            for i in range(min(k, len(docs)))
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def dense_search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Chroma 向量检索"""
        k = k or self._dense_k
        docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        # Cosine 相似度 [-1, 1] → [0, 1]
        return [(doc, (score + 1) / 2) for doc, score in docs_with_scores]

    @staticmethod
    def reciprocal_rank_fusion(
        sparse_results: List[Tuple[Document, float]],
        dense_results: List[Tuple[Document, float]],
        k: int = 60,
    ) -> List[Tuple[Document, float]]:
        """
        RRF 融合算法
        对每个文档在两个列表中的排名取倒数求和
        score = sum(1 / (k + rank) for each list)
        k=60 是经典参数（来自论文），防止极小排名产生极大分数
        """
        doc_scores = {}

        for rank, (doc, _) in enumerate(sparse_results):
            doc_id = doc.page_content[:200]
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (k + rank + 1)

        for rank, (doc, _) in enumerate(dense_results):
            doc_id = doc.page_content[:200]
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (k + rank + 1)

        # 合并去重，按 RRF 分数排序
        seen = {}
        merged = []
        for results in [sparse_results, dense_results]:
            for doc, _ in results:
                doc_id = doc.page_content[:200]
                if doc_id not in seen:
                    seen[doc_id] = doc
                    merged.append((doc, doc_scores.get(doc_id, 0)))

        merged.sort(key=lambda x: x[1], reverse=True)
        return merged

    def search(
        self,
        query: str,
        top_k: int = None,
        enable_hybrid: bool = True,
    ) -> List[Tuple[Document, float]]:
        """统一检索入口"""
        top_k = top_k or settings.RERANK_TOP_K
        dense_results = self.dense_search(query)

        if enable_hybrid and self.bm25 is not None:
            sparse_results = self.sparse_search(query)
            return self.reciprocal_rank_fusion(sparse_results, dense_results)[:top_k]

        return dense_results[:top_k]

    def search_with_source(self, query: str, top_k: int = None) -> List[dict]:
        """检索并返回带来源信息的结果（供外部使用）"""
        docs_with_scores = self.search(query, top_k=top_k)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "未知来源"),
                "relevance_score": round(score, 4),
            }
            for doc, score in docs_with_scores
        ]
