"""
重排序模块 - Cross-Encoder 精排 / LLM 打分 / 无重排
"""
import os
import logging
from typing import List, Tuple, Optional

from langchain_core.documents import Document

from core.config import settings

logger = logging.getLogger("qa.reranker")


class Reranker:
    """
    重排序器

    三种策略:
    1. cross_encoder → SentenceTransformer Cross-Encoder (准)
    2. llm           → 用 LLM 逐条打分 (慢)
    3. none          → 不重排 (快)
    """

    _SENTENCE_TRANSFORMER_AVAILABLE = True
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        _SENTENCE_TRANSFORMER_AVAILABLE = False
        CrossEncoder = None

    def __init__(self):
        self._cross_encoder = None

    def _load_cross_encoder(self):
        """加载 Cross-Encoder 模型"""
        if not self._SENTENCE_TRANSFORMER_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed")
        model_name = settings.LOCAL_RERANKER_MODEL
        self._cross_encoder = self.CrossEncoder(model_name)
        logger.info(f"Cross-Encoder loaded: {model_name}")

    # ──────── Cross-Encoder 重排 ────────

    def cross_encoder_rerank(
        self,
        query: str,
        docs_with_scores: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """用 Cross-Encoder 对 (query, doc) 联合打分"""
        if not docs_with_scores:
            return []

        if self._cross_encoder is None:
            try:
                self._load_cross_encoder()
            except Exception as e:
                logger.warning(f"Cross-Encoder load failed, skip rerank: {e}")
                self._cross_encoder = False  # 标记失败，下次不再尝试
                return docs_with_scores

        if self._cross_encoder is False:
            return docs_with_scores

        pairs = [(query, doc.page_content) for doc, _ in docs_with_scores]
        scores = self._cross_encoder.predict(pairs)

        # 合并排序
        results = zip(docs_with_scores, scores)
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return [(item[0][0], float(item[1])) for item in results]

    # ──────── LLM 重排 ────────

    async def llm_rerank(
        self,
        query: str,
        docs_with_scores: List[Tuple[Document, float]],
        llm,
    ) -> List[Tuple[Document, float]]:
        """用 LLM 对每个文档逐条打分（10 分制）"""
        if not docs_with_scores:
            return []

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_messages([
            ("system", """评估文档与查询的相关性，给出 0-10 分:
- 10: 完全匹配
- 7-9: 高度相关
- 4-6: 部分相关
- 1-3: 较弱相关
- 0: 不相关
只输出数字分数，不要解释。"""),
            ("human", "查询: {query}\n\n文档内容:\n{doc_content}\n\n相关性分数:"),
        ])

        chain = prompt | llm | StrOutputParser()
        new_scores = []

        for doc, _ in docs_with_scores:
            try:
                result = await chain.ainvoke({
                    "query": query,
                    "doc_content": doc.page_content[:500],
                })
                score = float(result.strip()) / 10.0
                new_scores.append(max(0.0, min(1.0, score)))
            except Exception:
                new_scores.append(0.0)

        results = sorted(
            zip(docs_with_scores, new_scores),
            key=lambda x: x[1], reverse=True,
        )
        return [(item[0][0], item[1]) for item in results]

    # ──────── 统一入口 ────────

    async def rerank(
        self,
        query: str,
        docs_with_scores: List[Tuple[Document, float]],
        top_k: int = 5,
        method: str = "cross_encoder",
        llm=None,
    ) -> List[Tuple[Document, float]]:
        """
        统一重排入口

        Args:
            query: 查询文本
            docs_with_scores: [(doc, score), ...]
            top_k: 返回条数
            method: "cross_encoder" | "llm" | "none"
            llm: method="llm" 时使用的 LLM 实例
        """
        if not docs_with_scores:
            return []

        if method == "cross_encoder":
            result = self.cross_encoder_rerank(query, docs_with_scores)
        elif method == "llm" and llm:
            result = await self.llm_rerank(query, docs_with_scores, llm)
        else:
            result = docs_with_scores

        return result[:top_k]
