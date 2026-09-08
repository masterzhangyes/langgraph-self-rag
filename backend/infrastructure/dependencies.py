"""
依赖注入容器 - 懒加载单例模式
管理 LLM、Embedding、Reranker、Evaluator 等重型对象的生命周期
"""
import logging
from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from core.config import settings

logger = logging.getLogger("qa.dependencies")

__slots__ = ("_llm", "_llm_no_stream", "_embeddings", "_evaluator", "_reranker")


class RAGDependencies:
    """RAG 依赖容器，所有重量级对象使用懒加载 (@property)"""

    __slots__ = ("_llm", "_llm_no_stream", "_embeddings", "_evaluator", "_reranker")

    def __init__(self):
        self._llm: Optional[ChatOpenAI] = None
        self._llm_no_stream: Optional[ChatOpenAI] = None
        self._embeddings = None
        self._evaluator = None
        self._reranker = None

    # ──────── LLM ────────

    @property
    def llm(self) -> ChatOpenAI:
        """流式 LLM（用于 SSE 推流对话）"""
        if self._llm is None:
            self._llm = self._build_llm(streaming=True)
        return self._llm

    @property
    def llm_no_stream(self) -> ChatOpenAI:
        """非流式 LLM（用于评估/上下文改写等需要完整文本的场景）"""
        if self._llm_no_stream is None:
            self._llm_no_stream = self._build_llm(streaming=False)
        return self._llm_no_stream

    def _build_llm(self, streaming: bool) -> ChatOpenAI:
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            streaming=streaming,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_BASE_URL,
        )

    # ──────── Embedding ────────

    @property
    def embeddings(self):
        """嵌入模型（云端或本地）"""
        if self._embeddings is None:
            if settings.USE_LOCAL_EMBEDDINGS:
                logger.info(f"Using local embeddings: {settings.LOCAL_EMBEDDING_MODEL}")
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=settings.LOCAL_EMBEDDING_MODEL,
                    model_kwargs={"device": settings.EMBEDDING_DEVICE},
                    encode_kwargs={"normalize_embeddings": True},
                )
            else:
                self._embeddings = OpenAIEmbeddings(
                    model=settings.EMBEDDING_MODEL,
                    openai_api_key=settings.OPENAI_API_KEY,
                    openai_api_base=settings.OPENAI_BASE_URL,
                )
        return self._embeddings

    # ──────── Evaluator ────────

    @property
    def evaluator(self):
        """RAG 评估器（使用非流式 LLM）"""
        if self._evaluator is None:
            from services.evaluation import RAGEvaluator
            self._evaluator = RAGEvaluator(self.llm_no_stream)
        return self._evaluator

    # ──────── Reranker ────────

    @property
    def reranker(self):
        """重排序器"""
        if self._reranker is None:
            from services.reranker import Reranker
            self._reranker = Reranker()
        return self._reranker


# 全局依赖单例
_deps_instance: Optional[RAGDependencies] = None


def get_deps() -> RAGDependencies:
    """获取全局依赖单例"""
    global _deps_instance
    if _deps_instance is None:
        _deps_instance = RAGDependencies()
    return _deps_instance
