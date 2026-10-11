"""
重排序模块 - Cross-Encoder 精排 / LLM 打分 / 无重排
====================================================

在 RAG 链路中的位置:
    混合检索(HybridRetriever) 召回 Top-N 候选
    → 【本模块】对候选精排，选出最终送入 LLM 的 Top-K 上下文
    → 生成答案

为什么需要重排（Rerank）:
    · 召回阶段为了效率用的是"双塔"结构 —— 查询和文档独立编码，
      语义表达精度有限（Bi-Encoder 只能算相似度，无法做词级交互）；
    · 精排阶段候选量小（通常 ≤ 20 条），可以负担"交叉"结构 ——
      Cross-Encoder 把 (query, doc) 拼接后整体过一遍 Transformer，
      注意力可以在两段文本间自由交互，精度显著更高。

模块依赖:
    · sentence-transformers   Cross-Encoder 实现（可选依赖，缺失时优雅降级）
    · langchain_core          LLM 打分路线的 Prompt/解析组件
"""
import os
import asyncio
import logging
from typing import List, Tuple, Optional

# Document: LangChain 标准文档载体（page_content + metadata）
from langchain_core.documents import Document

# settings: 全局配置（LOCAL_RERANKER_MODEL 指定本地精排模型名）
from core.config import settings

# 独立命名空间 "qa.reranker"，与 qa.retriever 等模块在日志中区分
logger = logging.getLogger("qa.reranker")


class Reranker:
    """
    重排序器

    三种策略:
    1. cross_encoder → SentenceTransformer Cross-Encoder (准)
    2. llm           → 用 LLM 逐条打分 (慢)
    3. none          → 不重排 (快)

    选型建议:
        · 有 GPU / 追求质量 → cross_encoder（一次前向批量打分，毫秒级）
        · 无本地模型但有多余 LLM 配额 → llm（每条一次 LLM 调用，秒级）
        · 追求延迟最低 → none（直接透传召回结果）
    """

    # ── 类级探测可选依赖 ──
    # import 失败不能发生在模块顶层（否则整个后端起不来），
    # 故在类定义时 try 一次并落标记，后续按标记走降级分支。
    # 注意：这是"类属性"而非实例属性 —— 所有实例共享探测结果，
    # 只需探测一次；CrossEncoder 也是类属性，成功时为真实类，失败时置 None。
    _SENTENCE_TRANSFORMER_AVAILABLE = True
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        _SENTENCE_TRANSFORMER_AVAILABLE = False
        CrossEncoder = None

    def __init__(self):
        # 三态延迟加载标记:
        #   None  → 尚未尝试加载（首次调用 cross_encoder_rerank 时触发）
        #   模型  → 加载成功，可直接使用
        #   False → 加载失败（依赖缺失/模型下载失败），后续调用直接跳过重排
        # 用 False 而非 None 表示"失败"，避免每次请求都重试耗时的加载
        self._cross_encoder = None

    def _load_cross_encoder(self):
        """
        加载 Cross-Encoder 模型（惰性，首次精排时触发）。

        抛异常而非内部捕获：让调用方决定降级策略
        （cross_encoder_rerank 捕获后置 False 标记并透传原结果）。
        首次加载会从 HuggingFace 下载模型权重到本地缓存。
        """
        if not self._SENTENCE_TRANSFORMER_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed")
        model_name = settings.LOCAL_RERANKER_MODEL
        # self.CrossEncoder 即类属性（探测时注入的类对象）
        self._cross_encoder = self.CrossEncoder(model_name)
        logger.info(f"Cross-Encoder loaded: {model_name}")

    # ──────── Cross-Encoder 重排 ────────

    def cross_encoder_rerank(
        self,
        query: str,
        docs_with_scores: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """
        用 Cross-Encoder 对 (query, doc) 联合打分。

        Args:
            query:            用户查询原文
            docs_with_scores: 召回阶段结果 [(doc, 召回分数), ...]

        Returns:
            [(doc, 精排分数), ...] 按精排分数降序。
            召回阶段的原始分数被丢弃 —— 精排分数直接取代之
            （精排模型的表达能力严格强于召回，保留旧分数无意义）。

        降级路径（任何一步失败都不抛错，透传原结果）:
            依赖未装 → 模型加载失败 → 跳过重排
        """
        # 空候选：无需精排
        if not docs_with_scores:
            return []

        # 惰性加载 + 失败标记（见 __init__ 注释的三态语义）
        if self._cross_encoder is None:
            try:
                self._load_cross_encoder()
            except Exception as e:
                logger.warning(f"Cross-Encoder load failed, skip rerank: {e}")
                self._cross_encoder = False  # 标记失败，下次不再尝试
                return docs_with_scores

        # 已确认加载失败：直接透传（保持调用方无感知的优雅降级）
        if self._cross_encoder is False:
            return docs_with_scores

        # 构造 (query, doc) 文本对，批量前向打分
        # Cross-Encoder 内部会拼接两段文本整体编码 → 输出每对的相关性分数
        pairs = [(query, doc.page_content) for doc, _ in docs_with_scores]
        scores = self._cross_encoder.predict(pairs)

        # 合并排序: zip 把 (原结果, 新分数) 对齐，
        # 按新分数降序；解包时取 item[0][0]（Document）搭配新分数
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
        """
        用 LLM 对每个文档逐条打分（10 分制）。

        适用场景: 无本地 Cross-Encoder 但有可用 LLM 的部署环境。
        代价: 每个候选文档一次 LLM 调用 —— 候选多时延迟与 token
        成本线性膨胀，故只适合小候选集（建议 ≤ 10 条）。

        Args:
            llm: LangChain 风格的聊天模型实例（需支持 ainvoke，
                由调用方注入，本模块不绑定具体厂商）。
        """
        if not docs_with_scores:
            return []

        # LCEL 链: prompt 模板 | llm | 输出解析器
        # （函数内导入：仅走此路线时才需要这两个组件）
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        # system 严格限定输出格式（只要数字），
        # 保证 float(result.strip()) 能稳定解析；
        # 分档标准给足锚点，降低 LLM 打分的随意性
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

        # ⚠️ 逐条串行调用（非并发）：候选间相互独立，理论上可 asyncio.gather
        #   并发提速；当前实现保守串行，避免触发 LLM 服务的并发限流
        for doc, _ in docs_with_scores:
            try:
                result = await chain.ainvoke({
                    "query": query,
                    # 截断到 500 字符：控制 token 成本，
                    # 判断相关性通常看开头即可
                    "doc_content": doc.page_content[:500],
                })
                # 10 分制 → [0, 1]，clamp 防御 LLM 输出越界数字
                score = float(result.strip()) / 10.0
                new_scores.append(max(0.0, min(1.0, score)))
            except Exception:
                # 单条打分失败（输出非数字/网络异常）记 0 分：
                # 宁可降权也不中断整次重排
                new_scores.append(0.0)

        # 按新分数降序，丢弃召回阶段旧分数
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
        统一重排入口（RAG 链路调用这里）。

        Args:
            query: 查询文本
            docs_with_scores: [(doc, score), ...] 召回阶段结果
            top_k: 返回条数（重排后截断）
            method: "cross_encoder" | "llm" | "none"
            llm: method="llm" 时使用的 LLM 实例

        路由逻辑:
            cross_encoder      → 本地模型精排（同步，内部无 IO 等待）
            llm 且传入了 llm   → LLM 逐条打分（异步）
            其他（none / 缺 llm）→ 不重排，原样透传

        注意: 本方法声明为 async 仅为统一调用方签名
             （调用方无需关心具体 method 是否异步）。
        """
        if not docs_with_scores:
            return []

        if method == "cross_encoder":
            # 🐛 修复: predict 是同步阻塞的模型推理（可达秒级），
            # 直接在 async 上下文调用会卡死整个事件循环 —— 放线程池执行
            result = await asyncio.to_thread(
                self.cross_encoder_rerank, query, docs_with_scores
            )
        elif method == "llm" and llm:
            result = await self.llm_rerank(query, docs_with_scores, llm)
        else:
            # "none"，或 method="llm" 但未注入 llm 实例 —— 透传
            result = docs_with_scores

        # 统一在出口截断 top_k，保证三条路径的返回形状一致
        return result[:top_k]
