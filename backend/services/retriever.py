"""
混合检索器 - BM25 关键词检索 + Chroma 向量检索 + RRF 融合
=========================================================

设计动机（为什么需要"混合"）:
    · 纯向量检索（Dense）擅长语义匹配 —— "怎么退货" ≈ "退款流程"，
      但对精确词（学号、课程代码、专有名词）容易漏召回；
    · 纯关键词检索（Sparse/BM25）擅长字面精确命中，但完全不理解同义改写；
    · 两者互补，用 RRF（Reciprocal Rank Fusion，排名倒数融合）
      把两路结果的排名合并为最终排序 —— 只依赖排名而非原始分数，
      天然规避了 BM25 分数与余弦相似度"量纲不一致"的问题。

模块依赖:
    · jieba          中文分词（BM25 的前置步骤，惰性导入）
    · rank_bm25      BM25Okapi 稀疏检索实现
    · vectorstore    LangChain 封装的 Chroma 客户端（由外部注入）
"""
import os
import pickle        # BM25 索引的磁盘序列化
import logging
import hashlib       # 由知识库目录生成缓存文件名的摘要
from typing import List, Tuple

# Document: LangChain 标准文档载体（page_content 正文 + metadata 元数据）
from langchain_core.documents import Document

# settings: 全局配置（RETRIEVAL_TOP_K 召回条数 / RERANK_TOP_K 最终返回条数）
from core.config import settings

# 独立命名空间 "qa.retriever"，便于在日志中区分检索器与其他模块
logger = logging.getLogger("qa.retriever")


class HybridRetriever:
    """
    混合检索器

    检索策略:
    1. Sparse (BM25): 关键词匹配，擅长精确词匹配
    2. Dense (Chroma): 语义匹配，擅长同义词/概念匹配
    3. RRF 融合: 排名倒数求和 fusion

    生命周期:
        实例化(注入 vectorstore) → load_bm25_index() 尝试读缓存
        → 未命中则 build_bm25_index() 从 Chroma 全量文档重建并写缓存
        → 之后 search() 即为无状态查询入口
    """

    def __init__(self, vectorstore):
        """
        Args:
            vectorstore: Chroma 向量库句柄（LangChain 风格）。
                        不在内部创建，由调用方（rag_chain）注入，
                        保证检索器与存储层解耦、便于测试时 mock。
        """
        self.vectorstore = vectorstore
        self.bm25 = None  # BM25Okapi 实例；None 表示索引未就绪（此时自动退化为纯向量检索）
        # 索引对应的文档快照（与 bm25 下标严格对齐）。
        # ⚠️ 不能在检索时实时 vectorstore.get()：外部传 docs 建索引时
        # 顺序可能与 Chroma 全量 get 不一致，导致分数张冠李戴；
        # 快照与索引一起 pickle，保证两者永远同步。
        self._bm25_docs: List[str] = None
        self._dense_k = settings.RETRIEVAL_TOP_K  # 两路检索各自的召回深度（融合前的候选数）

    # ──────── BM25 构建 / 缓存 ────────

    def _tokenize(self, text: str) -> list:
        """
        中文分词（jieba）。

        BM25 以"词"为统计单元，中文必须先分词（否则按字切会丢失词频语义）。
        统一 lower() 保证英文大小写不敏感。

        jieba 在函数内惰性导入：首次调用才加载词典（~1s），
        避免拖慢模块导入、也避免不使用 BM25 时白白占内存。
        """
        import jieba
        return list(jieba.cut(text.lower()))

    def _get_cache_path(self) -> str:
        """
        获取 BM25 索引缓存路径。

        策略: 缓存文件直接放在该知识库的 Chroma 持久化目录内，
        文件名含目录路径的 md5 前 12 位 —— 即使两个知识库目录同名，
        缓存也不会互相覆盖（不同 KB 实例各自隔离）。
        """
        # Chroma 底层持久化目录（如 backend/chroma_kb_xxx）；
        # hasattr 兜底：vectorstore 若无该属性（如内存库/测试替身）则退化为当前目录
        kb_dir = self.vectorstore._persist_directory if hasattr(self.vectorstore, '_persist_directory') else ""
        cache_key = hashlib.md5(kb_dir.encode()).hexdigest()[:12]
        return os.path.join(kb_dir, f"bm25_index_{cache_key}.pkl")

    def build_bm25_index(self, docs: List[Document] = None):
        """
        从全量文档构建 BM25 索引（并写磁盘缓存）。

        Args:
            docs: 可选。不传则自动从 Chroma 拉全量文档重建
                  （用于"缓存丢失后的冷启动"或"删除文档后的全量重建"）。
        """
        if docs is None:
            # 从 Chroma 获取全量文档
            # vectorstore.get() 返回 {"documents": [str, ...], ...}，
            # 需要手工包装回 Document 对象以保持类型一致
            docs = self.vectorstore.get()["documents"]
            docs = [Document(page_content=doc) for doc in docs]

        if not docs:
            # 空知识库：不建索引（bm25 保持 None，检索自动走纯向量）
            logger.warning("No documents to build BM25 index")
            return

        # 惰性导入：只有真正建索引时才依赖 rank_bm25
        from rank_bm25 import BM25Okapi
        # 语料库逐篇分词 → BM25 统计 TF/IDF 并计算各项参数
        tokenized = [self._tokenize(doc.page_content) for doc in docs]
        self.bm25 = BM25Okapi(tokenized)
        # 快照文档内容：顺序与索引下标一一对应（修复检索时顺序错位 bug）
        self._bm25_docs = [doc.page_content for doc in docs]

        # 缓存到磁盘：索引构建是 O(语料规模) 的重操作，
        # 序列化后下次启动 load_bm25_index() 可秒级恢复。
        # 索引与文档快照打包存 tuple，保证两者永不脱节
        cache_path = self._get_cache_path()
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump((self.bm25, self._bm25_docs), f)
        logger.info(f"BM25 index built ({len(docs)} docs) → {cache_path}")

    def load_bm25_index(self) -> bool:
        """
        从缓存加载 BM25 索引。

        Returns:
            True  = 命中缓存并成功恢复（免去全量重建）
            False = 缓存不存在（调用方应转而 build_bm25_index()）
        """
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            return False
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, tuple) and len(data) == 2:
            # 新版缓存: (BM25Okapi, 文档快照)
            self.bm25, self._bm25_docs = data
        else:
            # 旧版缓存只存了索引：文档快照缺失，标记为需要全量重建，
            # 否则 sparse_search 无法安全地把分数映射回文档
            logger.warning("BM25 cache is in legacy format, rebuild required")
            self.bm25 = None
            self._bm25_docs = None
            return False
        # 防御：快照与索引长度不一致说明缓存损坏，拒绝使用
        if self._bm25_docs is None or len(self._bm25_docs) != self.bm25.corpus_size:
            logger.warning("BM25 cache inconsistent (docs snapshot mismatch), rebuild required")
            self.bm25 = None
            self._bm25_docs = None
            return False
        logger.info(f"BM25 loaded from cache: {cache_path}")
        return True

    # ──────── 检索 ────────

    def sparse_search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        BM25 关键词检索（稀疏路）。

        Returns:
            [(Document, score), ...] 按分数降序，score 已归一化到 [0, 1]。
        """
        # 索引未就绪（空库 / 尚未构建 / 快照缺失）时返回空列表 ——
        # 上层 search() 会因 sparse 为空而自然退化为纯向量检索
        if self.bm25 is None or self._bm25_docs is None:
            return []

        k = k or self._dense_k
        # 查询与建库使用同一分词器，保证词表对齐
        tokenized_query = self._tokenize(query)
        # ⚠️ BM25Okapi.get_scores() 返回 ndarray，否则 if scores 会触发
        #   "The truth value of an array with more than one element is ambiguous"
        #   —— 故立即 .tolist() 转成 Python list
        scores = self.bm25.get_scores(tokenized_query).tolist()

        # 归一化到 [0, 1]
        # BM25 原始分数无上界（取决于语料），除以本次最大值做 min-max 归一；
        # max=0（查询词全未命中）时保持全 0，避免除零
        max_score = max(scores) if scores else 1.0
        if max_score > 0:
            scores = [s / max_score for s in scores]

        # 🐛 修复: 按分数取真正的 Top-K（原实现取的是"前 k 个下标"，
        # 即 Chroma 任意顺序的前 k 篇，BM25 排序形同虚设）
        docs = self._bm25_docs
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:min(k, len(docs))]
        results = [
            (Document(page_content=docs[i]), float(scores[i]))
            for i in order
        ]
        return results

    def dense_search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        Chroma 向量检索（稠密路）。

        流程: 查询文本 → Embedding 模型转向量 → Chroma 近邻搜索。
        similarity_search_with_relevance_scores 返回余弦相似度 [-1, 1]，
        线性映射到 [0, 1] 与稀疏路的量纲对齐（仅用于展示；
        RRF 融合本身只用排名，不受该映射影响）。
        """
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
        RRF 融合算法（Reciprocal Rank Fusion）

        对每个文档在两个列表中的排名取倒数求和:
            score(doc) = Σ 1 / (k + rank)     # rank 从 1 起算
        k=60 是经典参数（来自论文），防止极小排名产生极大分数 ——
        即平滑系数，让第 1 名与第 2 名的分差不会过于悬殊。

        特性:
            · 只用排名不用原始分数 → 两路检索器分数量纲天然解耦；
            · 一篇文档若同时被两路召回，分数叠加 → 双重命中者排名靠前
              （这正是混合检索的增益来源）。

        ⚠️ 文档去重键 = page_content 前 200 字符。
        同内容切片在两路结果中重复出现时视为同一文档；
        超过 200 字符才出现差异的重复切片会被误判为不同文档（可接受的近似）。
        """
        doc_scores = {}

        # 第一遍：累计每个文档的 RRF 分数
        # rank 从 0 起 enumerate，公式中 +1 转为从 1 起算
        for rank, (doc, _) in enumerate(sparse_results):
            doc_id = doc.page_content[:200]
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (k + rank + 1)

        for rank, (doc, _) in enumerate(dense_results):
            doc_id = doc.page_content[:200]
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (k + rank + 1)

        # 第二遍：合并去重，按 RRF 分数排序
        # 保持"首次出现"的 Document 对象（sparse 优先），同 ID 只保留一份
        seen = {}
        merged = []
        for results in [sparse_results, dense_results]:
            for doc, _ in results:
                doc_id = doc.page_content[:200]
                if doc_id not in seen:
                    seen[doc_id] = doc
                    # 丢弃两路各自的原始分数，统一替换为 RRF 融合分
                    merged.append((doc, doc_scores.get(doc_id, 0)))

        merged.sort(key=lambda x: x[1], reverse=True)
        return merged

    def search(
        self,
        query: str,
        top_k: int = None,
        enable_hybrid: bool = True,
    ) -> List[Tuple[Document, float]]:
        """
        统一检索入口（RAG 链路调用这里）。

        Args:
            query:         用户原始问题（不做任何预处理）
            top_k:         最终返回条数，默认取 RERANK_TOP_K
                           （注意与 _dense_k 的区别：后者是融合前
                            每一路的召回深度，应 ≥ top_k 才有融合意义）
            enable_hybrid: False 时跳过 BM25/RRR，纯向量检索
                           （如 BM25 未就绪或用户指定纯语义模式）

        决策逻辑:
            BM25 就绪 且 开混合 → 双路召回 + RRF 融合
            否则                → 仅向量检索
        """
        top_k = top_k or settings.RERANK_TOP_K
        dense_results = self.dense_search(query)

        if enable_hybrid and self.bm25 is not None:
            sparse_results = self.sparse_search(query)
            # 融合后再截断 top_k（先融合后截断，保证排名全局最优）
            return self.reciprocal_rank_fusion(sparse_results, dense_results)[:top_k]

        return dense_results[:top_k]

    def search_with_source(self, query: str, top_k: int = None) -> List[dict]:
        """
        检索并返回带来源信息的结果（供外部使用）。

        与 search() 的区别: 输出为纯 dict 列表（JSON 友好），
        附带 metadata.source（文档来源文件名/URL）与四位小数的
        相关度分数 —— 供前端"引用来源"面板或调试接口直接消费。
        """
        docs_with_scores = self.search(query, top_k=top_k)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "未知来源"),
                "relevance_score": round(score, 4),
            }
            for doc, score in docs_with_scores
        ]
