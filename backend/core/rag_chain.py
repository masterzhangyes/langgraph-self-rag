"""
增强版 RAG 链核心逻辑模块
========================

本模块是 RAG 智能问答系统的核心处理链，实现了从文档处理、检索到对话生成的完整流程。

主要功能:
  1. 混合检索 —— BM25 关键词检索 + Dense 向量检索，通过 RRF 融合排序
  2. Cross-Encoder 重排序 —— 对初步检索结果进行二次精排
  3. Self-RAG Agent —— 基于 LangGraph 的自检式问答（查询扩展→检索→评估→生成→幻觉检测）
  4. 查询扩展 —— 将用户问题扩展为多个搜索角度
  5. 源文档引用 —— 返回回答所引用的来源文档及相关性分数
  6. 多知识库支持 —— 支持创建和管理多个独立的知识库
  7. 依赖注入 —— 懒加载 LLM/Embeddings/Evaluator/Reranker，避免启动时初始化所有资源

模块结构:
  - 提示词模板: 定义系统提示词（上下文改写、问答、通用对话）
  - 知识库管理: 创建/清空/查询知识库，向量存储和检索器的缓存管理
  - 文档处理: 加载各类文档（txt/pdf/csv/md/网页）并切分入库
  - 检索: 混合检索 + 重排序流程
  - 对话生成: 流式/非流式对话，支持简化模式和 Agent 模式
  - 辅助函数: 格式化、历史转换等工具函数
"""

import os
import asyncio
import shutil
import json
import logging
from typing import List, Optional, AsyncGenerator

# LangChain 向量存储和文档加载器
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    TextLoader,    # 加载纯文本文件（.txt, .md）
    PyPDFLoader,   # 加载 PDF 文档
    WebBaseLoader, # 加载网页内容
    CSVLoader,     # 加载 CSV 文件
)
# 文本分割器：将长文档切分为固定大小的块
from langchain_text_splitters import RecursiveCharacterTextSplitter
# LangChain 提示词模板和消息类型
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# 输出解析器：将 LLM 输出转为纯字符串
from langchain_core.output_parsers import StrOutputParser
# Document 类：LangChain 的统一文档表示
from langchain_core.documents import Document

# 全局配置
from core.config import settings
# 混合检索器（BM25 + Dense）
from services.retriever import HybridRetriever
# 依赖注入容器（懒加载 LLM/Embeddings 等）
from infrastructure.dependencies import get_deps

# 本模块的日志记录器
logger = logging.getLogger("qa.rag")


# ══════════════════════════════════════════════
# 懒加载全局对象
# ══════════════════════════════════════════════
# 以下对象在模块加载时创建，通过 get_deps() 按需获取底层资源，
# 避免启动时初始化所有组件（如 LLM、Embeddings），节省启动时间和内存。

# 全局文本分割器（无状态，可安全共享）
# 使用递归字符分割策略，按优先级依次尝试以下分隔符：
#   "\n\n" → 段落分隔  |  "\n" → 换行分隔  |  "。" → 中文句号
#   "." → 英文句号  |  " " → 空格分隔  |  "" → 最终回退（单字符切分）
# 当某一分隔符产生的块仍大于 chunk_size 时，自动尝试下一级分隔符
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,         # 每个文本块的最大字符数（默认 1000）
    chunk_overlap=settings.CHUNK_OVERLAP,   # 相邻块之间的重叠字符数（默认 200）
    separators=["\n\n", "\n", "。", ".", " ", ""],
)

# 全局知识库缓存
# 结构: {kb_name: {"vectorstore": Chroma, "retriever": HybridRetriever, "docs": List[str]}}
# 缓存已加载的知识库实例，避免重复初始化
# 当知识库内容更新时，需手动清除对应缓存（见 create_knowledge_base）
_kb_cache: dict = {}


# ══════════════════════════════════════════════
# 提示词模板
# ══════════════════════════════════════════════
# 定义系统中使用的所有 LLM 提示词，控制 LLM 的行为和输出格式。

# 上下文改写提示词：用于多轮对话中将指代性查询改写为独立查询
# 例如：历史中讨论了 "RAG"，用户问 "它有什么优势？"
# → 改写为 "RAG技术有什么优势？"
CONTEXTUALIZE_Q_SYSTEM = """你是一个聊天助手。根据聊天历史和上下文，将用户的最新问题改写为一个独立、完整的检索查询。
规则:
1. 如果指代了历史中的实体（如"它"、"那个"），请替换为明确名称
2. 如果问题已经独立完整，保持原样
3. 只输出改写后的问题，不要额外解释"""

# RAG 问答提示词：当检索到相关文档时使用
# 指导 LLM 基于检索到的上下文回答问题，并区分知识库信息和补充知识
QA_SYSTEM_TEMPLATE = """你是一个专业的智能问答助手，基于检索到的上下文信息回答用户问题。

回答规则:
1. 优先使用上下文信息，准确、简洁地回答问题
2. 如果上下文充分，请给出详细、结构化的回答
3. 如果上下文不完整但部分相关，基于已有信息回答并说明局限性
4. 如果上下文完全不包含答案，告知用户并给出建议
5. 使用列表、分点等方式提高可读性
6. 可以结合你的知识进行适当补充，但要区分"来自知识库"和"补充知识"

上下文信息:
{context}"""

# 通用对话提示词：当知识库为空或无检索结果时使用
# 此时 LLM 仅依赖自身知识回答，不进行 RAG 增强
GENERAL_SYSTEM = "你是一个友好、专业的智能助手，请清晰、准确地回答用户的问题。如果问题超出你的知识范围，请如实说明。"

# ══════════════════════════════════════════════
# 知识库管理
# ══════════════════════════════════════════════
# 以下函数负责知识库的创建、查询、清空等操作，
# 以及向量存储和检索器的懒加载和缓存管理。

def _get_kb_path(kb_name: str) -> str:
    """
    获取知识库的本地存储路径
    
    每个知识库对应一个独立的目录，目录名为 "chroma_{kb_name}"。
    ChromaDB 会将该知识库的向量数据持久化存储在此目录中。
    
    Args:
        kb_name: 知识库名称
    
    Returns:
        知识库目录的绝对路径
    """
    base = os.path.dirname(settings.VECTOR_DB_PATH) or "."
    return os.path.join(base, f"chroma_{kb_name}")


def get_all_kb_names() -> List[str]:
    """
    获取所有已创建的知识库名称
    
    通过扫描存储目录中所有以 "chroma_" 开头的子目录来发现知识库。
    只返回非空目录（即至少包含一个文件的知识库）。
    
    Returns:
        知识库名称列表（不包含 "chroma_" 前缀）
    """
    base = os.path.dirname(settings.VECTOR_DB_PATH) or "."
    names = []
    for item in os.listdir(base):
        # 只处理以 "chroma_" 开头的目录
        if item.startswith("chroma_") and os.path.isdir(os.path.join(base, item)):
            name = item.replace("chroma_", "")
            # 只返回非空目录（排除已清空但未删除目录的情况）
            if os.listdir(os.path.join(base, item)):
                names.append(name)
    return names


async def get_vectorstore(kb_name: str = "default") -> Optional[Chroma]:
    """
    获取指定知识库的 Chroma 向量存储实例（异步安全）
    
    采用懒加载 + 缓存策略：
      1. 先检查缓存，命中则直接返回
      2. 缓存未命中时，检查磁盘上是否存在该知识库
      3. 存在则初始化 Chroma 实例并缓存
      4. 不存在则返回 None
    
    注意: Chroma 的初始化是同步操作，使用 asyncio.to_thread() 包装
    避免阻塞事件循环。
    
    Args:
        kb_name: 知识库名称，默认为 "default"
    
    Returns:
        Chroma 向量存储实例，如果知识库不存在则返回 None
    """
    # 先从缓存获取，命中则直接返回
    if kb_name in _kb_cache and _kb_cache[kb_name].get("vectorstore"):
        return _kb_cache[kb_name]["vectorstore"]

    kb_path = _get_kb_path(kb_name)
    # 检查磁盘上是否存在该知识库目录且非空
    if os.path.exists(kb_path) and os.listdir(kb_path):
        deps = get_deps()

        # Chroma 初始化是同步的，用 asyncio.to_thread() 包装避免阻塞事件循环
        def _init_chroma():
            return Chroma(
                persist_directory=kb_path,
                embedding_function=deps.embeddings,
            )

        vs = await asyncio.to_thread(_init_chroma)
        # 写入缓存
        if kb_name not in _kb_cache:
            _kb_cache[kb_name] = {}
        _kb_cache[kb_name]["vectorstore"] = vs
        return vs
    return None


async def get_hybrid_retriever(kb_name: str = "default") -> Optional[HybridRetriever]:
    """
    获取指定知识库的混合检索器实例（异步安全）
    
    混合检索器结合了 BM25 关键词检索和 Dense 向量检索，
    通过 RRF（Reciprocal Rank Fusion）融合两路检索结果。
    
    懒加载策略:
      1. 先确保向量存储已加载
      2. 检查缓存中是否已有检索器
      3. 缓存未命中时，加载或构建 BM25 索引并缓存
    
    Args:
        kb_name: 知识库名称
    
    Returns:
        HybridRetriever 实例，如果知识库不存在则返回 None
    """
    # 先确保向量存储已加载
    vs = await get_vectorstore(kb_name)
    if not vs:
        return None

    # 检查缓存中是否已有检索器
    if kb_name in _kb_cache and _kb_cache[kb_name].get("retriever"):
        return _kb_cache[kb_name]["retriever"]

    # BM25 索引加载/构建也是同步操作，包装为异步
    def _init_retriever():
        retriever = HybridRetriever(vs)
        # 尝试加载已持久化的 BM25 索引，如果不存在则重新构建
        if not retriever.load_bm25_index():
            retriever.build_bm25_index()
        return retriever

    retriever = await asyncio.to_thread(_init_retriever)

    # 写入缓存
    if kb_name not in _kb_cache:
        _kb_cache[kb_name] = {}
    _kb_cache[kb_name]["retriever"] = retriever
    return retriever


# ══════════════════════════════════════════════
# 文档处理
# ══════════════════════════════════════════════
# 以下函数负责加载各类文档（txt/pdf/csv/md/网页），
# 将其切分为文本块并存入向量数据库。

def _load_document(file_path: str) -> List[Document]:
    """
    加载单个文档文件
    
    根据文件扩展名自动选择对应的加载器：
      - .txt / .md: 使用 TextLoader（UTF-8 编码）
      - .pdf: 使用 PyPDFLoader
      - .csv: 使用 CSVLoader
    
    加载后会自动为每个文档片段添加 source 元数据，
    记录该文档的原始文件名，便于后续引用展示。
    
    Args:
        file_path: 文档文件的绝对或相对路径
    
    Returns:
        加载后的 Document 列表
    
    Raises:
        ValueError: 不支持的文件格式
    """
    ext = os.path.splitext(file_path)[1].lower()
    # 根据扩展名映射对应的加载器
    loaders = {
        ".txt": lambda p: TextLoader(p, encoding="utf-8"),
        ".pdf": lambda p: PyPDFLoader(p),
        ".csv": lambda p: CSVLoader(p),
        ".md": lambda p: TextLoader(p, encoding="utf-8"),
    }

    if ext not in loaders:
        raise ValueError(f"不支持的文件格式: {ext}，支持: {list(loaders.keys())}")

    loader = loaders[ext](file_path)
    docs = loader.load()

    # 为每个文档片段添加来源元数据（如果加载器未自动设置）
    for doc in docs:
        if "source" not in doc.metadata:
            doc.metadata["source"] = os.path.basename(file_path)

    return docs


def load_web_pages(urls: List[str]) -> List[Document]:
    """
    加载网页内容
    
    逐个抓取指定 URL 的网页内容，并提取为 Document 列表。
    单个 URL 抓取失败不会影响其他 URL 的处理。
    
    Args:
        urls: 待抓取的网页 URL 列表
    
    Returns:
        抓取成功的 Document 列表
    """
    docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            loaded = loader.load()
            # 为每个文档片段设置来源为 URL
            for doc in loaded:
                doc.metadata["source"] = url
            docs.extend(loaded)
            logger.info(f"已抓取网页: {url} ({len(loaded)} 段)")
        except Exception as e:
            # 单个 URL 失败不影响其他 URL
            logger.warning(f"抓取失败 {url}: {e}")
    return docs


def create_knowledge_base(
    file_paths: List[str] = None,
    urls: List[str] = None,
    kb_name: str = "default",
) -> int:
    """
    创建或更新知识库
    
    完整的知识库构建流程:
      1. 加载文档（文件 + 网页）
      2. 使用 text_splitter 将文档切分为固定大小的文本块
      3. 将文本块向量化并存入 ChromaDB
      4. 刷新缓存并重建 BM25 索引
    
    如果知识库已存在，新文档会追加到现有内容中。
    
    Args:
        file_paths: 待导入的文件路径列表
        urls: 待抓取的网页 URL 列表
        kb_name: 知识库名称，默认为 "default"
    
    Returns:
        本次处理产生的文本块数量
    """
    all_docs = []

    # 加载本地文件
    if file_paths:
        for path in file_paths:
            try:
                docs = _load_document(path)
                all_docs.extend(docs)
                logger.info(f"已加载文件: {path} ({len(docs)} 段)")
            except Exception as e:
                logger.error(f"加载文件失败 {path}: {e}")

    # 加载网页
    if urls:
        all_docs.extend(load_web_pages(urls))

    if not all_docs:
        logger.warning("没有可处理的文档")
        return 0

    # 文档分割：将长文档切分为固定大小的文本块
    chunks = text_splitter.split_documents(all_docs)
    logger.info(f"文档分割完成: {len(all_docs)} 个文档 → {len(chunks)} 个文本块")

    # 存入 Chroma 向量数据库
    # from_documents 会自动调用 embedding 模型将文本块向量化
    deps = get_deps()
    kb_path = _get_kb_path(kb_name)
    Chroma.from_documents(
        documents=chunks,
        embedding=deps.embeddings,
        persist_directory=kb_path,
    )

    # 刷新缓存：删除旧缓存，重新初始化
    if kb_name in _kb_cache:
        del _kb_cache[kb_name]

    # 重建检索器缓存并构建 BM25 索引
    vs = Chroma(persist_directory=kb_path, embedding_function=deps.embeddings)
    retriever = HybridRetriever(vs)
    retriever.build_bm25_index()
    _kb_cache[kb_name] = {"vectorstore": vs, "retriever": retriever}

    logger.info(f"知识库 [{kb_name}] 已更新: {len(chunks)} 个文本块")
    return len(chunks)


def clear_knowledge_base(kb_name: str = "default"):
    """
    清空知识库
    
    删除知识库的本地存储目录和内存缓存。
    操作不可逆，清空后需重新导入文档。
    
    Args:
        kb_name: 知识库名称
    """
    kb_path = _get_kb_path(kb_name)
    if os.path.exists(kb_path):
        # 递归删除整个知识库目录
        shutil.rmtree(kb_path)
        logger.info(f"知识库 [{kb_name}] 已清空")

    # 清除内存缓存
    if kb_name in _kb_cache:
        del _kb_cache[kb_name]


def get_kb_stats(kb_name: str = "default") -> dict:
    """
    获取知识库统计信息（同步版本）
    
    通过扫描知识库目录中的 parquet 文件数量来估算文本块数量。
    用于健康检查等不需要加载完整知识库的场景。
    
    Args:
        kb_name: 知识库名称
    
    Returns:
        包含 name, status, chunk_count, document_count 的字典
    """
    kb_path = _get_kb_path(kb_name)
    # 目录不存在或为空 → 知识库为空
    if not os.path.exists(kb_path) or not os.listdir(kb_path):
        return {"name": kb_name, "status": "empty", "chunk_count": 0, "document_count": 0}

    # ChromaDB 内部使用 parquet 文件存储向量数据，
    # 通过统计 parquet 文件数量来近似估算 chunk 数量
    try:
        import glob
        parquet_files = glob.glob(os.path.join(kb_path, "**", "*.parquet"), recursive=True)
        chunk_count = len(parquet_files)
    except Exception:
        chunk_count = 0

    return {
        "name": kb_name,
        "status": "active" if chunk_count > 0 else "empty",
        "chunk_count": chunk_count,
        "document_count": 0,
    }


def get_all_kb_stats() -> List[dict]:
    """
    获取所有知识库的统计信息
    
    Returns:
        每个知识库的统计信息列表
    """
    names = get_all_kb_names()
    if not names:
        return []
    return [get_kb_stats(name) for name in names]


# ══════════════════════════════════════════════
# 检索
# ══════════════════════════════════════════════
# 检索流程: 混合检索（BM25 + Dense）→ Cross-Encoder 重排序 → 格式化输出

async def retrieve_with_rerank(
    query: str,
    kb_name: str = "default",
    top_k: int = None,
) -> List[dict]:
    """
    检索并重排序，返回带来源信息的文档列表
    
    完整的检索流程:
      1. 混合检索: 同时使用 BM25 和 Dense 向量检索，通过 RRF 融合排序
      2. 重排序: 使用 Cross-Encoder 模型对初步结果进行精排
      3. 格式化: 将结果转换为包含内容、来源、分数的字典列表
    
    Args:
        query: 用户查询文本
        kb_name: 目标知识库名称
        top_k: 最终返回的文档数量，默认使用配置中的 RERANK_TOP_K
    
    Returns:
        文档列表，每个文档包含:
          - content: 文本内容
          - source: 来源文件
          - relevance_score: 相关性分数
          - metadata: 其他元数据
    """
    # 获取混合检索器（包含 BM25 和 Dense 索引）
    retriever = await get_hybrid_retriever(kb_name)
    if not retriever:
        return []

    from infrastructure.dependencies import get_deps
    deps = get_deps()

    # Step 1: 混合检索
    # BM25 和 Chroma 搜索都是同步操作，用线程池包装避免阻塞事件循环
    def _search():
        return retriever.search(
            query,
            top_k=settings.RETRIEVAL_TOP_K,           # 初检索返回数量（默认 20）
            enable_hybrid=settings.ENABLE_HYBRID_RETRIEVAL,  # 是否启用混合检索
        )
    docs_with_scores = await asyncio.to_thread(_search)

    if not docs_with_scores:
        return []

    # Step 2: Cross-Encoder 重排序
    # 对初步检索结果进行二次精排，提升最终文档质量
    if settings.ENABLE_RERANKING:
        docs_with_scores = await deps.reranker.rerank(
            query, docs_with_scores,
            top_k=top_k or settings.RERANK_TOP_K,  # 重排序后保留数量（默认 5）
            method="cross_encoder",                  # 使用 Cross-Encoder 模型
        )

    # Step 3: 格式化结果
    # 将 Document 对象转换为字典，便于序列化和传输
    limit = top_k or settings.RERANK_TOP_K
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "未知来源"),
            "relevance_score": round(score, 4),  # 保留 4 位小数
            "metadata": doc.metadata,
        }
        for doc, score in docs_with_scores[:limit]
    ]


# ══════════════════════════════════════════════
# 对话生成
# ══════════════════════════════════════════════
# 提供流式和非流式两种对话模式，每种模式又分为「简化流程」和「Agent 流程」。
# - 简化流程: 上下文改写 → 检索 → 生成（无自反思）
# - Agent 流程: 查询扩展 → 检索 → 相关性评估 → 生成 → 幻觉检测（Self-RAG）

async def _simple_chat_stream(
    query: str,
    history: List[dict] = None,
    kb_name: str = "default",
    session_id: str = None,
) -> AsyncGenerator[str, None]:
    """
    简化版流式对话（不使用 Self-RAG Agent）
    
    流程: 上下文改写 → 混合检索 → 重排序 → 流式生成
    
    当知识库有检索结果时，使用 RAG 模式（基于检索文档回答）；
    当知识库为空或无检索结果时，降级为通用模式（直接调用 LLM）。
    
    对话完成后会自动将消息保存到数据库（如果提供了 session_id）。
    
    Args:
        query: 用户问题
        history: 对话历史
        kb_name: 目标知识库
        session_id: 会话 ID（用于保存消息记录）
    
    Yields:
        生成的文本块（逐块输出，实现流式效果）
    """
    chat_history = _format_history(history or [])
    deps = get_deps()

    # Step 1: 上下文改写（消解指代）
    # 多轮对话时，用户的问题可能包含指代词（如"它"、"那个"），
    # 需要结合历史将其改写为独立完整的查询
    if chat_history:
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_Q_SYSTEM),
            ("human", "聊天历史:\n{history}\n\n最新问题: {query}\n\n改写后的独立查询:"),
        ])
        chain = contextualize_prompt | deps.llm_no_stream | StrOutputParser()
        standalone_query = await chain.ainvoke({
            "history": _history_to_text(chat_history),
            "query": query,
        })
    else:
        # 无历史时直接使用原始查询
        standalone_query = query

    # Step 2: 检索 + 重排序
    sources = await retrieve_with_rerank(standalone_query, kb_name)

    if sources:
        # ── RAG 模式: 有检索结果，基于上下文回答 ──
        context = _format_context(sources)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="history"),  # 插入对话历史
            ("human", "{input}"),
        ])

        # 使用流式 LLM（deps.llm）进行生成
        chain = qa_prompt | deps.llm | StrOutputParser()

        full_answer = ""
        # 流式输出：逐块 yield，前端可实时显示
        async for chunk in chain.astream({
            "context": context,
            "history": chat_history[-6:],  # 最多取最近 6 条历史
            "input": standalone_query,
        }):
            full_answer += chunk
            yield chunk

        # 保存对话记录到数据库
        if session_id:
            try:
                from infrastructure.database import save_message as db_save
                await db_save(session_id, "user", query)
                await db_save(
                    session_id, "assistant", full_answer,
                    sources=json.dumps(sources[:5], ensure_ascii=False),
                )
            except Exception as e:
                logger.warning(f"保存消息失败: {e}")

    else:
        # ── 通用模式: 无检索结果，直接调用 LLM ──
        messages = [SystemMessage(content=GENERAL_SYSTEM)]
        messages.extend(chat_history[-10:])  # 取最近 10 条历史
        messages.append(HumanMessage(content=query))

        full_answer = ""
        async for chunk in deps.llm.astream(messages):
            full_answer += chunk.content
            yield chunk.content

        if session_id:
            try:
                from infrastructure.database import save_message as db_save
                await db_save(session_id, "user", query)
                await db_save(session_id, "assistant", full_answer)
            except Exception as e:
                logger.warning(f"保存消息失败: {e}")


async def _agent_chat_stream(
    query: str,
    history: List[dict] = None,
    kb_name: str = "default",
    session_id: str = None,
) -> AsyncGenerator[str, None]:
    """
    Self-RAG Agent 流式对话
    
    使用 Self-RAG Agent 进行深度问答，流程:
      查询扩展 → 检索 → 相关性评估 → 生成 → 幻觉检测
    
    由于 Agent 内部已完成完整的检索和评估流程，
    流式输出通过模拟逐词输出实现（Agent 本身不支持真正的流式）。
    
    容错机制: 如果 Agent 执行失败，自动降级到简化流程。
    
    Args:
        query: 用户问题
        history: 对话历史
        kb_name: 目标知识库
        session_id: 会话 ID
    
    Yields:
        生成的文本块
    """
    from core.agent import SelfRAGAgent

    deps = get_deps()
    chat_history = _format_history(history or [])

    # 上下文改写（与简化流程相同）
    if chat_history:
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_Q_SYSTEM),
            ("human", "聊天历史:\n{history}\n\n最新问题: {query}\n\n改写后的独立查询:"),
        ])
        chain = contextualize_prompt | deps.llm_no_stream | StrOutputParser()
        standalone_query = await chain.ainvoke({
            "history": _history_to_text(chat_history),
            "query": query,
        })
    else:
        standalone_query = query

    # 构建 Agent 回调函数
    # Agent 通过这些回调与外部交互（检索、生成），实现解耦设计
    async def retrieve_fn(q: str) -> List[dict]:
        """检索回调: 执行混合检索 + 重排序"""
        return await retrieve_with_rerank(q, kb_name)

    async def generate_fn(q: str, docs: List[dict]) -> str:
        """
        生成回调: 基于检索文档生成回答
        有文档时使用 RAG 模式，无文档时降级为通用模式
        """
        if docs:
            context = _format_context(docs)
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", QA_SYSTEM_TEMPLATE),
                ("human", "{input}"),
            ])
            chain = qa_prompt | deps.llm_no_stream | StrOutputParser()
            return await chain.ainvoke({"context": context, "input": q})
        else:
            result = await deps.llm_no_stream.ainvoke(
                [SystemMessage(content=GENERAL_SYSTEM), HumanMessage(content=q)]
            )
            return result.content

    # 创建 Self-RAG Agent 实例
    agent = SelfRAGAgent(
        llm=deps.llm_no_stream,
        retriever_fn=retrieve_fn,
        generate_fn=generate_fn,
        evaluator=deps.evaluator if settings.ENABLE_SELF_RAG else None,
    )

    try:
        # 运行 Agent（内部会执行完整的自检流程）
        result = await agent.run(standalone_query)
        answer = result.get("answer", "")
        sources = result.get("retrieved_docs", [])
    except Exception as e:
        # Agent 执行失败时，降级到简化流程
        logger.exception(f"Agent 执行失败，降级到简化流程: {e}")
        async for chunk in _simple_chat_stream(query, history, kb_name, session_id):
            yield chunk
        return

    # Agent 结果输出
    # 说明: Agent 内部通过 llm_no_stream 串行完成「查询扩展→检索→相关性评估→
    # 生成→幻觉检测」多个节点，因此无法直接透传 LLM 的 token 级流式。
    # 这里将完整回答按句/标点切分后逐块输出，制造平滑的"打字机"效果；
    # 如需真正 token 级流式，应让 generate 节点使用 streaming LLM 并通过
    # asyncio.Queue 推送 token（当前版本不引入该复杂度，仅如实降级为分块输出）。
    chunks = _chunk_for_streaming(answer)
    for i, chunk in enumerate(chunks):
        yield chunk
        # 回答较长时加极短停顿，避免一次性刷屏
        if len(chunks) > 4 and i % 2 == 1:
            await asyncio.sleep(0.015)

    # 保存对话记录
    if session_id:
        try:
            from infrastructure.database import save_message as db_save
            await db_save(session_id, "user", query)
            await db_save(
                session_id, "assistant", answer,
                sources=json.dumps(sources[:5], ensure_ascii=False) if sources else None,
            )
        except Exception as e:
            logger.warning(f"保存消息失败: {e}")


async def chat_stream(
    query: str,
    history: List[dict] = None,
    kb_name: str = "default",
    session_id: str = None,
) -> AsyncGenerator[str, None]:
    """
    流式对话统一入口
    
    根据配置和知识库状态自动选择对话模式:
      - ENABLE_SELF_RAG=True 且知识库就绪 → Self-RAG Agent 模式（深度检索）
      - 否则 → 简化流程（快速响应）
    
    前端通过 SSE 接收流式输出，特殊前缀用于状态提示:
      - "__THINKING__": 表示系统正在思考/检索，前端可显示加载动画
    
    Args:
        query: 用户问题
        history: 对话历史
        kb_name: 目标知识库
        session_id: 会话 ID
    
    Yields:
        生成的文本块（包含状态提示和实际回答）
    """
    if settings.ENABLE_SELF_RAG:
        retriever = await get_hybrid_retriever(kb_name)
        if retriever:
            # Agent 模式: 知识库已就绪，使用深度检索
            logger.info("使用 Self-RAG Agent 流式对话")
            yield "__THINKING__🔍 正在深度检索与推理中，请稍候..."
            async for chunk in _agent_chat_stream(query, history, kb_name, session_id):
                yield chunk
            return
        else:
            # 知识库未就绪，降级为简化流程
            logger.info("知识库未就绪，降级为简化流式对话")
            yield "__THINKING__📖 知识库未就绪，正在直接回答..."

    # 简化流程: 快速检索 + 生成
    yield "__THINKING__🔍 正在检索知识库..."
    async for chunk in _simple_chat_stream(query, history, kb_name, session_id):
        yield chunk


async def _simple_chat(
    query: str,
    history: List[dict] = None,
    kb_name: str = "default",
    session_id: str = None,
) -> dict:
    """
    简化版非流式对话
    
    流程与 _simple_chat_stream 相同，但一次性返回完整回答。
    适用于 API 调用或批量处理场景。
    
    Args:
        query: 用户问题
        history: 对话历史
        kb_name: 目标知识库
        session_id: 会话 ID
    
    Returns:
        包含 answer, sources, kb_status 的字典
    """
    chat_history = _format_history(history or [])
    deps = get_deps()

    # 上下文改写（多轮对话时消解指代）
    if chat_history:
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_Q_SYSTEM),
            ("human", "聊天历史:\n{history}\n\n最新问题: {query}\n\n改写后的独立查询:"),
        ])
        chain = contextualize_prompt | deps.llm_no_stream | StrOutputParser()
        standalone_query = await chain.ainvoke({
            "history": _history_to_text(chat_history),
            "query": query,
        })
    else:
        standalone_query = query

    # 检索 + 重排序
    sources = await retrieve_with_rerank(standalone_query, kb_name)

    if sources:
        # RAG 模式: 有检索结果
        context = _format_context(sources)
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        chain = qa_prompt | deps.llm_no_stream | StrOutputParser()
        answer = await chain.ainvoke({
            "context": context,
            "history": chat_history[-6:],
            "input": standalone_query,
        })
        kb_status = "active"
    else:
        # 通用模式: 无检索结果
        messages = [SystemMessage(content=GENERAL_SYSTEM)]
        messages.extend(chat_history[-10:])
        messages.append(HumanMessage(content=query))
        result = await deps.llm_no_stream.ainvoke(messages)
        answer = result.content
        kb_status = "empty"
        sources = []

    # 保存对话记录
    if session_id and answer:
        try:
            from infrastructure.database import save_message as db_save
            await db_save(session_id, "user", query)
            await db_save(
                session_id, "assistant", answer,
                sources=json.dumps(sources[:5], ensure_ascii=False) if sources else None,
            )
        except Exception as e:
            logger.warning(f"保存消息失败: {e}")

    return {"answer": answer, "sources": sources[:5], "kb_status": kb_status}


async def _agent_chat(
    query: str,
    history: List[dict] = None,
    kb_name: str = "default",
    session_id: str = None,
) -> dict:
    """
    Self-RAG Agent 非流式对话
    
    使用 Self-RAG Agent 进行深度问答，一次性返回完整回答。
    流程与 _agent_chat_stream 相同，但不进行流式输出。
    
    容错机制: Agent 执行失败时自动降级到简化流程。
    
    Args:
        query: 用户问题
        history: 对话历史
        kb_name: 目标知识库
        session_id: 会话 ID
    
    Returns:
        包含 answer, sources, kb_status 的字典
    """
    from core.agent import SelfRAGAgent

    deps = get_deps()
    chat_history = _format_history(history or [])

    # 上下文改写
    if chat_history:
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_Q_SYSTEM),
            ("human", "聊天历史:\n{history}\n\n最新问题: {query}\n\n改写后的独立查询:"),
        ])
        chain = contextualize_prompt | deps.llm_no_stream | StrOutputParser()
        standalone_query = await chain.ainvoke({
            "history": _history_to_text(chat_history),
            "query": query,
        })
    else:
        standalone_query = query

    # 构建 Agent 回调
    async def retrieve_fn(q: str) -> List[dict]:
        return await retrieve_with_rerank(q, kb_name)

    async def generate_fn(q: str, docs: List[dict]) -> str:
        if docs:
            context = _format_context(docs)
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", QA_SYSTEM_TEMPLATE),
                ("human", "{input}"),
            ])
            chain = qa_prompt | deps.llm_no_stream | StrOutputParser()
            return await chain.ainvoke({"context": context, "input": q})
        else:
            result = await deps.llm_no_stream.ainvoke(
                [SystemMessage(content=GENERAL_SYSTEM), HumanMessage(content=q)]
            )
            return result.content

    # 创建并运行 Agent
    agent = SelfRAGAgent(
        llm=deps.llm_no_stream,
        retriever_fn=retrieve_fn,
        generate_fn=generate_fn,
        evaluator=deps.evaluator if settings.ENABLE_SELF_RAG else None,
    )

    try:
        result = await agent.run(standalone_query)
        answer = result.get("answer", "")
        sources = result.get("retrieved_docs", [])
        kb_status = "active" if sources else "empty"
    except Exception as e:
        # Agent 失败时降级到简化流程
        logger.exception(f"Agent 执行失败，降级到简化流程: {e}")
        return await _simple_chat(query, history, kb_name, session_id)

    # 保存对话记录
    if session_id and answer:
        try:
            from infrastructure.database import save_message as db_save
            await db_save(session_id, "user", query)
            await db_save(
                session_id, "assistant", answer,
                sources=json.dumps(sources[:5], ensure_ascii=False) if sources else None,
            )
        except Exception as e:
            logger.warning(f"保存消息失败: {e}")

    return {"answer": answer, "sources": sources[:5], "kb_status": kb_status}


async def chat(
    query: str,
    history: List[dict] = None,
    kb_name: str = "default",
    session_id: str = None,
) -> dict:
    """
    非流式对话统一入口
    
    根据配置和知识库状态自动选择对话模式:
      - ENABLE_SELF_RAG=True 且知识库就绪 → Self-RAG Agent 模式
      - 否则 → 简化流程
    
    Args:
        query: 用户问题
        history: 对话历史
        kb_name: 目标知识库
        session_id: 会话 ID
    
    Returns:
        包含 answer, sources, kb_status 的字典
    """
    if settings.ENABLE_SELF_RAG:
        retriever = await get_hybrid_retriever(kb_name)
        if retriever:
            logger.info("使用 Self-RAG Agent 非流式对话")
            return await _agent_chat(query, history, kb_name, session_id)
        else:
            logger.info("知识库未就绪，降级为简化对话")

    return await _simple_chat(query, history, kb_name, session_id)


# ══════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════
# 以下函数提供格式转换和数据处理的工具方法，被对话生成流程调用。

def _format_context(sources: List[dict]) -> str:
    """
    格式化检索结果为 LLM 上下文
    
    将检索到的文档列表格式化为结构化的文本，作为 LLM 的参考上下文。
    每个文档包含来源标识、相关性分数和文本内容。
    
    输出格式示例:
        [来源 1] (相关度: 0.95)
        来源文件: document.pdf
        内容: ...
        
        ---
        
        [来源 2] (相关度: 0.87)
        ...
    
    Args:
        sources: 检索结果列表，每个元素包含 content, source, relevance_score
    
    Returns:
        格式化后的上下文字符串
    """
    parts = []
    for i, s in enumerate(sources):
        source = s.get("source", "未知来源")
        score = s.get("relevance_score", 0)
        content = s.get("content", "")
        parts.append(f"[来源 {i+1}] (相关度: {score:.2f})\n来源文件: {source}\n内容: {content}")
    # 用分隔符连接所有文档片段
    return "\n\n---\n\n".join(parts)


def _format_history(history) -> list:
    """
    将前端对话历史转换为 LangChain 消息格式
    
    前端传入的历史记录可能是字典列表或对象列表，
    本函数统一转换为 LangChain 的 HumanMessage/AIMessage 格式。
    
    截断策略: 只保留最近 MAX_HISTORY_ROUNDS * 2 条消息
    （一问一答算 2 条，所以乘以 2）
    
    Args:
        history: 对话历史列表，每条包含 role 和 content
    
    Returns:
        LangChain 消息对象列表
    """
    messages = []
    # 只取最近 N 轮对话（每轮 2 条：用户 + 助手）
    for msg in history[-settings.MAX_HISTORY_ROUNDS * 2:]:
        # 兼容对象和字典两种格式
        role = msg.role if hasattr(msg, 'role') else msg.get("role", "user")
        content = msg.content if hasattr(msg, 'content') else msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _history_to_text(history: list) -> str:
    """
    将 LangChain 消息列表转换为纯文本格式
    
    用于上下文改写时提供给 LLM 的历史信息。
    格式为:
        用户: ...
        助手: ...
        用户: ...
    
    截断策略: 只保留最近 6 条消息（约 3 轮对话）
    
    Args:
        history: LangChain 消息对象列表
    
    Returns:
        纯文本格式的对话历史
    """
    lines = []
    # 只取最近 6 条消息，避免提示词过长
    for msg in history[-6:]:
        if isinstance(msg, HumanMessage):
            lines.append(f"用户: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"助手: {msg.content}")
    return "\n".join(lines)


def _chunk_for_streaming(text: str, max_len: int = 40) -> List[str]:
    """
    将完整回答切分为适合逐块输出的片段（供 Agent 伪流式使用）。

    注意: 中文回答通常不含空格，若直接 answer.split() 会把整段当成一个词，
    导致"打字机"效果失效。本函数优先按句子结束标点切分，超长句再按
    逗号/分句细分，最后以 max_len 为兜底窗口做硬切分。

    Args:
        text: LLM 生成的完整回答
        max_len: 单个片段的最大字符数（兜底窗口）

    Returns:
        保持原顺序的文本片段列表
    """
    import re

    if not text:
        return []

    # 1. 先按句子结束标点切分（?<= 保留标点在句尾）
    pieces = re.split(r"(?<=[。！？!?；;\n])", text)

    chunks = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        # 短句直接作为一块
        if len(piece) <= max_len:
            chunks.append(piece)
            continue

        # 长句按逗号等次级停顿再细分
        sub_pieces = re.split(r"(?<=[，,、：:])", piece)
        buffer = ""
        for sub in sub_pieces:
            if not sub:
                continue
            if buffer and len(buffer) + len(sub) > max_len:
                chunks.append(buffer)
                buffer = sub
            else:
                buffer += sub
        if buffer:
            chunks.append(buffer)

    # 3. 兜底: 若单块仍超过 max_len（如超长无标点串），按字符窗口硬切
    final_chunks = []
    for chunk in chunks:
        while len(chunk) > max_len:
            final_chunks.append(chunk[:max_len])
            chunk = chunk[max_len:]
        if chunk:
            final_chunks.append(chunk)
    return final_chunks
