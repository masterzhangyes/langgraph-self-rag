"""
FastAPI 主入口 — 智能问答系统 v2.1
=====================================

应用架构:
    main.py               ── 路由定义 + 中间件注册（本文件）
    ├── api/              ── 表示层: 请求/响应 Pydantic 模型 + 中间件
    │   ├── models.py          ── ChatRequest / ChatResponse 等数据模型
    │   └── middleware.py      ── 请求日志 / 认证 / 异常处理中间件
    ├── core/             ── 核心引擎层: RAG 链 + Agent + 统一配置
    │   ├── config.py          ── Pydantic Settings，读取 .env / 环境变量
    │   ├── rag_chain.py       ── 知识库 CRUD / 检索 / 生成（LangChain 封装）
    │   └── agent.py           ── Self-RAG Agent（LangGraph 6 节点状态图）
    ├── services/         ── 业务服务层: 检索器 / 重排序 / 评估器
    │   ├── retriever.py       ── BM25 + Dense + RRF 混合检索
    │   ├── reranker.py        ── Cross-Encoder / LLM-based 重排序
    │   └── evaluation.py      ── Faithfulness / AnswerRelevancy / ContextRelevancy
    ├── infrastructure/   ── 基础设施层: 数据库连接池 / 依赖注入容器
    │   ├── database.py        ── aiosqlite 异步 SQLite 操作（连接池模式）
    │   └── dependencies.py    ── 单例依赖容器 get_deps()
    └── tests/            ── 单元测试: test_retriever / test_reranker / test_evaluation

启动方式:
    python main.py                 # 直接运行（开发模式，支持 --reload）
    uvicorn main:app --port 8000   # 生产部署

核心特性:
    · 混合检索（BM25 稀疏 + Dense 稠密向量 + RRF 融合排序）
    · BGE-Reranker Cross-Encoder 重排序 / LLM 打分降级
    · Self-RAG Agent（LangGraph: 6 节点 → 查询扩展→检索→评估→生成→幻觉检测）
    · 流式输出（SSE，ChatCompletion Chunk → 打字机效果）
    · 多知识库隔离（命名空间，独立 Chroma 集合）
    · 会话持久化（aiosqlite 异步连接池）
    · RAG 质量评估（3 项指标：忠实度 / 答案相关性 / 上下文相关性）
    · 速率限制（slowapi，基于客户端 IP）
    · 暗色/亮色双主题前端（偏好持久化 localStorage）
"""

import asyncio
import json
import os
import shutil
import uuid
import warnings
from contextlib import asynccontextmanager
from typing import List

# ── FastAPI 核心 ───────────────────────────────────────────
# FastAPI:        ASGI Web 框架，自动生成 OpenAPI 文档（/docs）
# UploadFile:     异步文件上传（流式写入磁盘，不占用内存）
# HTTPException:  抛出 HTTP 错误（自动序列化为 JSON）
# Request:        原始请求对象（中间件读取 path / headers / client IP）
# Depends:        依赖注入（FastAPI 原生 DI 系统）
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends

# ── 中间件 ─────────────────────────────────────────────────
# CORSMiddleware:     跨域资源共享（前端 localhost:5173 → 后端 localhost:8000）
# StreamingResponse:  流式响应（SSE 长连接）
# JSONResponse:       自定义 JSON 错误响应
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import ValidationError

# ── 内部模块 ───────────────────────────────────────────────
# settings:  统一配置单例（从 .env / 环境变量加载，Pydantic 校验）
from core.config import settings

# 请求/响应数据模型（Pydantic BaseModel，自动校验 & 序列化）
from api.models import (
    ChatRequest, ChatResponse, WebLoadRequest,
    KBStatsResponse, KBListResponse, UploadResponse,
    EvaluationRequest, EvaluationResponse,
)

# 中间件组件
from api.middleware import (
    RequestLogMiddleware,        # 请求日志（method / path / status / duration）
    ExceptionHandlerMiddleware,  # 全局异常捕获 → 统一 JSON 错误格式
    APIAuthMiddleware,           # API Key 认证（默认关闭）
    setup_logging,               # 日志初始化（控制台 + 文件）
)

# 依赖注入容器
from infrastructure.dependencies import get_deps, RAGDependencies


# ═══════════════════════════════════════════════════════════════
# 日志初始化
# ═══════════════════════════════════════════════════════════════

# 在应用启动前初始化日志系统：
# · 控制台输出（StreamHandler）— 开发时实时查看
# · 文件输出（RotatingFileHandler）— 生产环境持久化，自动轮转
setup_logging(settings.LOG_LEVEL, settings.LOG_PATH)

import logging
logger = logging.getLogger("qa.main")  # 本模块专用 logger，命名空间 "qa.main"


# ═══════════════════════════════════════════════════════════════
# 速率限制（可选）
# ═══════════════════════════════════════════════════════════════
#
# slowapi: 基于客户端 IP 的令牌桶算法限流
# · 默认限制: 30/分钟（从 settings.RATE_LIMIT 读取，可在 .env 修改）
# · 超过限制返回 HTTP 429 Too Many Requests
# · 如果 slowapi 未安装，优雅降级为无限流

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    # ⚠ 重要兼容性处理:
    # slowapi 的 Limiter 初始化时会用 starlette.Config 以「系统本地编码」
    # 直接 open() 读取当前目录 .env（未指定 encoding）。在中文 Windows
    # （GBK）上，若 .env 含 UTF-8 中文注释，会抛 UnicodeDecodeError，
    # 导致整个应用无法启动。
    # 规避方式: 显式传入一个不存在的配置文件路径，让 slowapi 跳过文件读取；
    # 限流规则由下方 default_limits 显式给出，并不依赖该文件内容。
    _DUMMY_SLOWAPI_ENV = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".slowapi-empty.env"
    )
    # starlette 对缺失配置文件的告警与实际功能无关，屏蔽以保持启动日志干净
    warnings.filterwarnings("ignore", message=r"Config file .* not found\.")

    limiter = Limiter(
        key_func=get_remote_address,             # 以客户端 IP 为限流键
        default_limits=[settings.RATE_LIMIT],    # 如 "30/minute"
        config_filename=_DUMMY_SLOWAPI_ENV,
    )
    _HAS_RATE_LIMIT = True
except ImportError:
    limiter = None
    _HAS_RATE_LIMIT = False
    logger.warning("slowapi 未安装，速率限制已禁用")


# ═══════════════════════════════════════════════════════════════
# 应用生命周期管理
# ═══════════════════════════════════════════════════════════════
#
# lifespan: FastAPI 的生命周期上下文管理器
# · 启动时（yield 前）:  打印配置摘要 → 初始化 SQLite 连接池
# · 关闭时（yield 后）:  关闭数据库连接池
#
# 替代了旧版 @app.on_event("startup") / @app.on_event("shutdown")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器。

    ── 启动阶段（yield 前）──
    1. 打印运行时配置（模型名 / 混合检索 / 重排序 / Self-RAG / 速率限制开关状态）
    2. 调用 init_db() 初始化 aiosqlite 连接池（创建 chat.db + 建表）

    ── 关闭阶段（yield 后）──
    1. 调用 _close_pool() 关闭数据库连接池，释放文件句柄
    """
    # ── 启动日志 ──
    logger.info(f"智能问答系统 v2.1 启动中...")
    logger.info(f"模型: {settings.LLM_MODEL}")
    logger.info(f"混合检索: {'启用' if settings.ENABLE_HYBRID_RETRIEVAL else '禁用'}")
    logger.info(f"重排序: {'启用' if settings.ENABLE_RERANKING else '禁用'}")
    logger.info(f"Self-RAG: {'启用' if settings.ENABLE_SELF_RAG else '禁用'}")
    logger.info(f"速率限制: {settings.RATE_LIMIT if _HAS_RATE_LIMIT else '禁用'}")

    # 初始化数据库（异步建表 + 连接池预热）
    from infrastructure.database import init_db
    await init_db()

    yield  # ← 应用运行期间暂停在此处

    # ── 关闭清理 ──
    from infrastructure.database import _close_pool
    await _close_pool()
    logger.info("服务关闭")


# ═══════════════════════════════════════════════════════════════
# FastAPI 应用实例
# ═══════════════════════════════════════════════════════════════
#
# title / version / description → 自动注入 OpenAPI 文档（/docs 页面）
# lifespan → 启动/关闭钩子（见上方）

app = FastAPI(
    title="智能问答系统",
    version="2.1.0",
    description="基于 LangChain RAG 的企业级智能问答系统",
    lifespan=lifespan,
)

# 速率限制: 将 limiter 挂载到 app.state，并注册 429 异常处理器
if _HAS_RATE_LIMIT:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ═══════════════════════════════════════════════════════════════
# 中间件注册
# ═══════════════════════════════════════════════════════════════
#
# FastAPI 中间件执行顺序（后注册 → 外层的洋葱模型）:
#   请求 → CORSMiddleware → APIAuthMiddleware → ExceptionHandlerMiddleware
#       → RequestLogMiddleware → 路由处理函数
#   ← 路由处理函数 → RequestLogMiddleware → ExceptionHandlerMiddleware
#       → APIAuthMiddleware → CORSMiddleware → 响应

# 1. CORS 跨域 — 允许前端 (localhost:5173) 访问后端 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,  # 如 ["*"] 或 ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],                     # GET / POST / DELETE / OPTIONS 全部放行
    allow_headers=["*"],                     # Content-Type / Authorization 等全部放行
)

# 2. API 认证 — 默认关闭（API_AUTH_ENABLED=false）
#    开启后需要在请求头携带 X-API-Key
app.add_middleware(APIAuthMiddleware)

# 3. 异常处理 — 捕获所有未处理异常 → 统一 JSON 响应 {"detail": "...", "error_code": "..."}
app.add_middleware(ExceptionHandlerMiddleware)

# 4. 请求日志 — 记录每个请求的 method / path / status / duration
app.add_middleware(RequestLogMiddleware)


# ── 文件上传目录 ───────────────────────────────────────────
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 自定义异常处理
# ═══════════════════════════════════════════════════════════════
#
# 捕获 Pydantic ValidationError（请求体校验失败）→ 422 JSON 响应
# 区别于 ExceptionHandlerMiddleware 捕获的通用异常（→ 500）

@app.exception_handler(ValidationError)
async def validation_handler(request: Request, exc: ValidationError):
    """请求体校验失败时返回 422 + 详细错误信息"""
    return JSONResponse(
        status_code=422,
        content={
            "detail": str(exc.errors()),         # 如 [{"loc": ["query"], "msg": "field required"}]
            "error_code": "VALIDATION_ERROR",
        },
    )


# ═══════════════════════════════════════════════════════════════
# 会话管理 API
# ═══════════════════════════════════════════════════════════════
#
# RESTful 设计:
#   GET    /api/sessions          → 列出所有会话
#   POST   /api/sessions          → 创建新会话
#   GET    /api/sessions/{id}     → 查看会话详情（含历史消息）
#   DELETE /api/sessions/{id}     → 删除会话
#
# 数据存储: aiosqlite（SQLite 异步驱动）
#   表 sessions:  id, title, created_at, updated_at
#   表 messages:  id, session_id, role(user/assistant), content, created_at


@app.get("/api/sessions")
async def list_sessions():
    """
    获取所有会话列表。

    GET /api/sessions

    返回:
        {
            "sessions": [
                {"id": "uuid", "title": "关于...", "created_at": "...", "updated_at": "..."},
                ...
            ]
        }
    """
    from infrastructure.database import get_sessions
    return {"sessions": await get_sessions()}


@app.post("/api/sessions")
async def new_session():
    """
    创建新会话。

    POST /api/sessions  （无请求体）

    返回:
        {"id": "uuid", "title": "新对话", "created_at": "..."}

    说明:
        会话默认标题为"新对话"，可在后续对话中通过首条用户消息自动更新。
    """
    from infrastructure.database import create_session
    return await create_session()


@app.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """
    获取会话详情（含完整历史消息）。

    GET /api/sessions/{session_id}

    返回:
        {
            "id": "uuid",
            "title": "...",
            "created_at": "...",
            "updated_at": "...",
            "messages": [
                {"id": ..., "role": "user", "content": "你好", "created_at": "..."},
                {"id": ..., "role": "assistant", "content": "你好！...", "created_at": "..."},
            ]
        }

    错误:
        404 → 会话不存在
    """
    from infrastructure.database import get_session, get_messages

    # 先查会话基本信息
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 再查该会话的所有消息，合并返回
    messages = await get_messages(session_id)
    session["messages"] = messages
    return session


@app.delete("/api/sessions/{session_id}")
async def remove_session(session_id: str):
    """
    删除会话（级联删除所有关联消息）。

    DELETE /api/sessions/{session_id}

    返回:
        {"message": "会话已删除"}

    说明:
        SQLite 表设置了 FOREIGN KEY ... ON DELETE CASCADE，
        删除会话时 messages 表中关联记录自动清除。
    """
    from infrastructure.database import delete_session
    await delete_session(session_id)
    return {"message": "会话已删除"}


# ═══════════════════════════════════════════════════════════════
# 对话 API
# ═══════════════════════════════════════════════════════════════
#
# 两条路由共享同一 ChatRequest 模型:
#   POST /api/chat          → 非流式：完整响应 + 引用来源
#   POST /api/chat/stream   → SSE 流式：逐 token 推送（打字机效果）


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    """
    非流式对话 — 一次返回完整答案 + 引用来源。

    POST /api/chat
    Body: {
        "query": "什么是 RAG？",           // 必填: 用户问题
        "session_id": "uuid",            // 可选: 不传则自动创建新会话
        "kb_names": ["default"],         // 可选: 知识库名称列表
        "history": [                     // 可选: 历史对话（多轮上下文）
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "stream": false                  // 非流式模式
    }

    返回:
        {
            "answer": "RAG（检索增强生成）是...",
            "session_id": "uuid",
            "sources": [
                {"content": "...", "source": "doc.pdf", "score": 0.92}
            ],
            "kb_status": {"status": "ready", "chunk_count": 150}
        }

    处理流程:
        1. session_id 为空 → 自动创建新会话
        2. kb_names 为空 → 默认使用 "default" 知识库
        3. 调用 core.rag_chain.chat() → 检索 → 生成 → 返回
        4. 异常时记录完整堆栈 → 返回 500（不泄露内部细节）
    """
    try:
        from core.rag_chain import chat
        from infrastructure.database import create_session

        # 自动创建会话（如果未指定）
        session_id = req.session_id
        if not session_id:
            session_id = (await create_session())["id"]

        # 调用 RAG 对话引擎
        result = await chat(
            req.query,
            req.history,
            kb_name=(req.kb_names[0] if req.kb_names else "default"),
            session_id=session_id,
        )

        # 组装响应
        return ChatResponse(
            answer=result["answer"],
            kb_status=result["kb_status"],
            sources=result.get("sources", []),
            session_id=session_id,
        )
    except Exception:
        # 记录完整堆栈（开发排查用），前端只收到 500 通用错误
        logger.exception("对话失败")
        raise HTTPException(status_code=500, detail="服务器内部错误，请稍后重试")


@app.post("/api/chat/stream")
async def api_chat_stream(req: ChatRequest):
    """
    流式对话 — SSE (Server-Sent Events) 逐 token 推送。

    POST /api/chat/stream
    Body: （同上，但 stream 字段忽略，此路由始终流式）

    SSE 事件格式:
        event: session          → 首先返回 session_id
        data: "你"              → 逐 token 推送（每行以 "data: " 开头）
        data: "好"              → ...
        data: [DONE]            → 流结束信号

    错误时:
        event: error
        data: "错误信息"

    HTTP 响应头:
        Content-Type: text/event-stream   → 浏览器知道这是 SSE 流
        Cache-Control: no-cache           → 禁止缓存（中间代理/浏览器）
        Connection: keep-alive            → 长连接
        X-Accel-Buffering: no             → 禁用 Nginx 缓冲（生产部署需要）
    """
    from core.rag_chain import chat_stream
    from infrastructure.database import create_session

    # 自动创建会话
    session_id = req.session_id
    if not session_id:
        session_id = (await create_session())["id"]

    async def event_generator():
        """
        异步生成器 — SSE 事件流。

        每次 yield 一个 SSE 格式的字符串:
            · "event: <type>\ndata: <payload>\n\n"
            · "data: <chunk>\n\n"

        FastAPI StreamingResponse 会持续迭代此生成器，
        直到生成器结束（return）或客户端断开连接。
        """
        try:
            # ── 第一条事件: 告知前端 session_id ──
            yield f"event: session\ndata: {session_id}\n\n"

            # ── 逐 token 推送 ──
            async for chunk in chat_stream(
                req.query, req.history,
                kb_name=(req.kb_names[0] if req.kb_names else "default"),
                session_id=session_id,
            ):
                # 每个 chunk 是一个字符串（LLM 生成的文本片段）
                # 使用 JSON 编码传输，避免 token 内含换行符破坏 SSE 行协议
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            # ── 流结束信号: 前端收到后关闭 EventSource ──
            yield "data: [DONE]\n\n"

        except asyncio.CancelledError:
            # 客户端断开连接（浏览器关闭 / 用户中止）：不补发错误事件，
            # 直接向 Starlette 传递取消信号以完成清理。
            raise
        except Exception as e:
            logger.exception("流式对话失败")
            try:
                yield f"event: error\ndata: {str(e)}\n\n"
            except Exception:
                # 发送错误事件期间流可能已被客户端关闭，忽略即可
                pass

    # 返回 SSE 流式响应
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # Nginx 禁用缓冲，否则会等到流结束才发送
        },
    )


# ═══════════════════════════════════════════════════════════════
# 知识库管理 API
# ═══════════════════════════════════════════════════════════════
#
# 知识库 = 命名空间 + Chroma 向量集合 + 文件存储
#   每个知识库独立建索引，查询时可指定 kb_name 切换


@app.get("/api/kb/list", response_model=KBListResponse)
async def api_kb_list():
    """
    获取所有知识库列表（含统计信息）。

    GET /api/kb/list

    返回:
        {
            "knowledge_bases": [
                {"name": "default", "status": "ready", "chunk_count": 150, "document_count": 3},
                {"name": "技术文档", "status": "ready", "chunk_count": 300, "document_count": 5},
            ]
        }

    说明:
        status 取值:
            "empty"    → 知识库存在但无数据
            "loading"  → 正在索引文档
            "ready"    → 可正常查询
            "error"    → 索引出错
    """
    from core.rag_chain import get_all_kb_stats
    return KBListResponse(knowledge_bases=get_all_kb_stats())


@app.get("/api/kb/{kb_name}/stats", response_model=KBStatsResponse)
async def api_kb_stats(kb_name: str = "default"):
    """
    获取指定知识库的详细统计。

    GET /api/kb/{kb_name}/stats?kb_name=技术文档

    返回:
        {"name": "技术文档", "status": "ready", "chunk_count": 300, "document_count": 5}
    """
    from core.rag_chain import get_kb_stats
    return get_kb_stats(kb_name)


@app.get("/api/kb/stats", response_model=KBStatsResponse)
async def api_kb_stats_default():
    """
    获取默认知识库统计（兼容旧版 API）。

    GET /api/kb/stats

    说明:
        这是 /api/kb/{kb_name}/stats 的快捷方式，固定查询 "default" 知识库。
        保留此路由是为了向后兼容旧版前端。
    """
    from core.rag_chain import get_kb_stats
    return get_kb_stats("default")


@app.post("/api/kb/upload", response_model=UploadResponse)
async def api_upload(
    files: List[UploadFile] = File(...),   # File(...) 表示必填，支持多文件
    kb_name: str = "default",
):
    """
    上传文档到知识库 — 支持多文件批量上传。

    POST /api/kb/upload?kb_name=default
    Body: multipart/form-data, 字段名 "files"（可多选）

    处理流程:
        1. 校验文件格式（.txt / .pdf / .md / .docx / .html）
        2. UUID 重命名 + 安全路径拼接（防止路径遍历攻击）
        3. 流式写入 uploads/ 目录
        4. 调用 create_knowledge_base() 创建/更新向量索引

    安全措施:
        · 文件名使用 os.path.basename() 截断路径
        · UUID 前缀确保唯一性，防止文件名冲突
        · 扩展名白名单校验，仅允许 SUPPORTED_FILE_TYPES

    返回:
        {
            "message": "已处理 3 个文件",
            "chunk_count": 150,    # 向量数据库中的总块数
            "file_count": 3,       # 本次上传文件数
            "kb_name": "default"
        }

    错误:
        400 → 不支持的格式
        413 → 文件过大（可在 Nginx 层限制）
    """
    from core.rag_chain import create_knowledge_base

    saved_paths = []

    for file in files:
        # ── 1. 格式校验 ──
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in settings.SUPPORTED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {ext}，支持: {settings.SUPPORTED_FILE_TYPES}",
            )

        # ── 2. 安全命名 ──
        # uuid4().hex → 32 位随机字符串（无特殊字符）
        # os.path.basename → 去除路径部分，仅保留文件名
        # 示例: "a1b2c3d4..._技术文档.pdf"
        safe_name = f"{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
        file_path = os.path.join(UPLOAD_DIR, safe_name)

        # ── 3. 分块写入磁盘（避免大文件一次性读入内存） ──
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB 分块
                if not chunk:
                    break
                f.write(chunk)
        saved_paths.append(file_path)

    # ── 4. 索引到向量数据库 ──
    # create_knowledge_base 内部: 加载文档 → 分块 → Embedding → Chroma 写入
    count = create_knowledge_base(file_paths=saved_paths, kb_name=kb_name)

    return UploadResponse(
        message=f"已处理 {len(saved_paths)} 个文件",
        chunk_count=count,
        file_count=len(saved_paths),
        kb_name=kb_name,
    )


@app.post("/api/kb/load-web", response_model=UploadResponse)
async def api_load_web(req: WebLoadRequest):
    """
    加载网页到知识库。

    POST /api/kb/load-web
    Body: {
        "urls": ["https://example.com/doc1", "https://example.com/doc2"],
        "kb_name": "技术文档"
    }

    处理流程:
        1. 对每个 URL 发起 HTTP 请求获取 HTML
        2. BeautifulSoup 提取正文（去除 script / style / nav 等标签）
        3. 分块 → Embedding → Chroma 写入

    注意:
        · 仅提取正文文本，不保留 HTML 标签
        · 不递归爬取子链接
        · 需要网络访问目标 URL
    """
    from core.rag_chain import create_knowledge_base

    count = create_knowledge_base(urls=req.urls, kb_name=req.kb_name)
    return UploadResponse(
        message=f"已处理 {len(req.urls)} 个网页",
        chunk_count=count,
        file_count=len(req.urls),
        kb_name=req.kb_name,
    )


@app.delete("/api/kb/{kb_name}/clear")
async def api_clear_kb(kb_name: str = "default"):
    """
    清空指定知识库（仅删除向量数据，不删除上传文件）。

    DELETE /api/kb/{kb_name}/clear

    说明:
        · 调用 Chroma 客户端的 delete_collection() 删除整个集合
        · 上传的源文件保留在 uploads/ 目录（可重新索引）
        · 不可逆操作，清空后需重新上传文档才能查询
    """
    from core.rag_chain import clear_knowledge_base
    clear_knowledge_base(kb_name)
    return {"message": f"知识库 [{kb_name}] 已清空"}


@app.delete("/api/kb/clear")
async def api_clear_kb_default():
    """
    清空默认知识库 + 清理上传目录（兼容旧版）。

    DELETE /api/kb/clear

    说明:
        比 /api/kb/default/clear 更彻底 — 同时删除 uploads/ 下所有文件。
        保留此路由是为了向后兼容旧版前端。
    """
    from core.rag_chain import clear_knowledge_base

    # 清空向量数据
    clear_knowledge_base("default")

    # 清理上传文件目录
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)         # 递归删除整个目录
        os.makedirs(UPLOAD_DIR, exist_ok=True)  # 重建空目录

    return {"message": "知识库已清空"}


# ═══════════════════════════════════════════════════════════════
# RAG 质量评估 API
# ═══════════════════════════════════════════════════════════════
#
# 三项评估指标（使用 RAGAS 框架）:
#   · Faithfulness       — 答案是否忠实于检索到的上下文（有无幻觉）
#   · AnswerRelevancy   — 答案与问题的相关程度
#   · ContextRelevancy  — 检索到的上下文与问题的相关程度


@app.post("/api/evaluate", response_model=EvaluationResponse)
async def api_evaluate(
    req: EvaluationRequest,
    deps: RAGDependencies = Depends(get_deps),  # 依赖注入: 获取单例 evaluator
):
    """
    批量 RAG 质量评估。

    POST /api/evaluate
    Body: {
        "questions": ["什么是 RAG？", "RAG 有什么优势？"],
        "ground_truths": ["RAG 是检索增强生成...", "优势包括...（可选）"],
        "kb_name": "default"
    }

    处理流程:
        1. 对每个问题调用 chat() 获取答案 + 检索来源
        2. 评估器对比 (问题, 答案, 上下文, 标准答案) 计算 3 项指标
        3. 汇总返回平均分 + 每题的详细分数

    返回:
        {
            "summary": {
                "faithfulness": 0.95,
                "answer_relevancy": 0.88,
                "context_relevancy": 0.82
            },
            "details": [
                {
                    "question": "什么是 RAG？",
                    "answer": "...",
                    "faithfulness": 0.95,
                    "answer_relevancy": 0.88,
                    "context_relevancy": 0.82
                }
            ]
        }
    """
    from core.rag_chain import chat

    # 定义 RAG 回调函数 — 评估器通过此函数获取 (答案, 来源) 对
    async def rag_fn(query: str):
        """
        评估器调用的 RAG 引擎包装函数。

        输入: 用户问题
        输出: (生成的答案, 检索到的上下文来源列表)

        评估器对每个 question 都调用此函数，
        拿到答案和上下文后计算三项质量指标。
        """
        result = await chat(query, kb_name=req.kb_name)
        return result["answer"], result.get("sources", [])

    # 批量评估
    result = await deps.evaluator.batch_evaluate(
        questions=req.questions,
        ground_truths=req.ground_truths,
        rag_fn=rag_fn,
    )
    return result


# ═══════════════════════════════════════════════════════════════
# 健康检查 API
# ═══════════════════════════════════════════════════════════════
#
# 用于:
#   · Kubernetes / Docker 存活探针（livenessProbe）
#   · 负载均衡器后端健康检测
#   · 运维监控面板


@app.get("/api/health")
async def health():
    """
    健康检查 — 验证系统各组件的实际可用性。

    GET /api/health

    返回状态:
        "ok"       → 所有组件正常
        "degraded" → 部分组件异常但核心功能仍可用

    检查项:
        · database  → SQLite 连接 + SELECT 1 测试
        · chroma    → 向量数据库可连接性 + 知识库数量
        · llm       → API Key 是否已配置

    返回值示例:
        {
            "status": "ok",
            "version": "2.1.0",
            "model": "<settings.LLM_MODEL，如 glm-4-flash>",
            "kb_count": 3,
            "features": {
                "hybrid_retrieval": true,
                "reranking": false,
                "query_expansion": true,
                "self_rag": true
            },
            "services": {
                "database": "healthy",
                "chroma": "healthy (3 KBs)",
                "llm": "configured"
            }
        }
    """
    from core.rag_chain import get_all_kb_stats

    # ── 基础信息 ──
    checks = {
        "status": "ok",
        "version": "2.1.0",
        "model": settings.LLM_MODEL,
        "kb_count": len(get_all_kb_stats()),
        "features": {
            "hybrid_retrieval": settings.ENABLE_HYBRID_RETRIEVAL,
            "reranking": settings.ENABLE_RERANKING,
            "query_expansion": settings.ENABLE_QUERY_EXPANSION,
            "self_rag": settings.ENABLE_SELF_RAG,
        },
        "services": {},
    }

    # ── 数据库连接检查 ──
    # 从连接池获取连接 → 执行 SELECT 1 → 归还连接
    try:
        from infrastructure.database import _get_conn, _release_conn
        conn = await _get_conn()
        await conn.execute("SELECT 1")
        await _release_conn(conn)
        checks["services"]["database"] = "healthy"
    except Exception as e:
        checks["services"]["database"] = f"unhealthy: {e}"
        checks["status"] = "degraded"

    # ── Chroma 向量库检查 ──
    try:
        from core.rag_chain import get_all_kb_names
        kb_names = get_all_kb_names()
        checks["services"]["chroma"] = f"healthy ({len(kb_names)} KBs)"
    except Exception as e:
        checks["services"]["chroma"] = f"error: {e}"

    # ── LLM 配置检查 ──
    # 不实际调用 API（避免消耗 token），仅检查 Key 是否已配置
    if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "sk-your-api-key-here":
        checks["services"]["llm"] = "configured"
    else:
        checks["services"]["llm"] = "not_configured"
        if checks["status"] == "ok":
            checks["status"] = "degraded"

    return checks


# ═══════════════════════════════════════════════════════════════
# 直接运行入口
# ═══════════════════════════════════════════════════════════════
#
# python main.py → Uvicorn 启动，绑定 0.0.0.0:8000
# 开发模式（DEBUG=true）自动开启热重载

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",                          # 模块:应用实例（字符串形式以支持热重载）
        host=settings.HOST,                  # 0.0.0.0 → 监听所有网络接口
        port=settings.PORT,                  # 默认 8000
        reload=settings.DEBUG,               # 开发模式: 文件变更自动重启
        log_level=settings.LOG_LEVEL.lower(),# info / debug / warning
    )
