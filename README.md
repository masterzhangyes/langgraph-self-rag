# 智能问答系统 v2.1

> 基于 **LangChain RAG** + **LangGraph Agent** 的智能问答平台
> 默认使用 **智谱清言 `glm-4-flash` 免费模型**（OpenAI 兼容 API，可随时切换其它厂商）
<img width="2876" height="1472" alt="fb5dee4b-a661-491e-8691-f2bb221d1bdd" src="https://github.com/user-attachments/assets/353a3138-4500-4268-962d-91ebea2e85bb" />
<img width="2864" height="1468" alt="8ef9d138-e37f-4586-80ee-787567582728" src="https://github.com/user-attachments/assets/5d40fe2b-d64a-4507-b05c-c038842ab696" />

---

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | FastAPI + Uvicorn + LangChain/LangGraph |
| LLM | 智谱 `glm-4-flash`（免费）/ 任意 OpenAI 兼容 API (DeepSeek、OpenAI、通义等) |
| 检索 | BM25 稀疏 + Dense 稠密 + RRF 融合 |
| 重排序 | Cross-Encoder / LLM-based |
| 向量库 | Chroma (本地持久化) |
| 嵌入 | 智谱 `embedding-3`（云端）/ BGE 系列（本地） |
| 对话存储 | SQLite + aiosqlite |
| 前端 | React 18 + TypeScript + Vite 5 |
| 部署 | Docker + docker-compose + Nginx |

## 项目结构

```
智能问答/
├── start_all.py               # 一键启动脚本（核心，跨平台）
├── start_all.bat              # Windows 一键启动入口（双击即可）
├── docker-compose.yml
├── README.md
├── .github/
│   └── workflows/ci.yml       # CI：push/PR 自动跑后端 pytest + 前端构建
│
├── backend/
│   ├── main.py                # FastAPI 入口（路由、限流、生命周期）
│   ├── requirements.txt
│   ├── pytest.ini             # 测试配置（asyncio 模式、测试目录）
│   ├── Dockerfile
│   ├── .env                   # 环境变量（含密钥，不提交）
│   ├── .env.example           # 环境变量模板（多供应商说明）
│   ├── data/                  # SQLite 对话记录 (chat.db)
│   ├── uploads/               # 知识库上传文档
│   ├── logs/                  # 应用运行日志 (app.log)
│   ├── chroma_default/        # Chroma 向量库（目录名由 .env: VECTOR_DB_PATH 指定）
│   │
│   ├── api/                   # 表示层
│   │   ├── models.py          # 请求/响应 Pydantic 模型
│   │   └── middleware.py      # 请求日志、API 认证、异常处理
│   │
│   ├── core/                  # 核心引擎
│   │   ├── config.py          # Pydantic Settings 统一配置（含安全校验）
│   │   ├── rag_chain.py       # RAG 核心逻辑（多KB、对话、文档索引、流式）
│   │   └── agent.py           # LangGraph Self-RAG Agent (6 节点)
│   │
│   ├── services/              # 业务服务
│   │   ├── retriever.py       # 混合检索 (BM25 + Dense + RRF)
│   │   ├── reranker.py        # 重排序器（Cross-Encoder / LLM）
│   │   └── evaluation.py      # RAG 质量评估 (3 指标)
│   │
│   ├── infrastructure/        # 基础设施
│   │   ├── database.py        # SQLite 会话持久化
│   │   └── dependencies.py    # 依赖注入容器
│   │
│   └── tests/                 # 单元测试（conftest 自动 mock，无需真实 Key）
│       ├── conftest.py
│       ├── test_retriever.py
│       ├── test_reranker.py
│       └── test_evaluation.py
│
├── docs/
│   └── learning/              # 模块学习文档（Word，供快速上手）
│
└── frontend/
    ├── index.html
    ├── vite.config.ts
    ├── nginx.conf
    ├── Dockerfile
    └── src/
        ├── main.tsx           # React 入口
        ├── App.tsx            # 根组件 & 全局状态
        ├── App.css            # 双主题样式系统
        ├── index.css          # 全局重置与字体
        ├── components/
        │   ├── ChatArea.tsx   # 对话消息区
        │   ├── InputArea.tsx  # 输入框
        │   ├── KBPanel.tsx    # 知识库管理面板
        │   ├── SettingsPanel.tsx # 设置面板
        │   ├── Sidebar.tsx    # 会话历史
        │   ├── TopBar.tsx     # 顶栏
        │   └── Toast.tsx      # 通知系统
        └── hooks/
            └── useChat.ts     # SSE 流式对话 Hook
```

## 快速启动

### 环境要求

- Python 3.10+ / Node.js 18+
- 智谱开放平台 API Key（`https://open.bigmodel.cn` 控制台 → API Key，免费模型同样需要 Key）
- 需要外网访问 `https://open.bigmodel.cn`

### 方式一：Windows 一键启动（推荐）

**双击 `start_all.bat`**，或命令行执行：

```bash
python start_all.py             # 一键启动前后端（已运行则自动跳过）
python start_all.py --no-browser # 启动但不自动打开浏览器
python start_all.py --quick     # 快速返回，服务在后台继续启动
python start_all.py --status    # 查看当前运行状态
python start_all.py --stop      # 停止前后端服务
```

脚本会自动完成：创建后端虚拟环境（首次）→ 安装依赖 → 启动 FastAPI → 启动 Vite，并把输出写入 `logs/` 目录。

### 方式二：手动启动（本地开发）

```bash
# 1. 安装后端依赖（建议先创建虚拟环境）
cd backend
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，至少修改：OPENAI_API_KEY（智谱 Key）

# 3. 启动后端
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# 4. 启动前端（新终端）
cd frontend
npm install
npx vite --host 0.0.0.0 --port 5173
```

- 前端: `http://localhost:5173`
- 后端: `http://localhost:8000`
- API 文档: `http://localhost:8000/docs`

### Docker 部署

```bash
docker-compose up -d
```

后端镜像的默认环境变量已预置为智谱配置（`glm-4-flash` + `embedding-3`，见 `docker-compose.yml`）；启动前请确保 `backend/.env` 中的 `OPENAI_API_KEY` 已填写，Compose 会自动注入容器。

## 模型配置

本项目对 LLM / 嵌入层使用 OpenAI 兼容协议，全部通过 `backend/.env` 控制。默认配置（智谱）：

```dotenv
OPENAI_API_KEY=sk-你的智谱Key          # 必填
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LLM_MODEL=glm-4-flash                  # 免费模型，128K 上下文
EMBEDDING_MODEL=embedding-3            # 云端嵌入（输出 2048 维）
USE_LOCAL_EMBEDDINGS=false
```

**切换到其它厂商**（DeepSeek / OpenAI / Ollama 等），只需修改上面前三项，例如：

```dotenv
# DeepSeek
OPENAI_API_KEY=sk-xxxx
OPENAI_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 本地 Ollama（可离线）
OPENAI_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:7b
```

> 注意：**切换 LLM 厂商无需重建知识库**；但若更换 **Embedding 模型**（如 OpenAI `text-embedding-3-small` ↔ 智谱 `embedding-3`），两者向量维度不同，须先清空已有知识库再重新上传文档（页面操作或 `DELETE /api/kb/{name}/clear`）。

### 常用功能开关

全部在 `backend/.env` 中配置（下表为当前项目默认值）：

| 键 | 默认 | 说明 |
|---|---|---|
| `USE_LOCAL_EMBEDDINGS` | `false` | `true` 时改用本地 `BAAI/bge-small-zh-v1.5` 嵌入（离线免费，需 CPU/GPU） |
| `ENABLE_HYBRID_RETRIEVAL` | `true` | BM25 关键词 + Dense 向量混合检索 |
| `ENABLE_RERANKING` | `false` | Cross-Encoder 精排（首次需下载模型约 1GB+，建议评估收益后再开） |
| `ENABLE_QUERY_EXPANSION` | `true` | 检索前把问题扩展为多个搜索角度 |
| `ENABLE_SELF_RAG` | `true` | LangGraph Agent（查询扩展→检索→相关性评估→幻觉检测） |
| `RATE_LIMIT` | `30/minute` | 基于 IP 的全局限流 |
| `API_AUTH_ENABLED` | `false` | `true` 时请求须携带 `X-API-Key`；开启前必须配置强随机 `SECRET_KEY`（否则启动会拒绝） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | 文本切分参数 |
| `RETRIEVAL_TOP_K` / `RERANK_TOP_K` | `20` / `5` | 召回 / 精排文档数 |

> Self-RAG 每轮会多次调用 LLM（默认 `glm-4-flash` 免费额度可覆盖）；追求速度或省成本时可设 `ENABLE_SELF_RAG=false` 走「检索→生成」单轮直答。

## API 概览

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/chat` | 非流式对话 |
| `POST` | `/api/chat/stream` | SSE 流式对话 |
| `GET` | `/api/sessions` | 会话列表 |
| `POST` | `/api/sessions` | 新建会话 |
| `GET` | `/api/sessions/{id}` | 会话详情 |
| `DELETE` | `/api/sessions/{id}` | 删除会话 |
| `GET` | `/api/kb/list` | 知识库列表 |
| `GET` | `/api/kb/stats` | 全部知识库统计 |
| `GET` | `/api/kb/{name}/stats` | 指定知识库统计 |
| `POST` | `/api/kb/upload` | 上传文档 |
| `POST` | `/api/kb/load-web` | 加载网页 |
| `DELETE` | `/api/kb/{name}/clear` | 清空指定知识库 |
| `DELETE` | `/api/kb/clear` | 清空全部知识库 |
| `POST` | `/api/evaluate` | RAG 质量评估 |
| `GET` | `/api/health` | 健康检查 |

## 核心特性

- **混合检索**: BM25 (jieba 中文分词) + Dense Embedding + RRF 融合
- **重排序**: BGE-Reranker Cross-Encoder 精排 / LLM 打分（默认关闭，`.env` 设 `ENABLE_RERANKING=true` 开启）
- **Self-RAG Agent**: LangGraph 6 节点状态图（查询扩展→检索→相关性评估→生成→幻觉检测，默认开启）
- **多知识库**: 命名隔离，独立 Chroma 目录
- **流式输出**: SSE 打字机效果，支持中止生成
- **会话持久化**: SQLite 存储历史对话
- **质量评估**: Faithfulness / Answer Relevancy / Context Relevancy
- **双主题**: 亮色/暗色模式，偏好持久化

## 运行测试

```bash
cd backend
pytest            # 等价于 pytest tests/ -v（由 pytest.ini 指定目录与参数）
```

- 测试**不依赖真实模型与外部网络**：`tests/conftest.py` 会自动 mock 关键配置（API Key、向量库/数据库路径），对混合检索、重排序、LLM 评估器做单元级验证。
- 已内置 CI（`.github/workflows/ci.yml`）：每次 push / PR 自动执行后端 `pytest` 与前端 `npm run build`。

## 常见问题

- **切换 Embedding 模型后检索为空**：不同嵌入模型向量维度不同，须先清空旧知识库再重传文档（`DELETE /api/kb/{name}/clear`）。
- **知识库为空时**：对话自动降级为「通用 LLM 直答」模式，不会报错。
- **Self-RAG Agent 流式说明**：Agent 内部为多节点自检流程，无法透传 token 级流式，后端会按句切分逐块推送（近似打字机效果）；真正的 token 级流式需在生成节点接入 streaming LLM + 异步队列。
