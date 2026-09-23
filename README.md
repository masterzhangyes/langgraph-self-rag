# 智能问答系统 v2.2

> 基于 **LangChain RAG** + **LangGraph Agent** 的智能问答平台
> 默认使用 **智谱清言 `glm-4-flash` 免费模型**（OpenAI 兼容 API，可随时切换其它厂商）
> v2.2 新增：**JWT 用户认证 + 多用户会话隔离 + 管理后台（RBAC / 审计日志 / 运营统计）**
<img width="2876" height="1472" alt="fb5dee4b-a661-491e-8691-f2bb221d1bdd" src="https://github.com/user-attachments/assets/353a3138-4500-4268-962d-91ebea2e85bb" />
<img width="2864" height="1468" alt="8ef9d138-e37f-4586-80ee-787567582728" src="https://github.com/user-attachments/assets/5d40fe2b-d64a-4507-b05c-c038842ab696" />

---

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | FastAPI + Uvicorn + LangChain/LangGraph |
| 认证 | JWT 双令牌（PyJWT）+ bcrypt 密码哈希 + RBAC |
| LLM | 智谱 `glm-4-flash`（免费）/ 任意 OpenAI 兼容 API (DeepSeek、OpenAI、通义等) |
| 检索 | BM25 稀疏 + Dense 稠密 + RRF 融合 |
| 重排序 | Cross-Encoder / LLM-based |
| 向量库 | Chroma (本地持久化) |
| 嵌入 | 智谱 `embedding-3`（云端）/ BGE 系列（本地） |
| 对话存储 | SQLite + aiosqlite（用户 / 会话 / 消息 / 审计日志） |
| 前端 | React 18 + TypeScript + Vite 5 |
| 部署 | Docker + docker-compose + Nginx |

## 用户认证与管理后台（v2.2）

### 认证体系

- **注册 / 登录**：`POST /api/auth/register`、`POST /api/auth/login`，成功即返回双令牌
- **双令牌机制**：access token（默认 30 分钟）+ refresh token（默认 7 天），前端自动静默续期
- **密码安全**：bcrypt 自适应哈希（随机盐 + cost=12），数据库不存明文
- **防爆破**：同一 (IP, 用户名) 连续登录失败超过阈值（默认 5 次）临时锁定（默认 5 分钟）
- **引导约定**：**首个注册的用户自动成为管理员**，之后注册的均为普通用户
- **会话隔离**：每个用户的对话会话按 `user_id` 隔离，互相不可见；未携带令牌的请求落入匿名桶（兼容旧客户端 / 脚本调用）

### 管理后台（仅 admin 可见）

前端右上角「管理」按钮进入，或直接调用 `/api/admin/*`：

- **运营概览**：用户总数 / 今日新增 / 会话总数 / 消息总数 / 今日活跃 / 知识库规模
- **用户管理**：角色提升/降级、启用/禁用、重置密码、删除用户（级联删除其会话）
- **会话审计**：查看全部用户会话（含归属），支持删除
- **操作日志**：登录 / 注册 / 登录失败 / 管理操作全量审计（含 IP）

### 安全设计要点

| 风险 | 对策 |
|---|---|
| 密码泄露 | bcrypt 哈希存储，脱敏响应模型（永不出参 password_hash） |
| 令牌泄露 | access token 短有效期；禁用用户令牌即刻失效 |
| 暴力破解 | (IP, 用户名) 滑动窗口限流 + 锁定 |
| 越权访问（IDOR） | 会话读取/删除校验归属，他人会话返回 404（不泄露存在性） |
| 管理员自锁 | 禁止禁用/删除自己；禁止降级最后一个活跃管理员 |
| 撞库探测 | 用户不存在与密码错误返回统一提示 |
| 操作抵赖 | 全量审计日志（操作者 / 目标 / IP / 时间） |

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
│   │   ├── models.py          # 请求/响应 Pydantic 模型（含认证模型）
│   │   ├── middleware.py      # 请求日志、API 认证、异常处理
│   │   ├── auth.py            # JWT 双令牌认证（注册/登录/刷新/RBAC 依赖/防爆破）
│   │   └── admin.py           # 管理后台路由（用户/统计/会话审计/日志）
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
│   │   ├── database.py        # SQLite 会话持久化（含 v2.2 幂等迁移）
│   │   ├── user_store.py      # 用户 / 审计日志 / 管理统计数据层
│   │   └── dependencies.py    # 依赖注入容器
│   │
│   └── tests/                 # 单元测试（conftest 自动 mock，无需真实 Key）
│       ├── conftest.py
│       ├── test_auth.py       # 密码哈希 / JWT / 防爆破 / 注册校验
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
        ├── App.tsx            # 根组件 & 全局状态（登录态门卫）
        ├── api.ts             # 统一请求封装（自动附带 Bearer 令牌 / 401 处理）
        ├── App.css            # 双主题样式系统
        ├── index.css          # 全局重置与字体
        ├── components/
        │   ├── LoginPage.tsx  # 登录 / 注册页
        │   ├── AdminPanel.tsx # 管理后台（统计/用户/会话审计/日志）
        │   ├── ChatArea.tsx   # 对话消息区
        │   ├── InputArea.tsx  # 输入框
        │   ├── KBPanel.tsx    # 知识库管理面板
        │   ├── SettingsPanel.tsx # 设置面板
        │   ├── Sidebar.tsx    # 会话历史
        │   ├── TopBar.tsx     # 顶栏（用户信息 / 管理入口 / 登出）
        │   └── Toast.tsx      # 通知系统
        └── hooks/
            ├── useAuth.ts     # 认证 Hook（登录态恢复 / 静默刷新）
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

### 认证（v2.2）

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/auth/register` | 注册（首个用户自动成为 admin） |
| `POST` | `/api/auth/login` | 登录（返回双令牌） |
| `POST` | `/api/auth/refresh` | 刷新令牌 |
| `GET` | `/api/auth/me` | 当前用户信息 |
| `POST` | `/api/auth/logout` | 退出登录 |

> 认证后的请求须携带 `Authorization: Bearer <access_token>`。

### 管理后台（v2.2，仅 admin）

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/admin/stats` | 运营统计（用户/会话/消息/今日活跃/知识库） |
| `GET` | `/api/admin/users` | 用户列表 |
| `PATCH` | `/api/admin/users/{id}` | 更新用户（角色/启停/重置密码） |
| `DELETE` | `/api/admin/users/{id}` | 删除用户（级联删除会话） |
| `GET` | `/api/admin/sessions` | 全量会话列表（含归属用户） |
| `DELETE` | `/api/admin/sessions/{id}` | 删除任意会话 |
| `GET` | `/api/admin/audit-logs` | 审计日志 |

### 业务

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/chat` | 非流式对话 |
| `POST` | `/api/chat/stream` | SSE 流式对话 |
| `GET` | `/api/sessions` | 当前用户会话列表 |
| `POST` | `/api/sessions` | 新建会话 |
| `GET` | `/api/sessions/{id}` | 会话详情（仅归属者） |
| `DELETE` | `/api/sessions/{id}` | 删除会话（仅归属者） |
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

- **用户认证**: JWT 双令牌（access + refresh）+ bcrypt 密码哈希 + 登录防爆破
- **RBAC 权限**: user / admin 两级角色，管理后台全端点守卫
- **多用户会话隔离**: 会话按 `user_id` 隔离，支持越权防护（IDOR）与匿名兼容
- **管理后台**: 运营统计 / 用户管理 / 会话审计 / 操作日志
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

- **忘记管理员密码 / 想重置所有用户**：删除 `backend/data/chat.db` 中的 `users` 表记录（或整个库文件），重启后首个注册用户重新成为管理员。
- **旧数据库升级 v2.2**：启动时自动执行幂等迁移（`chat_sessions` 补 `user_id` 列 + 新建 `users` / `audit_logs` 表），旧会话归入匿名桶，无需手工处理。
- **API 脚本调用如何认证**：先 `POST /api/auth/login` 拿到 `access_token`，之后每个请求带 `Authorization: Bearer <token>`；不想登录的老脚本不带令牌也能用（匿名桶）。
- **切换 Embedding 模型后检索为空**：不同嵌入模型向量维度不同，须先清空旧知识库再重传文档（`DELETE /api/kb/{name}/clear`）。
- **知识库为空时**：对话自动降级为「通用 LLM 直答」模式，不会报错。
- **Self-RAG Agent 流式说明**：Agent 内部为多节点自检流程，无法透传 token 级流式，后端会按句切分逐块推送（近似打字机效果）；真正的 token 级流式需在生成节点接入 streaming LLM + 异步队列。
