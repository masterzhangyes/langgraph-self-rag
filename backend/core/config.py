"""
全局配置管理模块 - 基于 Pydantic Settings
========================================

本模块是系统的配置中心，统一管理整个 RAG 智能问答系统所需的所有配置项。
基于 pydantic-settings 的 BaseSettings 实现，支持以下配置方式（按优先级从高到低）：
  1. 环境变量（如 OPENAI_API_KEY=sk-xxx）
  2. .env 文件中的键值对
  3. 代码中定义的默认值

使用方式:
    from core.config import settings
    print(settings.OPENAI_API_KEY)
    print(settings.LLM_MODEL)

配置项按功能划分为以下分组:
  - OpenAI API 配置
  - LLM 大语言模型配置
  - Embedding 嵌入模型配置
  - 向量数据库配置
  - 文档处理与检索配置
  - RAG Agent 功能开关
  - 对话历史配置
  - 服务运行配置
  - 安全与认证配置
  - 支持的文件格式
"""

import os
from typing import List

# pydantic-settings 核心组件：
#   - BaseSettings: 类似 BaseModel，但额外支持从环境变量和 .env 文件自动加载配置
from pydantic_settings import BaseSettings
# Field: 用于对配置字段添加默认值、描述信息和校验规则
# model_validator: 用于对模型整体做跨字段校验（此处用于安全配置的强制校验）
from pydantic import Field, model_validator


class Settings(BaseSettings):
    """
    应用全局配置类
    
    继承自 BaseSettings，启动时自动从 .env 文件和环境变量中加载配置。
    所有字段均有默认值，即使不提供 .env 文件也能以默认配置运行。
    通过 .env 文件或环境变量可覆盖任意默认值。
    
    示例 .env 文件:
        OPENAI_API_KEY=sk-your-key-here
        LLM_MODEL=gpt-4
        EMBEDDING_DEVICE=cuda
    """

    # ── OpenAI API 配置 ──
    # 配置与 OpenAI 兼容的 API 服务连接信息。
    # 支持 OpenAI 官方 API 以及任何兼容 OpenAI 接口的第三方服务（如本地部署的 Ollama、vLLM 等）。

    # OpenAI API 密钥，用于身份认证
    # 生产环境必须设置为真实密钥，切勿在代码中硬编码
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API 密钥")
    # API 基础地址，默认为 OpenAI 官方端点
    # 可修改为代理地址或本地部署的地址（如 http://localhost:11434/v1）
    OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1", description="API 地址（支持代理）")

    # ── LLM 大语言模型配置 ──
    # 控制对话生成使用的模型及其参数。

    # 对话使用的 LLM 模型名称
    # 可选值取决于所使用的 API 服务，如 gpt-3.5-turbo、gpt-4、chatglm3-6b 等
    LLM_MODEL: str = Field(default="gpt-3.5-turbo", description="对话模型名称")
    # 生成温度，控制输出的随机性：
    #   - 0.0~0.3: 输出确定性高，适合事实性问答
    #   - 0.5~1.0: 平衡创造性和准确性
    #   - 1.0~2.0: 输出更随机、更有创造性
    LLM_TEMPERATURE: float = Field(default=0.3, description="生成温度 (0-2)")
    # 单次生成的最大 token 数量，限制回答的最大长度
    # 较大的值允许更长的回答，但也会增加响应时间和成本
    LLM_MAX_TOKENS: int = Field(default=2048, description="最大生成 token 数")

    # ── Embedding 嵌入模型配置 ──
    # 用于将文本转换为向量表示的模型配置，
    # 向量表示是语义检索的基础，直接影响检索质量。

    # 嵌入模型名称，用于将文本编码为向量
    # 默认使用 OpenAI 的 text-embedding-3-small（性价比高）
    # 也可切换为 text-embedding-ada-002 或 text-embedding-3-large
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="嵌入模型名称")
    # 本地嵌入模型的运行设备：
    #   - "cpu": 使用 CPU 推理（通用，无需 GPU）
    #   - "cuda": 使用 GPU 推理（速度更快，需要 NVIDIA GPU）
    EMBEDDING_DEVICE: str = Field(default="cpu", description="本地嵌入设备 (cpu/cuda)")

    # ── 本地 Embedding 备选配置 ──
    # 当不想使用 OpenAI 的嵌入服务时，可启用本地嵌入模型。
    # 本地模型基于 sentence-transformers 库运行，无需联网，无 API 费用。

    # 是否使用本地嵌入模型替代 OpenAI 嵌入
    # True: 使用本地模型（需安装 sentence-transformers）
    # False: 使用 OpenAI API 嵌入
    USE_LOCAL_EMBEDDINGS: bool = Field(default=False, description="是否使用本地嵌入模型")
    # 本地嵌入模型名称（HuggingFace 模型标识符）
    # 默认使用 BAAI/bge-small-zh-v1.5（中文优化，轻量高效）
    # 其他可选: BAAI/bge-base-zh-v1.5, shibing624/text2vec-base-chinese 等
    LOCAL_EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        description="本地嵌入模型名称（sentence-transformers）",
    )
    # 本地重排序（Reranker）模型名称
    # 用于对初步检索结果进行二次精排，提升最终送入 LLM 的文档质量
    # BAAI/bge-reranker-base 是中文场景下常用的轻量级重排序模型
    LOCAL_RERANKER_MODEL: str = Field(
        default="BAAI/bge-reranker-base",
        description="本地重排序模型名称",
    )

    # ── 向量数据库配置 ──
    # 系统使用 ChromaDB 作为向量数据库，存储文档的向量表示。

    # ChromaDB 数据持久化存储路径
    # 向量数据会保存在该目录下，重启后自动恢复
    VECTOR_DB_PATH: str = Field(default="./chroma_db", description="Chroma 持久化路径")

    # ── 文档处理与检索配置 ──
    # 控制文档切分策略和检索参数，直接影响 RAG 的检索质量。

    # 文本分块大小（字符数）
    # 文档会被切分为此大小的块，每块独立向量化后存入数据库
    # 较大的块保留更多上下文，但可能引入噪声
    CHUNK_SIZE: int = Field(default=1000, description="文本块大小")
    # 相邻块之间的重叠字符数
    # 重叠可以确保跨块的上下文信息不会丢失
    # 通常设置为 CHUNK_SIZE 的 10%~20%
    CHUNK_OVERLAP: int = Field(default=200, description="块重叠大小")
    # 初步检索返回的文档数量（Top-K）
    # 从向量数据库中返回最相似的 K 个文本块
    # 较大的值提供更多上下文，但也会增加 LLM 的处理负担
    RETRIEVAL_TOP_K: int = Field(default=20, description="初检索返回数量")
    # 重排序后最终保留的文档数量
    # 对初步检索结果进行精排后，只保留最相关的前 N 个文档送入 LLM
    # 通常远小于 RETRIEVAL_TOP_K，起到「精筛」作用
    RERANK_TOP_K: int = Field(default=5, description="重排序后保留数量")

    # ── RAG Agent 功能开关 ──
    # 控制 Self-RAG 流水线中各个增强功能的启用/禁用。
    # 每个功能都会增加一定的延迟和成本，可根据实际需求灵活开关。

    # 是否启用混合检索（向量检索 + BM25 关键词检索的融合）
    # 混合检索能同时利用语义相似度和关键词匹配，提升召回率
    ENABLE_HYBRID_RETRIEVAL: bool = Field(default=True, description="启用混合检索")
    # 是否启用重排序（Reranking）
    # 对初步检索结果进行二次精排，提升最终文档质量
    ENABLE_RERANKING: bool = Field(default=True, description="启用重排序")
    # 是否启用查询扩展
    # 将用户问题扩展为多个搜索角度，提高检索召回率
    ENABLE_QUERY_EXPANSION: bool = Field(default=True, description="启用查询扩展")
    # 是否启用 Self-RAG Agent（自反思机制）
    # 启用后会在生成回答后自动检测幻觉，发现不准确时重新生成
    ENABLE_SELF_RAG: bool = Field(default=True, description="启用 Self-RAG Agent")
    # 检索最大重试次数
    # 当检索相关性不足时，系统会改写查询并重试，最多重试此次数
    MAX_RETRIEVAL_RETRIES: int = Field(default=2, description="检索最大重试次数")

    # ── 对话历史配置 ──
    # 管理多轮对话的历史记录存储和保留策略。

    # 保留的最大对话轮数（一问一答算一轮）
    # 超过此数量的早期对话会被截断，避免提示词过长
    MAX_HISTORY_ROUNDS: int = Field(default=10, description="保留最大对话轮数")
    # SQLite 数据库文件路径，用于持久化存储对话历史
    DB_PATH: str = Field(default="./data/chat.db", description="SQLite 数据库路径")

    # ── 服务运行配置 ──
    # 控制 FastAPI 服务的监听地址、端口、日志等运行参数。

    # 服务监听的 IP 地址
    # "0.0.0.0" 表示监听所有网络接口（允许外部访问）
    # "127.0.0.1" 表示仅监听本机访问（更安全）
    HOST: str = Field(default="0.0.0.0", description="服务监听地址")
    # 服务监听的端口号
    PORT: int = Field(default=8000, description="服务端口")
    # 是否启用调试模式
    # True: 代码修改后自动重载，显示详细错误信息（仅开发环境使用）
    # False: 生产环境应关闭
    DEBUG: bool = Field(default=False, description="调试模式")
    # 日志记录级别：DEBUG / INFO / WARNING / ERROR / CRITICAL
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")
    # 日志文件存储路径
    LOG_PATH: str = Field(default="./logs/app.log", description="日志文件路径")

    # ── 安全与认证配置 ──
    # 控制 API 的访问安全策略，包括认证、限流和跨域设置。

    # API 认证密钥，用于生成和验证 API Token
    # 生产环境必须修改为强随机字符串
    SECRET_KEY: str = Field(default="change-me-in-production", description="API 认证密钥")
    # 是否启用 API 认证（Key 验证）
    # True: 所有 API 请求需携带有效的 API Key
    # False: 开放访问（适用于开发/内网环境）
    API_AUTH_ENABLED: bool = Field(default=False, description="是否启用 API 认证")
    # API 认证使用的 HTTP 请求头名称
    # 客户端需在请求头中携带: X-API-Key: your-key-here
    API_AUTH_HEADER: str = Field(default="X-API-Key", description="API 认证请求头名称")
    # 全局限流策略，格式为 "请求数/时间窗口"
    # "30/minute" 表示每个客户端每分钟最多 30 次请求
    RATE_LIMIT: str = Field(default="30/minute", description="全局限流策略")
    # 允许的跨域来源列表（CORS）
    # ["*"] 表示允许所有来源（开发环境）
    # 生产环境应限制为具体的前端域名，如 ["http://localhost:5173"]
    ALLOWED_ORIGINS: List[str] = Field(default=["*"], description="允许的跨域来源")

    # ── 支持的文件格式 ──
    # 定义知识库支持上传的文档类型，上传时会校验文件扩展名。

    # 允许上传的文档格式列表
    # 系统支持解析这些格式的文本内容并导入知识库
    SUPPORTED_FILE_TYPES: List[str] = Field(
        default=[".txt", ".pdf", ".csv", ".md", ".docx"],
        description="支持上传的文档格式",
    )

    @model_validator(mode="after")
    def _validate_security(self):
        """
        安全配置强制校验：
        - 开启 API 认证（API_AUTH_ENABLED=true）时，不允许继续使用默认 SECRET_KEY，
          否则认证形同虚设。
        - 认证默认关闭时不做限制，保证开发环境开箱即用。
        """
        if self.API_AUTH_ENABLED and self.SECRET_KEY in (
            "change-me-in-production",
            "change-me-in-production-use-a-random-string",
        ):
            raise ValueError(
                "API_AUTH_ENABLED=true 时必须配置强随机的 SECRET_KEY，"
                '可执行: python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
        return self

    class Config:
        """
        Pydantic Settings 的内部配置
        
        控制配置的加载行为：
          - env_file: 指定 .env 文件路径，启动时自动读取
          - env_file_encoding: .env 文件的字符编码
          - extra: 遇到未定义的字段时的行为（"ignore" 表示忽略）
        """
        # .env 文件路径（相对于项目运行目录）
        env_file = ".env"
        # .env 文件使用 UTF-8 编码
        env_file_encoding = "utf-8"
        # 忽略 .env 中未在 Settings 类中定义的额外字段，避免报错
        extra = "ignore"


# ── 全局配置单例 ──
# 创建 Settings 实例并赋值给全局变量 settings。
# 其他模块通过 `from core.config import settings` 导入使用。
# 由于 Python 模块的缓存机制，settings 在整个应用生命周期中只会被创建一次（单例模式）。
settings = Settings()
