"""
Pydantic 数据模型模块
==================

本模块定义了 RAG（检索增强生成）智能问答系统中所有 API 接口所使用的数据模型。
基于 Pydantic BaseModel 实现，提供字段类型校验、默认值设定及描述信息。

模型按功能划分为以下四大类：
    1. 对话相关模型 —— 处理用户提问、AI 回答、会话管理等
    2. 知识库相关模型 —— 管理知识库的增删查改、文件上传、网页加载等
    3. 评估相关模型 —— 对 RAG 系统的回答质量进行量化评估
    4. 通用模型 —— 健康检查、错误响应等公共结构
"""

# typing 模块提供 Python 类型提示支持：
#   - List: 列表类型，如 List[str] 表示字符串列表
#   - Optional: 可选类型，表示该字段可以为 None
#   - Literal: 字面量类型，限制字段只能取指定的固定值
from typing import List, Optional, Literal

# Pydantic 核心组件：
#   - BaseModel: 所有数据模型的基类，提供自动数据验证、序列化/反序列化
#   - Field: 用于对字段进行更精细的约束（如最小长度、取值范围）和描述
from pydantic import BaseModel, Field


# ─────────────────── 对话相关模型 ───────────────────
# 这一组模型用于处理智能问答的核心对话流程，
# 包括：单条消息、用户请求、AI 响应、引用来源、会话信息。

class ChatMessage(BaseModel):
    """
    单条对话消息模型
    
    用于表示对话中的任意一条消息，遵循 OpenAI 消息格式规范。
    每条消息由「角色」和「内容」两部分组成：
      - role:    标识消息的发送者身份
      - content: 消息的文本内容
    """
    # 消息角色，使用 Literal 类型严格限制为三种值之一：
    #   - "user":      用户发送的消息
    #   - "assistant":  AI 助手生成的回复
    #   - "system":    系统级指令（用于设定 AI 的行为规则）
    # 默认值为 "user"，即如果不指定角色，则视为用户消息
    role: Literal["user", "assistant", "system"] = "user"
    # 消息的文本内容，为必填字段，无默认值
    content: str


class ChatRequest(BaseModel):
    """
    对话请求模型
    
    前端发起对话时提交的请求体。包含用户问题、历史对话记录、
    会话标识、目标知识库、输出方式及生成参数等控制选项。
    """
    # 用户本次提出的问题，必填字段（... 表示必填）
    # 约束：最小长度为 1，即不允许提交空问题
    query: str = Field(..., min_length=1, description="用户问题")
    # 历史对话记录，用于维持多轮对话的上下文连贯性
    # default_factory=list 表示每次创建新实例时生成一个空列表（避免可变默认值陷阱）
    history: List[ChatMessage] = Field(default_factory=list, description="对话历史")
    # 会话唯一标识符，用于关联同一轮对话的多条消息
    # 为 None 时表示开启一个全新会话
    session_id: Optional[str] = Field(default=None, description="会话 ID")
    # 指定本次查询所使用的知识库名称列表
    # 为 None 时使用系统默认知识库；可指定多个知识库进行联合检索
    kb_names: Optional[List[str]] = Field(default=None, description="指定知识库名称")
    # 是否采用流式输出（Server-Sent Events）
    # True: 逐 token 实时返回，用户体验更好（打字机效果）
    # False: 等待完整生成后一次性返回
    stream: bool = Field(default=True, description="是否流式输出")
    # LLM 生成温度参数，控制输出的随机性：
    #   - 较低值（如 0.1）：输出更确定、更保守
    #   - 较高值（如 1.5）：输出更随机、更有创造性
    # 取值范围 [0, 2]；为 None 时使用模型默认温度
    temperature: Optional[float] = Field(default=None, ge=0, le=2, description="生成温度")


class SourceDocument(BaseModel):
    """
    检索到的源文档模型
    
    表示 RAG 系统从知识库中检索到的一个文本片段（chunk），
    作为生成回答的参考依据。前端可展示这些信息供用户核查。
    """
    # 检索到的文本片段内容
    content: str = Field(..., description="文本内容")
    # 该文本片段的来源标识，可以是文件路径、网页 URL 等
    source: str = Field(..., description="来源路径/URL")
    # 相关性评分（通常为 0~1 之间的浮点数）
    # 分数越高表示该片段与用户问题的语义相似度越高
    relevance_score: float = Field(..., description="相关性分数")
    # 该片段在原始文档中的分块索引位置
    # 用于标识这是文档被切分后的第几个块，默认为 0
    chunk_index: int = Field(default=0, description="块索引")


class ChatResponse(BaseModel):
    """
    对话响应模型
    
    AI 处理完用户请求后返回的完整响应结构，包含：
    生成的回答、引用的来源文档、置信度评分、会话标识等。
    """
    # AI 生成的回答文本内容
    answer: str = Field(..., description="回答内容")
    # 当前知识库的状态标识：
    #   - "empty":   知识库为空，尚未导入任何文档
    #   - "loading": 知识库正在加载/索引中
    #   - "active":  知识库已就绪，可正常检索
    kb_status: str = Field(default="empty", description="知识库状态")
    # 回答所引用的来源文档列表，按相关性降序排列
    sources: List[SourceDocument] = Field(default_factory=list, description="引用来源")
    # 回答的置信度评分（0~1），表示系统对回答准确性的自信程度
    # 为 None 表示未计算置信度
    confidence: Optional[float] = Field(default=None, description="置信度")
    # 当前会话的唯一标识符，前端可用此 ID 继续同一会话的后续对话
    session_id: Optional[str] = Field(default=None, description="会话 ID")


class ChatSession(BaseModel):
    """
    对话会话模型
    
    用于表示一个完整的对话会话的元数据信息，
    通常在「会话列表」接口中返回，供前端展示历史会话。
    """
    # 会话的唯一标识符（通常为 UUID）
    id: str
    # 会话标题，通常由首条用户消息自动生成或由用户手动设定
    title: str
    # 会话创建时间（ISO 8601 格式字符串，如 "2025-01-01T12:00:00"）
    created_at: str
    # 会话最后更新时间，用于排序显示最近活跃的会话
    updated_at: str
    # 该会话中包含的消息总数（一问一答算两条）
    message_count: int = 0


# ─────────────────── 知识库相关模型 ───────────────────
# 这一组模型用于管理 RAG 系统的知识库（Knowledge Base），
# 包括：知识库信息查询、文件上传响应、网页加载、统计与列表展示。

class KnowledgeBaseInfo(BaseModel):
    """
    知识库信息模型
    
    描述一个知识库的完整元数据，用于创建知识库或查询知识库详情时返回。
    知识库是 RAG 系统中存储和管理文档的核心容器。
    """
    # 知识库的唯一名称，用于在 API 调用中标识和引用该知识库
    name: str = Field(..., description="知识库名称")
    # 知识库的文字描述，说明其用途或包含的内容类型
    description: str = Field(default="", description="描述")
    # 知识库当前的状态标识：
    #   - "active":  已加载完成，可正常进行检索
    #   - "empty":   已创建但尚未导入任何文档
    #   - "loading": 正在导入文档或重建索引中（此时不可检索）
    status: Literal["active", "empty", "loading"] = "empty"
    # 知识库中文本块（chunk）的总数量
    # 文档被切分后产生的所有片段之和
    chunk_count: int = 0
    # 知识库中导入的原始文档数量
    document_count: int = 0
    # 知识库的创建时间
    created_at: Optional[str] = None
    # 知识库的最后更新时间（如新增文档后更新）
    updated_at: Optional[str] = None


class WebLoadRequest(BaseModel):
    """
    网页加载请求模型
    
    用于从指定 URL 列表抓取网页内容，并将抓取结果导入到目标知识库中。
    系统会对网页进行解析、清洗、分块和向量化处理。
    """
    # 待抓取的网页 URL 列表，至少需要提供一个 URL
    # min_length=1 确保列表不能为空
    urls: List[str] = Field(..., min_length=1, description="网页 URL 列表")
    # 目标知识库名称，抓取的内容将存入该知识库
    # 默认存入名为 "default" 的默认知识库
    kb_name: str = Field(default="default", description="目标知识库")


class KBStatsResponse(BaseModel):
    """
    知识库统计信息模型
    
    返回单个知识库的核心统计数据，用于列表展示或快速查看知识库概况。
    """
    # 知识库名称
    name: str
    # 知识库当前状态（active / empty / loading）
    status: str
    # 文本块总数，反映知识库的内容规模
    chunk_count: int
    # 原始文档总数
    document_count: int


class KBListResponse(BaseModel):
    """
    知识库列表响应模型
    
    用于返回系统中所有知识库的列表，
    前端可在「知识库管理」页面展示该列表。
    """
    # 包含所有知识库统计信息的列表
    knowledge_bases: List[KBStatsResponse]


class UploadResponse(BaseModel):
    """
    文件上传响应模型
    
    文件上传并处理完成后返回的结果，告知前端上传的处理情况。
    """
    # 处理结果的消息描述（如 "上传成功" 或错误提示）
    message: str
    # 本次上传经切分后新增的文本块数量
    chunk_count: int
    # 本次成功处理的文件数量
    file_count: int
    # 文件被导入的目标知识库名称
    kb_name: str


# ─────────────────── 评估相关模型 ───────────────────
# 这一组模型用于 RAG 系统的回答质量评估（Evaluation），
# 通过量化指标衡量系统的检索准确性和生成质量。
# 评估维度参考了 RAGAS 等主流 RAG 评估框架。

class EvaluationRequest(BaseModel):
    """
    RAG 评估请求模型
    
    批量提交测试问题以评估 RAG 系统的回答质量。
    可选提供标准答案（ground truth）用于自动化评分。
    """
    # 待评估的测试问题列表，至少包含一个问题
    questions: List[str] = Field(..., min_length=1, description="测试问题列表")
    # 与问题一一对应的标准答案列表（可选）
    # 如果提供，系统将自动计算忠实度、相关性等评分指标
    # 如果为 None，则仅返回生成的回答，不做自动评分
    ground_truths: Optional[List[str]] = Field(default=None, description="标准答案")
    # 评估时使用的目标知识库名称
    kb_name: str = Field(default="default", description="目标知识库")


class EvaluationResult(BaseModel):
    """
    单条评估结果模型
    
    记录一个测试问题的完整评估结果，包括生成的回答、
    各项质量评分指标以及响应延迟。
    """
    # 原始测试问题
    question: str
    # RAG 系统生成的回答
    answer: str
    # 人工标注的标准答案（用于对比评估）
    ground_truth: Optional[str] = None
    # 忠实度评分（0~1）：衡量回答是否忠于检索到的上下文，
    # 即回答中的信息是否都能在源文档中找到依据，避免「幻觉」
    faithfulness: Optional[float] = None
    # 答案相关性评分（0~1）：衡量生成的回答与问题的相关程度，
    # 即回答是否直接、准确地回应了用户的问题
    answer_relevancy: Optional[float] = None
    # 上下文相关性评分（0~1）：衡量检索到的文档片段与问题的相关程度，
    # 即检索器是否找到了正确的参考信息
    context_relevancy: Optional[float] = None
    # 本次问答的响应延迟，单位为毫秒（ms）
    # 用于评估系统的性能表现
    latency_ms: float = 0


class EvaluationResponse(BaseModel):
    """
    批量评估响应模型
    
    汇总所有测试问题的评估结果，并提供各指标的平均值，
    用于整体衡量 RAG 系统的表现。
    """
    # 每个测试问题的详细评估结果列表
    results: List[EvaluationResult]
    # 所有问题的平均忠实度得分
    avg_faithfulness: Optional[float] = None
    # 所有问题的平均答案相关性得分
    avg_answer_relevancy: Optional[float] = None
    # 所有问题的平均上下文相关性得分
    avg_context_relevancy: Optional[float] = None
    # 所有问题的平均响应延迟（毫秒）
    avg_latency_ms: float = 0


# ─────────────────── 通用模型 ───────────────────
# 公共的基础模型，适用于所有 API 接口的通用响应结构。

class HealthResponse(BaseModel):
    """
    健康检查响应模型
    
    用于 /health 或 /status 端点，返回系统的运行状态信息。
    前端或监控系统可定期调用此接口判断服务是否正常运行。
    """
    # 系统运行状态（如 "ok"、"degraded"、"error"）
    status: str
    # 系统版本号，便于运维确认当前部署的版本
    version: str
    # 当前系统中已加载的知识库数量
    kb_count: int = 0
    # 当前使用的 LLM 模型名称（如 "gpt-4"、"chatglm3-6b" 等）
    model: str = ""


class APIError(BaseModel):
    """
    统一错误响应模型
    
    当 API 请求处理过程中发生错误时，返回此结构化的错误信息。
    遵循 RESTful API 的错误响应最佳实践。
    """
    # 错误的详细描述信息，便于开发者或用户理解错误原因
    detail: str
    # 可选的错误代码，用于程序化地识别错误类型
    # 例如："KB_NOT_FOUND"、"INVALID_INPUT"、"LLM_TIMEOUT" 等
    error_code: Optional[str] = None
