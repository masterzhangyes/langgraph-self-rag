"""
Self-RAG Agent - 基于 LangGraph 的自检式 RAG 问答代理
====================================================

本模块实现了 Self-RAG（Self-Reflective Retrieval-Augmented Generation）架构，
通过 LangGraph 状态图构建了一条包含 6 个处理节点和 3 个条件分支的智能问答流水线。

核心思想：传统的 RAG 流程是「检索 → 生成」的单向流程，无法判断检索质量
和生成准确性。Self-RAG 在此基础上引入了「反思」机制，形成闭环：
  1. 检索后自动评估文档相关性 → 不相关则改写查询重新检索
  2. 生成后自动检测幻觉（是否忠于原文）→ 有幻觉则重新生成

整体流程图:
    expand_query → retrieve → grade_relevance
      → (不相关 + 重试<上限) transform_query → retrieve（循环）
      → (相关) generate
        → (SelfRAG=on)  grade_hallucination
          → (有幻觉 + 重试<上限) → generate（循环）
          → (无幻觉) → END
        → (SelfRAG=off) → END
"""

import json
import logging
# typing 模块提供类型提示支持：
#   - List: 列表类型    - Dict: 字典类型    - Any: 任意类型
#   - TypedDict: 带类型标注的字典，用于定义 Agent 状态结构
#   - Optional: 可选类型，表示字段可以为 None
from typing import List, Dict, Any, TypedDict, Optional

# LangGraph 核心组件：
#   - StateGraph: 状态图类，用于构建基于状态的有向工作流
#   - END: 特殊标记节点，表示工作流的终止点
from langgraph.graph import StateGraph, END
# LangChain 输出解析器：
#   - StrOutputParser: 将 LLM 的输出解析为纯字符串
from langchain_core.output_parsers import StrOutputParser

# 从核心配置模块导入全局配置（包含 MAX_RETRIEVAL_RETRIES、ENABLE_SELF_RAG 等）
from core.config import settings

# 创建本模块的日志记录器，名称为 "qa.agent"，
# 便于在日志输出中区分来自不同模块的日志信息
logger = logging.getLogger("qa.agent")


# ──────── Agent 状态定义 ────────
# AgentState 定义了在整个 LangGraph 工作流中流转的数据结构。
# 每个节点读取状态中的字段作为输入，处理后将新字段写回状态。
# 使用 TypedDict + total=False 使得所有字段都是可选的，
# 因为不同节点只需要读写自己关心的字段。

class AgentState(TypedDict, total=False):
    """
    Agent 全局状态定义
    
    在 LangGraph 的各个节点之间传递的共享数据结构。
    每个节点从 state 中读取所需字段，处理后将结果字段写回 state。
    total=False 表示所有字段都是可选的，节点可以只更新部分字段。
    
    数据流向:
        输入阶段: query, history
        → 扩展阶段: expanded_queries
        → 检索阶段: retrieved_docs
        → 评估阶段: relevance_scores, retry_count
        → 生成阶段: answer
        → 幻觉检测: hallucination_score, regeneration_count
        → 最终输出: final_answer
    """
    # ── 原始输入字段 ──
    # 用户本次提出的问题文本
    query: str
    # 多轮对话的历史记录列表，每条记录包含 role 和 content
    history: List[dict]
    # ── 查询扩展阶段字段 ──
    # 经 LLM 扩展后的多个搜索查询列表（如将 "什么是RAG" 扩展为
    # ["RAG定义", "检索增强生成原理", "RAG流程详解"]）
    expanded_queries: List[str]
    # ── 检索阶段字段 ──
    # 从知识库中检索到的文档列表，每个文档为字典格式，
    # 至少包含 "content"（文本内容）和 "source"（来源）
    retrieved_docs: List[Dict[str, Any]]
    # ── 相关性评估阶段字段 ──
    # 每条检索文档的相关性评分列表（1.0=相关，0.0=不相关）
    relevance_scores: List[float]
    # 检索重试计数器，记录已经重试了多少次查询改写
    retry_count: int
    # ── 生成阶段字段 ──
    # LLM 生成的回答文本
    answer: str
    # ── 幻觉检测阶段字段 ──
    # 忠实度评分（0~1），衡量回答是否忠于检索到的上下文
    # 分数越高表示幻觉越少
    hallucination_score: float
    # 重新生成计数器，记录因幻觉检测不通过而重新生成的次数
    regeneration_count: int
    # ── 最终输出字段 ──
    # 经过所有评估环节后确定的最终回答
    final_answer: str


class SelfRAGAgent:
    """
    Self-RAG（自反思检索增强生成）代理
    
    核心类，封装了整个 Self-RAG 工作流。通过 LangGraph 的 StateGraph
    将 6 个处理节点和 3 个条件分支组装成一个可执行的有向图。
    
    6 个处理节点:
        1. expand_query      - 查询扩展：将用户问题扩展为多个搜索角度
        2. retrieve          - 文档检索：用多个查询从知识库中检索文档并去重
        3. grade_relevance   - 相关性评估：用 LLM 判断每条检索文档是否与问题相关
        4. transform_query   - 查询改写：当相关性不足时，改写查询以提升检索效果
        5. generate          - 回答生成：基于检索到的文档和对话历史生成回答
        6. grade_hallucination - 幻觉检测：检测回答是否包含知识库中不存在的信息
    
    3 个条件分支:
        1. _should_retry:              相关性不足 → 改写查询重试 / 继续生成
        2. _should_check_hallucination: SelfRAG 开启 → 幻觉检测 / 直接输出
        3. _is_hallucinating:          检测到幻觉 → 重新生成 / 输出最终结果
    
    流程示意:
        expand_query → retrieve → grade_relevance
          → (不相关 + 重试<上限) transform_query → retrieve（循环）
          → (相关) generate
            → (SelfRAG=on)  grade_hallucination
              → (有幻觉 + 重试<上限) → generate（循环）
              → (无幻觉) → END
            → (SelfRAG=off) → END
    """

    def __init__(self, llm, retriever_fn, generate_fn, evaluator=None):
        """
        初始化 Self-RAG 代理
        
        Args:
            llm: LangChain ChatModel 实例
                用于执行查询扩展、相关性评估、查询改写、幻觉检测等 LLM 调用。
                需要支持 ainvoke 异步调用。
            retriever_fn: 异步检索回调函数
                签名: async fn(query: str) -> List[dict]
                接收查询字符串，返回文档列表，每个文档至少包含 "content" 字段。
            generate_fn: 异步生成回调函数
                签名: async fn(query: str, docs: List[dict], history: List[dict]) -> str
                接收问题、检索文档和历史记录，返回生成的回答文本。
            evaluator: RAGEvaluator 实例（可选）
                用于幻觉检测的评估器，需提供 evaluate_faithfulness 方法。
                如果为 None 且 SelfRAG 未开启，则跳过幻觉检测步骤。
        """
        self.llm = llm                    # LLM 模型实例
        self.retrieve = retriever_fn       # 检索回调
        self.generate = generate_fn        # 生成回调
        self.evaluator = evaluator         # 评估器（可选）
        self.graph = self._build_graph()   # 构建并编译 LangGraph 状态图

    # ──────── LangGraph 图构建 ────────
    # 以下方法负责将 6 个节点和 3 个条件分支组装成完整的状态图。
    # 图在 __init__ 时构建并编译，之后在 run() 中执行。

    def _build_graph(self) -> StateGraph:
        """
        构建并编译 LangGraph 状态图
        
        将 6 个处理节点注册到 StateGraph 中，通过有向边和条件边
        定义节点间的执行顺序和分支逻辑。
        
        Returns:
            编译后的 LangGraph 可执行图
        """
        # 创建以 AgentState 为状态类型的状态图
        workflow = StateGraph(AgentState)

        # ── 注册 6 个处理节点 ──
        # 每个节点绑定一个异步处理方法，节点名称即为方法的功能标识
        workflow.add_node("expand_query", self._expand_query)          # 节点1: 查询扩展
        workflow.add_node("retrieve", self._retrieve)                  # 节点2: 文档检索
        workflow.add_node("grade_relevance", self._grade_relevance)    # 节点3: 相关性评估
        workflow.add_node("transform_query", self._transform_query)    # 节点4: 查询改写
        workflow.add_node("generate", self._generate)                  # 节点5: 回答生成
        workflow.add_node("grade_hallucination", self._grade_hallucination)  # 节点6: 幻觉检测

        # ── 设置固定边（无条件顺序执行） ──
        # 入口点: 工作流从 expand_query 节点开始执行
        workflow.set_entry_point("expand_query")
        # expand_query → retrieve: 查询扩展完成后立即执行检索
        workflow.add_edge("expand_query", "retrieve")
        # retrieve → grade_relevance: 检索完成后立即进行相关性评估
        workflow.add_edge("retrieve", "grade_relevance")

        # ── 条件边 1: 相关性评估后的分支决策 ──
        # grade_relevance 节点执行完毕后，由 _should_retry 方法判断:
        #   - 返回 "retry"    → 跳转到 transform_query（改写查询后重新检索）
        #   - 返回 "generate"  → 跳转到 generate（相关性足够，直接生成回答）
        workflow.add_conditional_edges(
            "grade_relevance",
            self._should_retry,
            {"retry": "transform_query", "generate": "generate"},
        )
        # transform_query → retrieve: 查询改写后回到检索节点，形成重试循环
        workflow.add_edge("transform_query", "retrieve")

        # ── 条件边 2: 生成回答后的分支决策 ──
        # generate 节点执行完毕后，由 _should_check_hallucination 方法判断:
        #   - 返回 "check"  → 跳转到 grade_hallucination（进行幻觉检测）
        #   - 返回 "output" → 直接结束（SelfRAG 未开启时跳过检测）
        workflow.add_conditional_edges(
            "generate",
            self._should_check_hallucination,
            {"check": "grade_hallucination", "output": END},
        )

        # ── 条件边 3: 幻觉检测后的分支决策 ──
        # grade_hallucination 节点执行完毕后，由 _is_hallucinating 方法判断:
        #   - 返回 "rethink" → 回到 generate 节点重新生成回答
        #   - 返回 "output"  → 结束流程，输出最终回答
        workflow.add_conditional_edges(
            "grade_hallucination",
            self._is_hallucinating,
            {"rethink": "generate", "output": END},
        )

        # 编译图：将定义好的工作流编译为可执行的计算图
        return workflow.compile()

    # ──────── 节点实现 ────────
    # 以下每个方法对应状态图中的一个处理节点。
    # 所有节点方法接收当前 AgentState，返回需要更新的字段字典。
    # LangGraph 会自动将返回的字典合并到 state 中。

    async def _expand_query(self, state: AgentState) -> dict:
        """
        节点1: 查询扩展
        
        利用 LLM 将用户的原始问题扩展为 2-3 个不同角度的搜索查询，
        以提高检索的召回率。例如：
          用户问 "什么是RAG？"
          → 扩展为 ["RAG技术定义", "检索增强生成原理", "RAG流程详解"]
        
        扩展后的查询列表会连同原始查询一起传入检索节点，
        从多个角度搜索知识库，增加找到相关文档的概率。
        
        容错机制: 如果 LLM 调用失败或返回格式不正确，
        则降级为仅使用原始查询进行检索。
        
        Args:
            state: 当前 Agent 状态，至少包含 "query" 字段
        
        Returns:
            包含 "expanded_queries" 字段的字典
        """
        query = state["query"]           # 用户原始问题
        history = state.get("history", [])  # 对话历史（用于上下文理解）

        # 延迟导入，避免模块加载时的循环依赖
        from langchain_core.prompts import ChatPromptTemplate

        # 构建提示词模板：
        # - system: 定义 LLM 的角色为「搜索意图分析专家」，并给出扩展示例
        # - 历史消息: 取最近 4 条对话记录作为上下文（帮助理解指代和省略）
        # - 用户消息: 将当前查询嵌入到提示词中
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是搜索意图分析专家。将用户问题扩展为 2-3 个不同的搜索角度，用 [] 列表格式输出。

示例:
用户: 什么是RAG？
输出: ["RAG技术定义", "检索增强生成原理", "RAG流程详解"]

用户: Python多线程怎么用？
输出: ["Python threading模块用法", "Python多线程注意事项", "Python并发编程实践"]

只输出 JSON 列表，不要其他内容。"""),
            # 将对话历史转换为 LangChain 消息格式
            # 取最近 4 条（[-4:]），避免提示词过长
            # 根据 role 字段判断消息类型：user → human，其他 → assistant
            *[("human" if m["role"] == "user" else "assistant", m["content"]) 
              for m in history[-4:]],
            ("human", f"用户: {query}\n输出搜索角度:"),
        ])

        # 构建 LangChain 调用链: 提示词 → LLM → 字符串解析器
        chain = prompt | self.llm | StrOutputParser()
        try:
            # 异步调用 LLM 生成扩展查询
            result = await chain.ainvoke({})
            # 尝试将 LLM 输出解析为 JSON 列表
            expanded = json.loads(result.strip())
            # 验证解析结果：必须是包含至少一个元素的列表
            if isinstance(expanded, list) and len(expanded) > 0:
                # 将原始查询放在列表首位，确保原始意图也被检索
                expanded = [query] + expanded
            else:
                expanded = [query]
        except Exception as e:
            # 扩展失败时降级：仅使用原始查询
            logger.warning(f"Query expansion failed: {e}")
            expanded = [query]

        logger.info(f"Query expanded: {expanded}")
        # 将扩展后的查询列表写入状态
        return {"expanded_queries": expanded}

    async def _retrieve(self, state: AgentState) -> dict:
        """
        节点2: 多查询检索 + 去重
        
        使用扩展后的多个查询分别从知识库中检索文档，
        然后对所有检索结果进行去重处理，避免同一文档被重复返回。
        
        去重策略: 取文档内容的前 100 个字符作为文档的唯一标识符（doc_id），
        通过 set 进行快速查重。这是一种轻量级的近似去重方式，
        适用于同一文档被不同查询命中时的去重场景。
        
        Args:
            state: 当前 Agent 状态，包含 "expanded_queries" 或 "query"
        
        Returns:
            包含 "retrieved_docs" 字段的字典（去重后的文档列表）
        """
        # 获取扩展查询列表；如果不存在则回退到原始查询
        queries = state.get("expanded_queries", [state["query"]])
        all_docs = []   # 存放所有检索到的文档
        seen = set()    # 用于去重的集合，存储已见过的文档标识

        # 遍历每个扩展查询，分别调用检索回调
        for q in queries:
            docs = await self.retrieve(q)
            for doc in docs:
                # 用文档内容的前 100 个字符作为去重标识
                # 这是一个启发式方法：如果两篇文档前 100 字相同，
                # 大概率是同一文档（或同一文档的不同切片）
                doc_id = doc["content"][:100]
                if doc_id not in seen:
                    seen.add(doc_id)
                    all_docs.append(doc)

        logger.info(f"Retrieved {len(all_docs)} unique docs from {len(queries)} queries")
        # 将去重后的文档列表写入状态
        return {"retrieved_docs": all_docs}

    async def _grade_relevance(self, state: AgentState) -> dict:
        """
        节点3: 相关性评估
        
        使用 LLM 逐条判断检索到的文档是否与用户问题相关。
        这是一种基于 LLM 的相关性评估方式（相比传统的余弦相似度，
        LLM 能更好地理解语义层面的相关性）。
        
        评估策略:
          - 最多评估前 10 条文档（docs[:10]），避免 LLM 调用过多导致延迟
          - 每条文档截取前 500 个字符送入 LLM 判断
          - LLM 只需输出 "yes" 或 "no"，降低生成成本
          - 评估失败的文档给予 0.5 的中间分数（保守处理）
        
        评估结果将写入 relevance_scores，供后续条件分支 _should_retry 使用。
        
        Args:
            state: 当前 Agent 状态，包含 "query" 和 "retrieved_docs"
        
        Returns:
            包含 "relevance_scores" 和 "retry_count" 字段的字典
        """
        query = state["query"]
        docs = state.get("retrieved_docs", [])

        # 如果没有检索到任何文档，直接返回空评分列表
        if not docs:
            return {"relevance_scores": [], "retry_count": state.get("retry_count", 0)}

        from langchain_core.prompts import ChatPromptTemplate

        # 构建相关性判断的提示词模板
        # 要求 LLM 严格输出 "yes" 或 "no"，不做额外解释
        prompt = ChatPromptTemplate.from_messages([
            ("system", """判断文档是否与用户查询相关，只输出 "yes" 或 "no"。

判断标准:
- yes: 文档内容能直接用于回答查询
- no: 文档与查询无关

只输出一个词，不要解释。"""),
            ("human", "查询: {query}\n\n文档: {doc_content}\n\n相关性:"),
        ])

        relevant_count = 0  # 相关文档计数器
        scores = []         # 每条文档的相关性评分列表

        # 逐条评估文档相关性（最多评估 10 条，控制总耗时）
        for doc in docs[:10]:
            try:
                chain = prompt | self.llm | StrOutputParser()
                result = await chain.ainvoke({
                    "query": query,
                    # 截取文档前 500 个字符，足够判断主题相关性，同时节省 token
                    "doc_content": doc["content"][:500],
                })
                # 判断 LLM 输出中是否包含 "yes"（不区分大小写）
                is_relevant = "yes" in result.strip().lower()
                # 相关 → 1.0，不相关 → 0.0
                scores.append(1.0 if is_relevant else 0.0)
                if is_relevant:
                    relevant_count += 1
            except Exception:
                # LLM 调用异常时给予 0.5 的中间分数
                # 既不偏向「重试」也不偏向「生成」，保守处理
                scores.append(0.5)

        logger.info(f"Relevance: {relevant_count}/{len(docs[:10])} relevant")
        return {
            "relevance_scores": scores,
            "retry_count": state.get("retry_count", 0),  # 透传重试计数器
        }

    def _should_retry(self, state: AgentState) -> str:
        """
        条件分支1: 判断是否需要重试检索
        
        根据相关性评估的结果，决定是继续生成回答，还是改写查询后重新检索。
        
        决策逻辑:
          1. 如果没有检索到任何文档（scores 为空）→ 跳过重试，直接生成
             （因为没有文档可评估，重试也无意义）
          2. 计算相关文档占比 = 相关文档数 / 总评估文档数
          3. 如果相关占比 < 30% 且重试次数未达上限 → 返回 "retry"（改写查询重试）
          4. 否则 → 返回 "generate"（相关性足够，继续生成）
        
        Args:
            state: 当前 Agent 状态，包含 "relevance_scores" 和 "retry_count"
        
        Returns:
            "retry"   - 跳转到 transform_query 节点改写查询
            "generate" - 跳转到 generate 节点生成回答
        """
        scores = state.get("relevance_scores", [])
        retry_count = state.get("retry_count", 0)
        # 从全局配置获取最大重试次数
        max_retries = settings.MAX_RETRIEVAL_RETRIES

        # 无文档时跳过重试，直接进入生成（生成时会处理无文档的情况）
        if not scores:
            return "generate"

        # 计算相关文档占比（所有评分的平均值）
        relevant_ratio = sum(scores) / len(scores) if scores else 0

        # 相关文档占比低于 30% 且未超过重试上限 → 改写查询重新检索
        if relevant_ratio < 0.3 and retry_count < max_retries:
            logger.info(f"Low relevance ({relevant_ratio:.2f}), retry {retry_count + 1}/{max_retries}")
            return "retry"

        # 相关性足够或已达重试上限 → 继续生成
        return "generate"

    async def _transform_query(self, state: AgentState) -> dict:
        """
        节点4: 查询改写
        
        当相关性评估结果显示检索质量不佳时，利用 LLM 将原始查询
        改写为更适合搜索的表述，以提高下一轮检索的效果。
        
        改写策略（由 LLM 自主决定）:
          - 替换同义词（如 "RAG" → "检索增强生成"）
          - 使用更宽泛或更具体的表述
          - 分解复杂问题为多个简单子问题
        
        每次改写后 retry_count + 1，防止无限重试循环。
        
        容错机制: 如果 LLM 调用失败，保留原始查询不变。
        
        Args:
            state: 当前 Agent 状态，包含 "query" 和 "retry_count"
        
        Returns:
            包含改写后的 "query" 和递增后的 "retry_count" 的字典
        """
        from langchain_core.prompts import ChatPromptTemplate

        # 构建查询改写的提示词模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是查询改写专家。将用户问题改写为更适合搜索的表述，可以:
- 替换为同义词
- 使用更宽泛或更具体的表述
- 分解复杂问题
只输出改写后的查询，不要解释。"""),
            ("human", "原查询: {query}\n改写后查询:"),
        ])

        chain = prompt | self.llm | StrOutputParser()
        try:
            # 异步调用 LLM 生成改写后的查询
            new_query = (await chain.ainvoke({"query": state["query"]})).strip()
        except Exception:
            # 改写失败时保留原始查询
            new_query = state["query"]

        logger.info(f"Query transformed: '{state['query']}' → '{new_query}'")
        return {
            "query": new_query,                                    # 更新查询为改写后的版本
            "retry_count": state.get("retry_count", 0) + 1,       # 重试计数器 +1
        }

    async def _generate(self, state: AgentState) -> dict:
        """
        节点5: 回答生成
        
        基于检索到的文档和对话历史，调用生成回调函数生成最终回答。
        这是整个流程的核心节点，将检索到的知识转化为面向用户的回答。
        
        生成过程委托给外部传入的 generate_fn 回调，
        该回调通常会将 query、docs、history 组装成提示词后调用 LLM。
        
        Args:
            state: 当前 Agent 状态，包含 "query"、"retrieved_docs"、"history"
        
        Returns:
            包含 "answer" 和 "regeneration_count" 字段的字典
        """
        query = state["query"]                        # 当前查询（可能已被改写）
        docs = state.get("retrieved_docs", [])         # 检索到的文档列表
        history = state.get("history", [])             # 对话历史

        # 调用外部传入的生成回调，生成回答文本
        answer = await self.generate(query, docs, history)

        return {
            "answer": answer,                                         # 生成的回答
            "regeneration_count": state.get("regeneration_count", 0), # 透传重生成计数器
        }

    def _should_check_hallucination(self, state: AgentState) -> str:
        """
        条件分支2: 判断是否需要进行幻觉检测
        
        根据系统配置决定是否启用 Self-RAG 的幻觉检测机制。
        
        决策逻辑:
          - 如果 settings.ENABLE_SELF_RAG 为 True 且评估器已配置
            → 返回 "check"，进入幻觉检测节点
          - 否则 → 返回 "output"，跳过检测直接输出
        
        这意味着幻觉检测是一个可选功能，可以通过配置开关控制。
        在不需要高可靠性的场景下可以关闭以节省 LLM 调用成本。
        
        Args:
            state: 当前 Agent 状态（本方法未使用状态字段，仅依赖配置）
        
        Returns:
            "check"  - 跳转到 grade_hallucination 进行幻觉检测
            "output" - 直接结束流程
        """
        # 同时满足两个条件才启用幻觉检测:
        # 1. 配置中启用了 Self-RAG 功能
        # 2. 初始化时传入了评估器实例
        if settings.ENABLE_SELF_RAG and self.evaluator is not None:
            return "check"
        return "output"

    async def _grade_hallucination(self, state: AgentState) -> dict:
        """
        节点6: 幻觉检测
        
        检测 LLM 生成的回答是否包含「幻觉」——即回答中出现了
        检索文档中不存在的信息。这是 Self-RAG 架构的关键环节，
        用于确保回答的可靠性和事实准确性。
        
        评估方法:
          - 取检索文档中前 5 条的文本内容作为参考上下文
          - 调用评估器的 evaluate_faithfulness 方法计算忠实度分数
          - 分数范围 0~1，越高表示回答越忠于原文
        
        容错处理:
          - 如果没有检索到文档或评估器不可用，默认给予满分 1.0
            （即假定无幻觉，直接通过）
        
        Args:
            state: 当前 Agent 状态，包含 "answer" 和 "retrieved_docs"
        
        Returns:
            包含 "hallucination_score" 和 "regeneration_count" 字段的字典
        """
        answer = state.get("answer", "")
        docs = state.get("retrieved_docs", [])

        # 无参考文档或无评估器时，默认通过（满分表示无幻觉）
        if not docs or not self.evaluator:
            return {"hallucination_score": 1.0}

        # 取前 5 条文档的内容作为参考上下文
        contexts = [doc["content"] for doc in docs[:5]]
        # 调用评估器计算忠实度分数
        score = await self.evaluator.evaluate_faithfulness(answer, contexts)

        logger.info(f"Hallucination score: {score:.4f}")
        return {
            "hallucination_score": score,                                    # 忠实度分数
            "regeneration_count": state.get("regeneration_count", 0),        # 透传重生成计数器
        }

    def _is_hallucinating(self, state: AgentState) -> str:
        """
        条件分支3: 判断是否存在幻觉，决定是否需要重新生成
        
        根据幻觉检测的忠实度分数，决定是接受当前回答还是重新生成。
        
        决策逻辑:
          - 忠实度分数 < 0.6 且重生成次数未达上限
            → 返回 "rethink"，跳回 generate 节点重新生成
          - 否则 → 将当前回答设为 final_answer，返回 "output" 结束流程
        
        阈值说明:
          - 0.6 是幻觉检测的判定阈值
          - 低于此分数说明回答中有较多信息无法在原文中找到依据
          - 重生成次数限制防止无限循环（与检索重试共用 MAX_RETRIEVAL_RETRIES）
        
        Args:
            state: 当前 Agent 状态，包含 "hallucination_score"、"answer" 等
        
        Returns:
            "rethink" - 跳回 generate 节点重新生成回答
            "output"  - 结束流程，输出最终回答
        """
        score = state.get("hallucination_score", 1.0)   # 忠实度分数，默认 1.0（无幻觉）
        retry_count = state.get("regeneration_count", 0) # 已重生成次数
        max_retries = settings.MAX_RETRIEVAL_RETRIES     # 最大重试次数

        # 检测到幻觉（分数低于 0.6）且未超过重生成上限 → 重新生成
        if score < 0.6 and retry_count < max_retries:
            logger.info(f"Hallucination detected (score={score:.2f}), rethinking ({retry_count + 1}/{max_retries})")
            return "rethink"

        # 无幻觉或已达上限 → 将当前回答设为最终答案并结束
        state["final_answer"] = state.get("answer", "")
        return "output"

    # ──────── 运行入口 ────────
    # 以下方法是 Agent 的外部调用接口，负责初始化状态并启动图执行。

    async def run(self, query: str, history: List[dict] = None) -> AgentState:
        """
        运行 Self-RAG Agent 的主入口
        
        接收用户问题和对话历史，初始化 Agent 状态，
        然后启动 LangGraph 状态图执行整个处理流程。
        
        调用示例:
            agent = SelfRAGAgent(llm, retriever_fn, generate_fn)
            result = await agent.run("什么是RAG？", history=[])
            print(result["final_answer"])
        
        Args:
            query: 用户提出的问题文本
            history: 对话历史记录列表，每条包含 "role" 和 "content" 字段
                     默认为 None，会被转换为空列表
        
        Returns:
            AgentState: 完整的最终状态，包含以下关键字段:
                - final_answer: 最终回答文本
                - retrieved_docs: 检索到的文档列表
                - sources: 引用来源信息
        """
        # 初始化 Agent 状态，设置所有计数器的初始值为 0
        initial_state: AgentState = {
            "query": query,             # 用户原始问题
            "history": history or [],   # 对话历史（None 时转为空列表）
            "retry_count": 0,           # 检索重试计数器
            "regeneration_count": 0,    # 重生成计数器
        }

        logger.info(f"Agent started: '{query[:50]}...'")
        # 异步执行编译后的状态图，传入初始状态
        # ainvoke 会按照图中定义的边和条件边自动流转各节点
        result = await self.graph.ainvoke(initial_state)
        logger.info(f"Agent completed")
        # 返回最终的完整状态，调用方可从中提取 final_answer 等字段
        return result
