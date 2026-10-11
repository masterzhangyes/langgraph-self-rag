"""
RAG 评估模块 - 基于 LLM-as-Judge 的自动质量评估
================================================

评估三个维度: Faithfulness / Answer Relevancy / Context Relevancy

什么是 LLM-as-Judge（为什么不用人工评估）:
    人工评估准确但昂贵、慢、不可重复；本模块改用另一个 LLM 充当"裁判"，
    按照给定的评分标准（Rubric）对 RAG 的产出逐项打分 ——
    牺牲一点客观性，换来可批量、可回归、可持续监控的自动化评估。

三指标含义（对应 RAG 的三个环节）:
    ┌────────────────┬──────────────────────────┬──────────────────┐
    │ 指标            │ 回答的问题                │ 定位故障环节      │
    ├────────────────┼──────────────────────────┼──────────────────┤
    │ Context Relev. │ 检索到的文档和问题相关吗？  │ 检索器（召回质量） │
    │ Faithfulness   │ 回答都是上下文里有依据的吗？ │ 生成器（幻觉程度） │
    │ Answer Relev.  │ 回答切中用户问题了吗？      │ 生成器（对题程度） │
    └────────────────┴──────────────────────────┴──────────────────┘
    排查思路: Context 低 → 先修检索；Context 高但 Faithfulness 低 → 模型在编造；
    两者都高但 Answer Relevancy 低 → 提示词/生成策略问题。

与 RAGAS 等框架的关系:
    指标定义参考了 RAGAS 论文，但这里是轻量自实现 ——
    不引入额外依赖，裁判提示词完全可控可调。

模块依赖:
    · langchain_core   LCEL 链（prompt | llm | parser）构造裁判调用
"""
import time
import logging
from typing import List, Dict, Optional

# ChatPromptTemplate: 声明式对话模板（system/human 双角色）
# StrOutputParser:     把 LLM 的消息输出剥离成纯字符串
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 独立命名空间，日志中与其他 qa.* 模块区分
logger = logging.getLogger("qa.evaluation")


# ──────── 评估提示词 ────────
# 三个裁判提示词遵循同一套设计范式（读代码时对照着看）:
#   1. system 先给"角色设定"（严格的事实核查员/评估者）—— 约束打分立场；
#   2. 再给"分档评分标准"(Rubric) —— 每一档都有具体描述作锚点，
#      避免 LLM 打分忽高忽低（没有锚点时 LLM 倾向给中庸分）；
#   3. 最后强制"只输出一个 0-1 浮点数，不要解释" ——
#      保证下游 float(result.strip()) 能稳定解析，无需结构化输出解析器；
#   4. human 消息按 "输入材料 → 评分请求" 的顺序排版。

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    # 忠实度: 衡量"幻觉"程度 —— 回答里的每句话是否都能在上下文中找到依据。
    # 注意评估对象是 (context, answer) 对，问题本身不参与（有依据但答非所问
    # 的情况由 Answer Relevancy 负责，各指标职责单一）
    ("system", """你是一个严格的事实核查员。判断回答中的每一句是否都能从给定的上下文信息中推导出来。

评分标准 (0-1 之间的浮点数):
- 1.0: 回答中的所有陈述都能在上下文中找到直接依据
- 0.7-0.9: 大部分陈述有依据，少量合理推断
- 0.4-0.6: 部分有依据，部分可能是编造的
- 0.1-0.3: 大部分无依据，可能大量编造
- 0.0: 完全编造，上下文无法支持任何陈述

只输出一个 0-1 之间的浮点数，不要额外解释。"""),
    ("human", """上下文信息:
{context}

回答:
{answer}

忠实度分数:"""),
])

ANSWER_RELEVANCY_PROMPT = ChatPromptTemplate.from_messages([
    # 答案相关性: 回答是否"切题且完整"。只看 (question, answer)，
    # 不需要上下文 —— 即使回答全是编造但很切题，此项也可能高分
    # （编造问题由 Faithfulness 单独抓）
    ("system", """评估回答与问题的相关性。回答是否直接、完整地回应了问题？

评分标准 (0-1 之间的浮点数):
- 1.0: 回答完全切中问题，没有无关内容
- 0.7-0.9: 回答主要回应了问题，有少量无关信息
- 0.4-0.6: 部分回应了问题，但偏离较多
- 0.1-0.3: 只有很少部分与问题相关
- 0.0: 回答完全无关

只输出一个 0-1 之间的浮点数，不要额外解释。"""),
    ("human", """问题: {question}

回答: {answer}

答案相关性分数:"""),
])

CONTEXT_RELEVANCY_PROMPT = ChatPromptTemplate.from_messages([
    # 上下文相关性: 检索质量的直接指标。
    # 若此项持续偏低 → 该去修 HybridRetriever（换 Embedding 模型/
    # 调 Top-K/优化切片），而不是调生成端
    ("system", """评估检索到的上下文文档与问题的相关性。这些文档是否有助于回答问题？

评分标准 (0-1 之间的浮点数):
- 1.0: 所有文档都高度相关，可以直接用于回答问题
- 0.7-0.9: 大部分文档相关，少量不相关
- 0.4-0.6: 部分文档相关，很多不相关
- 0.1-0.3: 只有极少文档与问题稍有关联
- 0.0: 所有文档都与问题无关

只输出一个 0-1 之间的浮点数，不要额外解释。"""),
    ("human", """问题: {question}

检索到的上下文文档:
{context}

上下文相关性分数:"""),
])


class RAGEvaluator:
    """RAG 质量评估器 (LLM-as-Judge)"""

    def __init__(self, llm):
        """
        Args:
            llm: 裁判模型（LangChain 聊天模型实例，需支持 ainvoke）。
                由外部注入 —— 生产上通常用与生成相同的 LLM，
                严谨场景可换更强的模型当裁判（裁判水平决定评估上限）。
        """
        self.llm = llm

    async def _safe_score(self, chain, inputs: dict, default: float = 0.5) -> float:
        """
        安全打分：LLM 输出解析失败时返回默认值。

        统一收口三个风险点，让所有评估方法共享同一套防御:
            1. 网络/API 异常        → ainvoke 抛错
            2. 裁判没听话，输出了    → float() 抛 ValueError
               "评分：0.8" 之类的文本
            3. 输出越界数字（如 1.2）→ clamp 到 [0, 1]

        default=0.5（中性分）而非 0.0:
            单次打分失败不应让该样本被误判为"最差"，
            中性值对均值的扰动最小。
        """
        try:
            result = await chain.ainvoke(inputs)
            # clamp: max(0, min(1, x)) 双向截断
            score = float(result.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            # 只告警不抛错：评估是旁路功能，绝不能把主链路打挂
            logger.warning(f"Score parsing failed, fallback to {default}: {e}")
            return default

    async def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        评估忠实度：answer 中的陈述是否都能从 contexts 推导。

        空上下文直接判 0 分 —— 没有任何依据可谈，回答必然是无本之木。
        （这也是一种信号: Context Relevancy/召回环节出了问题）
        """
        if not contexts:
            return 0.0
        # 给每篇文档编号 [1] [2] ...，方便裁判逐篇对照核查依据
        context_text = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
        # LCEL 管道: 模板填充 → 裁判模型 → 剥成纯字符串
        chain = FAITHFULNESS_PROMPT | self.llm | StrOutputParser()
        return await self._safe_score(chain, {"context": context_text, "answer": answer})

    async def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """评估答案相关性：answer 是否切中 question"""
        chain = ANSWER_RELEVANCY_PROMPT | self.llm | StrOutputParser()
        return await self._safe_score(chain, {"question": question, "answer": answer})

    async def evaluate_context_relevancy(self, question: str, contexts: List[str]) -> float:
        """评估上下文相关性：contexts 是否与 question 相关"""
        # 同 faithfulness: 空上下文 = 检索一无所获，判 0 分
        if not contexts:
            return 0.0
        context_text = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
        chain = CONTEXT_RELEVANCY_PROMPT | self.llm | StrOutputParser()
        return await self._safe_score(chain, {"question": question, "context": context_text})

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None,
    ) -> dict:
        """
        完整评估：返回三指标（单条样本的评估单元）。

        Args:
            question:     用户问题
            answer:       RAG 生成的回答
            contexts:     检索到的上下文文本列表
            ground_truth: 可选的标准答案。
                          ⚠️ 当前三指标均未使用它 —— 预留字段，
                          供后续扩展（如与标准答案的语义相似度对比）。
        """
        # 端到端计时: 覆盖三次裁判调用，用于观察评估本身的成本
        start = time.time()

        # 三指标依次串行调用（裁判间无依赖，可 asyncio.gather 并发提速，
        # 当前实现保守串行，避免触发 LLM 服务的并发限流）
        faithfulness = await self.evaluate_faithfulness(answer, contexts)
        answer_relevancy = await self.evaluate_answer_relevancy(question, answer)
        context_relevancy = await self.evaluate_context_relevancy(question, contexts)

        latency_ms = (time.time() - start) * 1000

        # round(…, 4): 分数保留 4 位小数，报告足够精确且 JSON 简洁
        return {
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "faithfulness": round(faithfulness, 4),
            "answer_relevancy": round(answer_relevancy, 4),
            "context_relevancy": round(context_relevancy, 4),
            "latency_ms": round(latency_ms, 0),
        }

    async def batch_evaluate(
        self,
        questions: List[str],
        ground_truths: List[str] = None,
        rag_fn=None,
        answers: List[str] = None,
        contexts_list: List[List[str]] = None,
    ) -> dict:
        """
        批量评估：对每个问题执行 (可选检索生成) + 三指标打分，并聚合平均分。

        数据来源（两种方式，至少提供一种）:
          - rag_fn:  异步回调，签名 `async (q) -> (answer, contexts)`
                     contexts 可为 dict 列表（含 content 字段，如检索来源）或纯文本列表。
                     通常由上层调用方注入（内部调用 chat()）。
          - answers + contexts_list: 直接提供已生成的答案与上下文，跳过检索生成。

        两种模式的用途:
            · rag_fn 模式     → "在线评估": 真实跑一遍检索+生成，评估全链路
            · answers 模式    → "离线回放": 只对历史已有的 (q, a, ctx) 打分，
                                省去重复生成，适合对线上日志做质量回溯

        Args:
            questions: 待评估的问题列表
            ground_truths: 可选，与 questions 一一对应的标准答案
            rag_fn: 可选异步函数，负责检索 + 生成
            answers: 可选，直接提供答案（与 questions 等长）
            contexts_list: 可选，直接提供每个问题的上下文列表

        Returns:
            {
                "results": [单条评估结果...],
                "avg_faithfulness": float|None,
                "avg_answer_relevancy": float|None,
                "avg_context_relevancy": float|None,
                "avg_latency_ms": float
            }

        ⚠️ 注意: 平均分是"宏平均"(对每条样本的分数求平均)，
           而非把所有上下文拼在一起评一次 —— 样本间互不干扰。
        """
        # 前置校验: 标准答案数量必须能和问题对齐，错了早失败好过静默错位
        if ground_truths is not None and len(ground_truths) != len(questions):
            raise ValueError("ground_truths 数量必须与 questions 一致")

        results = []

        # 逐题串行处理（题目间相互独立，同样可并发优化，见 evaluate 注释）
        for i, question in enumerate(questions):
            answer = ""
            raw_contexts = []
            ground_truth = ground_truths[i] if ground_truths else None

            # ── 获取本条样本的 (answer, contexts) ──
            if rag_fn is not None:
                # 在线模式: 现场跑一遍 RAG。rag_fn 返回
                # (answer, contexts)；contexts or [] 兜底 None
                answer, raw_contexts = await rag_fn(question)
                raw_contexts = raw_contexts or []
            elif answers is not None:
                # 离线回放模式: 直接取调用方给的数据。
                # i < len(...) 的越界防御保证短列表时降级为空串而非 IndexError
                answer = answers[i] if i < len(answers) else ""
                if contexts_list and i < len(contexts_list):
                    raw_contexts = contexts_list[i] or []

            # 兼容检索来源 dict（含 content）与纯文本两种格式
            # 上游 search_with_source() 返回 [{"content":…, "source":…}]，
            # 这里归一化成纯文本列表，让三个评估方法只面对一种类型
            contexts = [
                c["content"] if isinstance(c, dict) else str(c)
                for c in raw_contexts
            ]

            # 单条样本的完整三指标评估（见 evaluate）
            results.append(await self.evaluate(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            ))

        def _avg(key: str) -> Optional[float]:
            """内部工具: 求某指标的平均分；无样本时返回 None（而非除零崩溃）"""
            values = [r[key] for r in results if r.get(key) is not None]
            return round(sum(values) / len(values), 4) if values else None

        # 延迟单独聚合: or 0 把 None 归零，保证延迟恒有数值
        # （延迟不像分数有"无意义"的情况，0 即代表未测到）
        latencies = [r.get("latency_ms") or 0 for r in results]

        return {
            "results": results,
            "avg_faithfulness": _avg("faithfulness"),
            "avg_answer_relevancy": _avg("answer_relevancy"),
            "avg_context_relevancy": _avg("context_relevancy"),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 0) if latencies else 0,
        }
