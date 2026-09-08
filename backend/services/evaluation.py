"""
RAG 评估模块 - 基于 LLM-as-Judge 的自动质量评估
评估三个维度: Faithfulness / Answer Relevancy / Context Relevancy
"""
import time
import logging
from typing import List, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger("qa.evaluation")


# ──────── 评估提示词 ────────

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
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
        self.llm = llm

    async def _safe_score(self, chain, inputs: dict, default: float = 0.5) -> float:
        """安全打分：LLM 输出解析失败时返回默认值"""
        try:
            result = await chain.ainvoke(inputs)
            score = float(result.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Score parsing failed, fallback to {default}: {e}")
            return default

    async def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """评估忠实度：answer 中的陈述是否都能从 contexts 推导"""
        if not contexts:
            return 0.0
        context_text = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
        chain = FAITHFULNESS_PROMPT | self.llm | StrOutputParser()
        return await self._safe_score(chain, {"context": context_text, "answer": answer})

    async def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """评估答案相关性：answer 是否切中 question"""
        chain = ANSWER_RELEVANCY_PROMPT | self.llm | StrOutputParser()
        return await self._safe_score(chain, {"question": question, "answer": answer})

    async def evaluate_context_relevancy(self, question: str, contexts: List[str]) -> float:
        """评估上下文相关性：contexts 是否与 question 相关"""
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
        """完整评估：返回三指标"""
        start = time.time()

        faithfulness = await self.evaluate_faithfulness(answer, contexts)
        answer_relevancy = await self.evaluate_answer_relevancy(question, answer)
        context_relevancy = await self.evaluate_context_relevancy(question, contexts)

        latency_ms = (time.time() - start) * 1000

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
        """
        if ground_truths is not None and len(ground_truths) != len(questions):
            raise ValueError("ground_truths 数量必须与 questions 一致")

        results = []

        for i, question in enumerate(questions):
            answer = ""
            raw_contexts = []
            ground_truth = ground_truths[i] if ground_truths else None

            if rag_fn is not None:
                answer, raw_contexts = await rag_fn(question)
                raw_contexts = raw_contexts or []
            elif answers is not None:
                answer = answers[i] if i < len(answers) else ""
                if contexts_list and i < len(contexts_list):
                    raw_contexts = contexts_list[i] or []

            # 兼容检索来源 dict（含 content）与纯文本两种格式
            contexts = [
                c["content"] if isinstance(c, dict) else str(c)
                for c in raw_contexts
            ]

            results.append(await self.evaluate(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            ))

        def _avg(key: str) -> Optional[float]:
            values = [r[key] for r in results if r.get(key) is not None]
            return round(sum(values) / len(values), 4) if values else None

        latencies = [r.get("latency_ms") or 0 for r in results]

        return {
            "results": results,
            "avg_faithfulness": _avg("faithfulness"),
            "avg_answer_relevancy": _avg("answer_relevancy"),
            "avg_context_relevancy": _avg("context_relevancy"),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 0) if latencies else 0,
        }
