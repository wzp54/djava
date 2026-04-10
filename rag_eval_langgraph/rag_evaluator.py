"""
RAG 系统自动化评测流水线 - LangGraph 版本
使用原项目的检索、重排、生成模块，并使用阿里百炼作为 Judge。
"""

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, TypedDict

from openai import OpenAI

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, APP_DIR)

from langgraph.graph import END, StateGraph
from model_api import resolve_judge_api_key, resolve_judge_base_url, resolve_judge_model
from qwen_sglang import generate
from retriever import hybrid_retrieve
from try_reraken import rerank

JUDGE_API_KEY = resolve_judge_api_key()
JUDGE_BASE_URL = resolve_judge_base_url()
JUDGE_MODEL = resolve_judge_model("qwen-plus")
EVAL_DATASET_PATH = "eval_dataset.json"
REPORT_PATH = "evaluation_report.csv"

judge_client = OpenAI(
    api_key=JUDGE_API_KEY,
    base_url=JUDGE_BASE_URL,
)


class EvaluationState(TypedDict):
    query: str
    retrieved_context: str
    retrieved_docs: List[dict]
    system_answer: str
    faithfulness: int
    answer_relevance: int
    overall_effectiveness: int
    reason: str
    error: str
    prompt: str


@dataclass
class EvaluationResult:
    query: str
    retrieved_context: str
    system_answer: str
    faithfulness: int
    answer_relevance: int
    overall_effectiveness: int
    reason: str


def retrieve_docs(state: EvaluationState) -> EvaluationState:
    query = state["query"]
    try:
        print(f"[检索] 正在查询: {query[:30]}...")
        docs = hybrid_retrieve(query, top_k=5)
        if docs:
            context_parts = []
            for index, doc in enumerate(docs, 1):
                content = doc.get("content", "")[:500]
                context_parts.append(f"【文档{index}】{content}")
            state["retrieved_context"] = "\n\n".join(context_parts)
            state["retrieved_docs"] = docs
            print(f"[检索] 成功检索到 {len(docs)} 个文档")
        else:
            state["retrieved_context"] = "未检索到相关文档。"
            state["retrieved_docs"] = []
            print("[检索] 未检索到相关文档")
    except Exception as exc:
        print(f"[检索] 失败: {exc}")
        state["retrieved_context"] = f"检索失败: {exc}"
        state["retrieved_docs"] = []
        state["error"] = f"检索失败: {exc}"
    return state


def rerank_docs(state: EvaluationState) -> EvaluationState:
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    if not docs:
        return state

    try:
        doc_contents = [doc.get("content", "") for doc in docs]
        print(f"[重排序] 正在处理 {len(doc_contents)} 个文档...")
        rerank_results = rerank(query, doc_contents, top_n=3)

        reranked_docs = []
        for item in rerank_results:
            index = item["index"]
            score = item.get("relevance_score", item.get("score", 0))
            reranked_docs.append({**docs[index], "rerank_score": score})

        state["retrieved_docs"] = reranked_docs
        state["retrieved_context"] = "\n\n".join(
            f"【文档{i}】(相关度 {doc.get('rerank_score', 0):.2f}) {doc.get('content', '')[:500]}"
            for i, doc in enumerate(reranked_docs, 1)
        )
        print(
            f"[重排序] 完成，前3个文档相关度: "
            f"{[item.get('relevance_score', item.get('score', 0)) for item in rerank_results]}"
        )
    except Exception as exc:
        print(f"[重排序] 失败: {exc}")
        state["error"] = f"重排序失败: {exc}"
    return state


def generate_answer(state: EvaluationState) -> EvaluationState:
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    try:
        print("[生成] 正在调用阿里百炼生成回答...")
        answer = generate(query, chunks=docs, history=[])
        state["system_answer"] = answer
        print(f"[生成] 回答生成完成，长度: {len(answer)}")
    except Exception as exc:
        print(f"[生成] 失败: {exc}")
        state["system_answer"] = f"生成失败: {exc}"
        state["error"] = f"生成失败: {exc}"
    return state


def evaluate_answer(state: EvaluationState) -> EvaluationState:
    state["prompt"] = f"""你是一个极其严格的 RAG 系统评测专家。

请根据以下内容从三个维度打分：
1. faithfulness：事实一致性
2. answer_relevance：回答相关性
3. overall_effectiveness：综合有效性

【用户问题】
{state["query"]}

【检索到的知识库内容】
{state["retrieved_context"]}

【AI 回答】
{state["system_answer"]}

请仅输出 JSON：
{{
  "faithfulness": <1-5整数>,
  "answer_relevance": <1-5整数>,
  "overall_effectiveness": <1-5整数>,
  "reason": "<一句中文理由>"
}}"""
    return state


def parse_evaluation_result(response_text: str) -> dict:
    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        result = json.loads(cleaned)
        return {
            "faithfulness": max(1, min(5, int(result.get("faithfulness", 3)))),
            "answer_relevance": max(1, min(5, int(result.get("answer_relevance", 3)))),
            "overall_effectiveness": max(
                1, min(5, int(result.get("overall_effectiveness", 3)))
            ),
            "reason": result.get("reason", "无评分理由"),
        }
    except Exception:
        import re

        numbers = re.findall(r"\d+", response_text)
        if len(numbers) >= 3:
            return {
                "faithfulness": max(1, min(5, int(numbers[0]))),
                "answer_relevance": max(1, min(5, int(numbers[1]))),
                "overall_effectiveness": max(1, min(5, int(numbers[2]))),
                "reason": "JSON 解析失败，已按数字兜底",
            }
        return {
            "faithfulness": 3,
            "answer_relevance": 3,
            "overall_effectiveness": 3,
            "reason": "解析失败",
        }


def call_llm_judge(state: EvaluationState) -> EvaluationState:
    try:
        print("[评测] 正在调用阿里百炼 Judge...")
        response = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "你是一个严格的专业评测助手。"},
                {"role": "user", "content": state["prompt"]},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        response_text = response.choices[0].message.content or ""
        result = parse_evaluation_result(response_text)
        state["faithfulness"] = result["faithfulness"]
        state["answer_relevance"] = result["answer_relevance"]
        state["overall_effectiveness"] = result["overall_effectiveness"]
        state["reason"] = result["reason"]
        print(
            f"[评测] 得分: 事实一致性 {result['faithfulness']}, "
            f"回答相关性 {result['answer_relevance']}, "
            f"综合有效性 {result['overall_effectiveness']}"
        )
    except Exception as exc:
        print(f"[评测] 失败: {exc}")
        state["faithfulness"] = -1
        state["answer_relevance"] = -1
        state["overall_effectiveness"] = -1
        state["reason"] = f"评测失败: {exc}"
    return state


def load_eval_dataset(path: str) -> list:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_report(results: list, path: str):
    fieldnames = [
        "query",
        "retrieved_context",
        "system_answer",
        "faithfulness",
        "answer_relevance",
        "overall_effectiveness",
        "reason",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "query": item.query,
                    "retrieved_context": item.retrieved_context,
                    "system_answer": item.system_answer,
                    "faithfulness": item.faithfulness,
                    "answer_relevance": item.answer_relevance,
                    "overall_effectiveness": item.overall_effectiveness,
                    "reason": item.reason,
                }
            )
    print(f"\n[INFO] 报告已保存: {path}")


def calculate_average_scores(results: list) -> dict:
    valid = [item for item in results if item.faithfulness > 0]
    if not valid:
        return {
            "faithfulness": 0,
            "answer_relevance": 0,
            "overall_effectiveness": 0,
        }
    count = len(valid)
    return {
        "faithfulness": round(sum(item.faithfulness for item in valid) / count, 2),
        "answer_relevance": round(
            sum(item.answer_relevance for item in valid) / count, 2
        ),
        "overall_effectiveness": round(
            sum(item.overall_effectiveness for item in valid) / count, 2
        ),
    }


def build_evaluation_graph() -> StateGraph:
    workflow = StateGraph(EvaluationState)
    workflow.add_node("retrieve", retrieve_docs)
    workflow.add_node("rerank", rerank_docs)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("evaluate", evaluate_answer)
    workflow.add_node("judge", call_llm_judge)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", "evaluate")
    workflow.add_edge("evaluate", "judge")
    workflow.add_edge("judge", END)
    return workflow.compile()


def main():
    print("=" * 60)
    print("RAG 系统自动化评测流水线 - LangGraph 版本")
    print("使用原项目模块: retriever + rerank + qwen + 阿里百炼 Judge")
    print("=" * 60)

    if not JUDGE_API_KEY or JUDGE_API_KEY == "YOUR_API_KEY_HERE":
        print("\n[ERROR] 请配置 `JUDGE_API_KEY` 或 `LLM_API_KEY`")
        return

    app = build_evaluation_graph()
    print("\n[INFO] LangGraph 工作流已构建")
    print("  流程: retrieve -> rerank -> generate -> evaluate -> judge")

    try:
        eval_cases = load_eval_dataset(EVAL_DATASET_PATH)
        print(f"[INFO] 成功加载 {len(eval_cases)} 个测试用例")
    except FileNotFoundError:
        print(f"[ERROR] 文件不存在: {EVAL_DATASET_PATH}")
        return

    all_results: List[EvaluationResult] = []
    for index, case in enumerate(eval_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"[{index}/{len(eval_cases)}] {case['query'][:40]}...")
        print(f"{'=' * 60}")

        initial_state: EvaluationState = {
            "query": case.get("query", ""),
            "retrieved_context": "",
            "retrieved_docs": [],
            "system_answer": "",
            "faithfulness": 0,
            "answer_relevance": 0,
            "overall_effectiveness": 0,
            "reason": "",
            "error": "",
            "prompt": "",
        }

        try:
            final_state = app.invoke(initial_state)
            all_results.append(
                EvaluationResult(
                    query=case["query"],
                    retrieved_context=final_state.get("retrieved_context", ""),
                    system_answer=final_state.get("system_answer", ""),
                    faithfulness=final_state.get("faithfulness", -1),
                    answer_relevance=final_state.get("answer_relevance", -1),
                    overall_effectiveness=final_state.get("overall_effectiveness", -1),
                    reason=final_state.get("reason", ""),
                )
            )
        except Exception as exc:
            print(f"[ERROR] 执行失败: {exc}")
            all_results.append(
                EvaluationResult(
                    query=case["query"],
                    retrieved_context="",
                    system_answer="",
                    faithfulness=-1,
                    answer_relevance=-1,
                    overall_effectiveness=-1,
                    reason=f"执行失败: {exc}",
                )
            )

        if index < len(eval_cases):
            time.sleep(1)

    avg_scores = calculate_average_scores(all_results)
    print("\n" + "=" * 60)
    print("评测结果汇总")
    print("=" * 60)
    print(f"总测试用例数: {len(all_results)}")
    print(f"平均事实一致性: {avg_scores['faithfulness']}/5")
    print(f"平均回答相关性: {avg_scores['answer_relevance']}/5")
    print(f"平均综合有效性: {avg_scores['overall_effectiveness']}/5")

    overall_score = (
        avg_scores["faithfulness"]
        + avg_scores["answer_relevance"]
        + avg_scores["overall_effectiveness"]
    ) / 3
    print(f"综合评分: {overall_score:.2f}/5")

    save_report(all_results, REPORT_PATH)
    print("\n[完成]")


if __name__ == "__main__":
    main()
