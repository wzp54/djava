"""
RAG 系统自动化评测脚本
基于 LLM-as-a-Judge 思想，使用阿里百炼模型作为裁判模型
"""

import csv
import json
import time

from openai import OpenAI

from app.model_api import (
    resolve_judge_api_key,
    resolve_judge_base_url,
    resolve_judge_model,
)

JUDGE_API_KEY = resolve_judge_api_key()
JUDGE_BASE_URL = resolve_judge_base_url()
JUDGE_MODEL = resolve_judge_model("qwen-plus")
EVAL_DATASET_PATH = "eval_dataset.json"
REPORT_PATH = "evaluation_report.csv"

judge_client = OpenAI(
    api_key=JUDGE_API_KEY,
    base_url=JUDGE_BASE_URL,
)


def call_judge(prompt: str, system_prompt: str = "你是一个严格的专业评测助手。") -> str:
    response = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=2048,
    )
    return response.choices[0].message.content or ""


def build_evaluation_prompt(query: str, retrieved_context: str, system_answer: str) -> str:
    return f"""你是一个极其严格的 RAG 系统评测专家。你的任务是对 AI 助手的回答进行客观、公正的评估。

请根据以下内容从三个维度打分：
1. faithfulness：事实一致性
2. answer_relevance：回答相关性
3. overall_effectiveness：综合有效性

【用户问题】
{query}

【检索到的知识库内容】
{retrieved_context}

【AI 回答】
{system_answer}

请仅输出 JSON：
{{
  "faithfulness": <1-5整数>,
  "answer_relevance": <1-5整数>,
  "overall_effectiveness": <1-5整数>,
  "reason": "<一句中文理由>"
}}"""


def parse_evaluation_result(response_text: str) -> dict:
    try:
        result = json.loads(response_text.strip())
        return {
            "faithfulness": max(1, min(5, int(result.get("faithfulness", 3)))),
            "answer_relevance": max(1, min(5, int(result.get("answer_relevance", 3)))),
            "overall_effectiveness": max(
                1, min(5, int(result.get("overall_effectiveness", 3)))
            ),
            "reason": result.get("reason", "无评分理由"),
        }
    except json.JSONDecodeError:
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
            "reason": "评分解析失败，使用默认值",
        }


def evaluate_single_case(case: dict) -> dict:
    query = case.get("query", "")
    retrieved_context = case.get("retrieved_context", "")
    system_answer = case.get("system_answer", "")

    print(f"\n{'=' * 60}")
    print(f"正在评测问题: {query[:50]}...")
    print(f"{'=' * 60}")

    prompt = build_evaluation_prompt(query, retrieved_context, system_answer)

    try:
        print("[INFO] 正在调用阿里百炼 Judge...")
        response_text = call_judge(prompt)
        eval_result = parse_evaluation_result(response_text)
        print("[INFO] 评测完成")
        print(f"  - 事实一致性: {eval_result['faithfulness']}/5")
        print(f"  - 回答相关性: {eval_result['answer_relevance']}/5")
        print(f"  - 综合有效性: {eval_result['overall_effectiveness']}/5")
        print(f"  - 理由: {eval_result['reason']}")
        return eval_result
    except Exception as exc:
        print(f"[ERROR] 评测失败: {exc}")
        return {
            "faithfulness": -1,
            "answer_relevance": -1,
            "overall_effectiveness": -1,
            "reason": f"评测失败: {exc}",
        }


def load_eval_dataset(path: str) -> list:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_report(results: list, path: str):
    fieldnames = [
        "query",
        "faithfulness",
        "answer_relevance",
        "overall_effectiveness",
        "reason",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    print(f"\n[INFO] 评测报告已保存至: {path}")


def calculate_average_scores(results: list) -> dict:
    valid_results = [item for item in results if item.get("faithfulness", -1) > 0]
    if not valid_results:
        return {
            "faithfulness": 0,
            "answer_relevance": 0,
            "overall_effectiveness": 0,
        }

    count = len(valid_results)
    return {
        "faithfulness": round(
            sum(item["faithfulness"] for item in valid_results) / count, 2
        ),
        "answer_relevance": round(
            sum(item["answer_relevance"] for item in valid_results) / count, 2
        ),
        "overall_effectiveness": round(
            sum(item["overall_effectiveness"] for item in valid_results) / count, 2
        ),
    }


def main():
    print("=" * 60)
    print("RAG 系统自动化评测流水线")
    print("基于 LLM-as-a-Judge | 裁判模型: 阿里百炼")
    print("=" * 60)

    if not JUDGE_API_KEY or JUDGE_API_KEY == "YOUR_API_KEY_HERE":
        print("\n[ERROR] 请先配置 `JUDGE_API_KEY` 或 `LLM_API_KEY`")
        return

    print(f"\n[INFO] 正在加载评测数据集: {EVAL_DATASET_PATH}")
    try:
        eval_cases = load_eval_dataset(EVAL_DATASET_PATH)
        print(f"[INFO] 成功加载 {len(eval_cases)} 个测试用例")
    except FileNotFoundError:
        print(f"[ERROR] 评测数据集不存在: {EVAL_DATASET_PATH}")
        return
    except json.JSONDecodeError as exc:
        print(f"[ERROR] 评测数据集 JSON 解析失败: {exc}")
        return

    all_results = []
    for index, case in enumerate(eval_cases, 1):
        print(f"\n[{index}/{len(eval_cases)}] 评测进度")
        eval_result = evaluate_single_case(case)
        all_results.append(
            {
                "query": case.get("query", ""),
                "faithfulness": eval_result["faithfulness"],
                "answer_relevance": eval_result["answer_relevance"],
                "overall_effectiveness": eval_result["overall_effectiveness"],
                "reason": eval_result["reason"],
            }
        )
        if index < len(eval_cases):
            print("[INFO] 等待 2 秒后继续下一条评测...")
            time.sleep(2)

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
    print("\n[INFO] 评测流水线执行完成")


if __name__ == "__main__":
    main()
