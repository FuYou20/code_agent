"""Compare full L0 loading with retrieval plus main-model reranking."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myagent.config import load_config
from myagent.llm.client import OpenAIClient
from myagent.llm.types import Message
from myagent.skills.retriever import retrieve_relevant
from scripts.evaluate_skill_loading import build_retrieval_cases, build_skill_index


def _json_object(content: str) -> dict:
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def _render_candidates(candidates: list[dict]) -> str:
    return "\n".join(
        json.dumps(
            {
                "name": item["name"],
                "description": item["description"],
                "triggers": item["triggers"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for item in candidates
    )


def expand_query(client, query: str) -> tuple[str, int]:
    """Use the main model to produce search terms without seeing the skill catalog."""
    response = client.chat(
        [
            Message(
                role="system",
                content=(
                    "把用户请求改写成用于技能检索的短语。输出严格 JSON："
                    '{"terms":["term 1","term 2"]}。最多 8 个中英文技术短语；'
                    "不得解释，不得猜测技能名称。"
                ),
            ),
            Message(role="user", content=query),
        ],
        tools=[],
    )
    payload = _json_object(response.content or "")
    terms = payload.get("terms", [])
    if not isinstance(terms, list):
        raise ValueError("query expansion terms must be a list")
    expanded = " ".join([query, *(str(term) for term in terms[:8])])
    return expanded, int(response.usage.get("prompt_tokens", 0))


def rerank(client, query: str, candidates: list[dict]) -> tuple[list[str], int]:
    """Ask the same main model to select zero to two skills from supplied L0 rows."""
    if not candidates:
        return [], 0
    allowed = {str(item["name"]) for item in candidates}
    response = client.chat(
        [
            Message(
                role="system",
                content=(
                    "根据用户真实意图，从候选技能中选择完成任务必需的技能。"
                    "不要仅凭关键词，要区分讨论、写作和实际执行。"
                    "闲聊或都不适用时返回空数组。输出严格 JSON："
                    '{"selected":["skill-name"]}。最多选择 2 个，只能使用候选 name。'
                ),
            ),
            Message(
                role="user",
                content=f"用户请求：\n{query}\n\n候选技能：\n{_render_candidates(candidates)}",
            ),
        ],
        tools=[],
    )
    payload = _json_object(response.content or "")
    raw_selected = payload.get("selected", [])
    if not isinstance(raw_selected, list):
        raise ValueError("selected must be a list")
    selected: list[str] = []
    for value in raw_selected[:2]:
        name = str(value)
        if name in allowed and name not in selected:
            selected.append(name)
    return selected, int(response.usage.get("prompt_tokens", 0))


def _hit(expected: str | None, selected: list[str]) -> bool:
    return selected == [] if expected is None else expected in selected


def evaluate(client, *, candidate_k: int = 20) -> dict:
    """Run both strategies and return only token and required-skill hit metrics."""
    index = build_skill_index()
    cases = [case for case in build_retrieval_cases() if case.expected is not None]
    full_tokens = 0
    rag_tokens = 0
    full_hits = 0
    rag_hits = 0
    misses: list[dict] = []

    for case in cases:
        full_selected, used = rerank(client, case.query, index)
        full_tokens += used
        full_ok = _hit(case.expected, full_selected)
        full_hits += int(full_ok)

        expanded, used = expand_query(client, case.query)
        rag_tokens += used
        matches = retrieve_relevant(expanded, index, k=candidate_k)
        by_name = {str(item["name"]): item for item in index}
        candidates = [by_name[match.name] for match in matches]
        rag_selected, used = rerank(client, case.query, candidates)
        rag_tokens += used
        rag_ok = _hit(case.expected, rag_selected)
        rag_hits += int(rag_ok)

        if not full_ok or not rag_ok:
            misses.append(
                {
                    "query": case.query,
                    "required": case.expected,
                    "full_selected": full_selected,
                    "rag_candidates": [item["name"] for item in candidates],
                    "rag_selected": rag_selected,
                    "full_hit": full_ok,
                    "rag_hit": rag_ok,
                }
            )

    reduction = 1 - rag_tokens / full_tokens if full_tokens else 0.0
    return {
        "required_cases": len(cases),
        "candidate_k": candidate_k,
        "full_input_tokens": full_tokens,
        "rag_input_tokens": rag_tokens,
        "token_reduction": reduction,
        "full_skill_hits": full_hits,
        "rag_skill_hits": rag_hits,
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.candidate_k <= 0:
        raise SystemExit("--candidate-k must be greater than zero")

    config = load_config()
    key_name = str(config["llm"]["api_key_env"])
    api_key = os.environ.get(key_name)
    if not api_key:
        raise SystemExit(f"缺少环境变量：{key_name}")
    base_url = config["llm"].get("base_url")
    if config["llm"].get("provider") == "deepseek" and not base_url:
        base_url = "https://api.deepseek.com"
    client = OpenAIClient(
        model=str(config["llm"]["model"]),
        api_key=api_key,
        base_url=base_url,
    )
    result = evaluate(client, candidate_k=args.candidate_k)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"required_cases={result['required_cases']} candidate_k={result['candidate_k']}")
    print(f"full_input_tokens={result['full_input_tokens']}")
    print(f"rag_input_tokens={result['rag_input_tokens']}")
    print(f"token_reduction={result['token_reduction']:.2%}")
    print(f"full_skill_hits={result['full_skill_hits']}/{result['required_cases']}")
    print(f"rag_skill_hits={result['rag_skill_hits']}/{result['required_cases']}")
    if result["misses"]:
        print("misses:")
        for miss in result["misses"]:
            print(json.dumps(miss, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
