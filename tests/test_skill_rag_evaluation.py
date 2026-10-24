"""Verify the two-arm token-and-hit Skill RAG evaluator."""

import json

from myagent.llm.types import LLMResponse
from scripts.evaluate_skill_rag import _json_object, evaluate, rerank


class PerfectFakeClient:
    """Resolve fixture intent while returning deterministic synthetic usage."""

    def chat(self, messages, tools) -> LLMResponse:
        system = messages[0].content
        user = messages[-1].content
        if "技能检索的短语" in system:
            mappings = {
                "depend on one another": ["circular import"],
                "assertion is red": ["pytest failure"],
                "leadership overview": ["executive summary"],
                "expected contract": ["schema validation"],
                "previously deployed": ["deployment rollback"],
                "attacker goals": ["threat model"],
                "prior studies": ["literature search"],
                "scanned report": ["pdf extract"],
            }
            terms = next((value for key, value in mappings.items() if key in user), [])
            return LLMResponse(
                content=json.dumps({"terms": terms}),
                usage={"prompt_tokens": 20},
            )

        candidate_lines = user.split("候选技能：\n", 1)[1].splitlines()
        names = [json.loads(line)["name"] for line in candidate_lines]
        expected = None
        intent_rules = [
            ("tutorial about circular", "writing-technical-tutorial"),
            ("circular import", "code-circular-import"),
            ("pytest", "code-pytest-failure"),
            ("first assertion is red", "code-pytest-failure"),
            ("leadership overview", "writing-executive-summary"),
            ("release notes", "writing-release-notes"),
            ("expected contract", "data-schema-validation"),
            ("data schema", "data-schema-validation"),
            ("previously deployed", "devops-deployment-rollback"),
            ("attacker goals", "security-threat-model"),
            ("prior studies", "research-literature-search"),
            ("scanned report", "documents-pdf-extract"),
            ("depend on one another", "code-circular-import"),
        ]
        request = user.split("用户请求：\n", 1)[1].split("\n\n候选技能：", 1)[0]
        lower = request.casefold()
        for phrase, name in intent_rules:
            if phrase in lower:
                expected = name
                break
        selected = [expected] if expected in names else []
        return LLMResponse(
            content=json.dumps({"selected": selected}),
            usage={"prompt_tokens": len(names) * 10},
        )


def test_json_object_accepts_fenced_json() -> None:
    assert _json_object('```json\n{"selected": []}\n```') == {"selected": []}


def test_rerank_rejects_names_outside_candidates() -> None:
    class InvalidClient:
        def chat(self, messages, tools):
            return LLMResponse(
                content='{"selected":["not-allowed"]}',
                usage={"prompt_tokens": 10},
            )

    selected, tokens = rerank(
        InvalidClient(),
        "query",
        [{"name": "allowed", "description": "d", "triggers": []}],
    )
    assert selected == []
    assert tokens == 10


def test_two_arm_evaluation_reports_only_tokens_hits_and_misses() -> None:
    result = evaluate(PerfectFakeClient(), candidate_k=20)
    assert result["required_cases"] == 14
    assert result["full_input_tokens"] > result["rag_input_tokens"]
    assert result["token_reduction"] > 0
    assert result["full_skill_hits"] == result["required_cases"]
    assert result["rag_skill_hits"] == result["required_cases"]
    assert result["misses"] == []
