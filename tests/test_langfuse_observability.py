"""Verify the optional Langfuse adapter without a running backend."""

from dataclasses import dataclass, field

from myagent.llm.types import LLMResponse, Message, ToolCall
from myagent.observability import LangfuseReporter


@dataclass
class FakeSpan:
    trace_id: str = "trace-1"
    id: str = "span-1"
    updates: list[dict] = field(default_factory=list)
    trace_updates: list[dict] = field(default_factory=list)
    scores: list[dict] = field(default_factory=list)
    ended: bool = False

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def update_trace(self, **kwargs):
        self.trace_updates.append(kwargs)

    def score(self, **kwargs):
        self.scores.append(kwargs)

    def end(self):
        self.ended = True


class FakeLangfuse:
    def __init__(self) -> None:
        self.spans: list[tuple[dict, FakeSpan]] = []
        self.flushed = False

    def start_observation(self, **kwargs):
        span = FakeSpan(id=f"span-{len(self.spans) + 1}")
        self.spans.append((kwargs, span))
        return span

    def flush(self):
        self.flushed = True


class Task:
    task_id = "optional-transfer-1"
    family = "optional-normalization"
    stage = "transfer"
    prompt = "Repair display name normalization"


def test_reporter_records_redacted_generation_and_deterministic_scores() -> None:
    backend = FakeLangfuse()
    reporter = LangfuseReporter(backend, run_name="compact-memory-ab")
    task_span = reporter.start_task(Task(), arm="evolved", model="deepseek-chat")
    generation = reporter.start_generation(
        [Message(role="user", content="private source code")],
        [{"function": {"name": "file_read"}}],
        model="deepseek-chat",
    )
    reporter.finish_generation(
        generation,
        LLMResponse(
            content="done",
            tool_calls=[ToolCall(id="1", name="file_read", arguments={})],
            usage={"prompt_tokens": 100, "cached_tokens": 80, "completion_tokens": 5},
        ),
    )
    reporter.finish_task(
        task_span,
        {
            "passed": True,
            "error": "",
            "answer": "done",
            "wall_seconds": 1.25,
            "usage": {
                "prompt_tokens": 100,
                "cached_tokens": 80,
                "completion_tokens": 5,
                "tool_calls": 1,
            },
            "recalled_memory_ids": [7],
            "evolution_eligible": True,
        },
    )
    reporter.flush()

    task_args, recorded_task = backend.spans[0]
    generation_args, recorded_generation = backend.spans[1]
    assert task_args["input"]["chars"] == len(Task.prompt)
    assert Task.prompt not in str(task_args["input"])
    assert generation_args["trace_context"]["parent_span_id"] == recorded_task.id
    assert generation_args["input"]["messages"][0]["chars"] == 19
    assert "private source code" not in str(generation_args["input"])
    assert recorded_generation.ended
    scores = {item["name"]: item["value"] for item in recorded_task.scores}
    assert scores["task_passed"] == 1.0
    assert scores["uncached_tokens"] == 20.0
    assert scores["memory_recalled"] == 1.0
    assert scores["infrastructure_error"] == 0.0
    assert recorded_task.ended
    assert backend.flushed


def test_reporter_classifies_balance_failure_as_infrastructure_error() -> None:
    backend = FakeLangfuse()
    reporter = LangfuseReporter(backend, run_name="failed-run")
    task_span = reporter.start_task(Task(), arm="evolved", model="deepseek-chat")
    reporter.finish_task(
        task_span,
        {
            "passed": False,
            "error": "APIStatusError: Error code: 402 - insufficient_balance_error",
            "answer": "",
            "wall_seconds": 0.1,
            "usage": {
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "completion_tokens": 0,
                "tool_calls": 0,
            },
            "recalled_memory_ids": [],
            "evolution_eligible": False,
        },
    )

    scores = {item["name"]: item["value"] for item in task_span.scores}
    assert scores["task_passed"] == 0.0
    assert scores["infrastructure_error"] == 1.0
    assert task_span.updates[-1]["output"]["error_kind"] == "infrastructure_balance"


def test_reporter_can_attach_generation_to_current_experiment_context() -> None:
    backend = FakeLangfuse()
    reporter = LangfuseReporter(
        backend,
        run_name="long-reuse",
        inherit_current_context=True,
    )
    generation = reporter.start_generation(
        [Message(role="user", content="repair workflow")],
        [],
        model="deepseek-flash",
    )
    assert generation is not None
    args, _ = backend.spans[0]
    assert args["trace_context"] is None
