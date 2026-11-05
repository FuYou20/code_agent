"""Small optional Langfuse adapter for the stateful evolution benchmark."""

from __future__ import annotations

from hashlib import sha256
import os
from typing import Any

from ..llm.types import LLMResponse, Message


def _redacted_text(value: str) -> dict[str, object]:
    return {
        "sha256": sha256(value.encode("utf-8")).hexdigest(),
        "chars": len(value),
    }


def _error_kind(error: str) -> str:
    lowered = error.lower()
    if not error:
        return "none"
    if any(marker in lowered for marker in ("insufficient_balance", "error code: 402")):
        return "infrastructure_balance"
    if any(marker in lowered for marker in ("timeout", "connection", "rate limit")):
        return "infrastructure_transport"
    return "agent_error"


class LangfuseReporter:
    """Record task, generation, and deterministic score data in Langfuse.

    The adapter is deliberately absent from the normal runtime unless the
    benchmark is launched with ``--langfuse``. Content capture is opt-in;
    hashes and sizes are sufficient for the default cost/debug experiment.
    """

    def __init__(
        self,
        client: Any,
        *,
        run_name: str,
        capture_content: bool = False,
        inherit_current_context: bool = False,
    ) -> None:
        self.client = client
        self.run_name = run_name
        self.capture_content = capture_content
        self.inherit_current_context = inherit_current_context
        self._current_task: Any | None = None

    @classmethod
    def from_env(cls, *, enabled: bool, run_name: str) -> "LangfuseReporter | None":
        if not enabled:
            return None
        required = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                "Langfuse enabled but environment variables are missing: "
                + ", ".join(missing)
            )
        try:
            from langfuse import Langfuse
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Langfuse SDK is not installed; run pip install -e .[observability]"
            ) from exc

        client = Langfuse()
        if not client.auth_check():
            raise RuntimeError("Langfuse authentication or base URL check failed")
        capture = os.environ.get("MYAGENT_LANGFUSE_CAPTURE_CONTENT", "false")
        return cls(
            client,
            run_name=run_name,
            capture_content=capture.lower() == "true",
        )

    def start_task(self, task: Any, *, arm: str, model: str) -> Any:
        task_input: object = (
            task.prompt if self.capture_content else _redacted_text(task.prompt)
        )
        span = self.client.start_observation(
            name=f"evolution-task:{task.task_id}",
            as_type="agent",
            input=task_input,
            metadata={
                "arm": arm,
                "family": task.family,
                "stage": task.stage,
                "model": model,
            },
        )
        span.update_trace(
            name=self.run_name,
            session_id=f"{self.run_name}:{arm}:{task.family}",
            tags=["myagent", "evolution-eval", arm, task.stage],
            metadata={"task_id": task.task_id, "family": task.family},
        )
        self._current_task = span
        return span

    def start_generation(
        self,
        messages: list[Message],
        tools: list[dict],
        *,
        model: str,
    ) -> Any | None:
        if self._current_task is None and not self.inherit_current_context:
            return None
        if self.capture_content:
            generation_input: object = [
                {"role": message.role, "content": message.content}
                for message in messages
            ]
        else:
            generation_input = {
                "messages": [
                    {"role": message.role, "chars": len(message.content)}
                    for message in messages
                ],
                "tool_names": [
                    str(item.get("function", {}).get("name", "")) for item in tools
                ],
            }
        trace_context = (
            {
                "trace_id": self._current_task.trace_id,
                "parent_span_id": self._current_task.id,
            }
            if self._current_task is not None
            else None
        )
        return self.client.start_observation(
            trace_context=trace_context,
            name="llm.chat",
            as_type="generation",
            input=generation_input,
            model=model,
        )

    def finish_generation(
        self,
        generation: Any | None,
        response: LLMResponse | None,
        *,
        error: Exception | None = None,
    ) -> None:
        if generation is None:
            return
        if error is not None:
            generation.update(
                level="ERROR",
                status_message=f"{type(error).__name__}: {error}",
            )
            generation.end()
            return
        assert response is not None
        output: object = (
            response.content
            if self.capture_content
            else {
                "content_chars": len(response.content or ""),
                "tool_calls": [call.name for call in response.tool_calls],
            }
        )
        generation.update(
            output=output,
            usage_details={key: int(value) for key, value in response.usage.items()},
        )
        generation.end()

    def finish_task(self, task_span: Any, row: dict[str, Any]) -> None:
        usage = row["usage"]
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        cached_tokens = int(usage.get("cached_tokens", 0))
        uncached_tokens = prompt_tokens - cached_tokens
        error_kind = _error_kind(str(row.get("error", "")))
        task_span.update(
            output={
                "passed": bool(row["passed"]),
                "error_kind": error_kind,
                "answer_chars": len(str(row.get("answer", ""))),
            },
            metadata={
                "usage": usage,
                "wall_seconds": float(row["wall_seconds"]),
                "memory_recalled": bool(row.get("recalled_memory_ids")),
                "evolution_eligible": bool(row.get("evolution_eligible")),
            },
            level="ERROR" if error_kind != "none" else "DEFAULT",
            status_message=str(row.get("error", "")) or None,
        )
        scores: tuple[tuple[str, float, str], ...] = (
            ("task_passed", float(bool(row["passed"])), "BOOLEAN"),
            ("prompt_tokens", float(prompt_tokens), "NUMERIC"),
            ("cached_tokens", float(cached_tokens), "NUMERIC"),
            ("uncached_tokens", float(uncached_tokens), "NUMERIC"),
            ("tool_calls", float(usage.get("tool_calls", 0)), "NUMERIC"),
            ("wall_seconds", float(row["wall_seconds"]), "NUMERIC"),
            (
                "memory_recalled",
                float(bool(row.get("recalled_memory_ids"))),
                "BOOLEAN",
            ),
            (
                "infrastructure_error",
                float(error_kind.startswith("infrastructure_")),
                "BOOLEAN",
            ),
        )
        for name, value, data_type in scores:
            task_span.score(name=name, value=value, data_type=data_type)
        task_span.end()
        self._current_task = None

    def flush(self) -> None:
        self.client.flush()
