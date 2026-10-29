"""Verify the shared gate used by experience memory and Skill learning."""

from myagent.evolution import ToolObservation, evaluate_turn
from myagent.llm.types import ToolCall


def _observation(name: str, arguments: dict, result: dict) -> ToolObservation:
    return ToolObservation(ToolCall(id=name, name=name, arguments=arguments), result)


def test_verified_write_is_eligible() -> None:
    decision = evaluate_turn(
        "Updated and verified.",
        [
            _observation(
                "file_write",
                {"path": "src/app.py", "content": "value = 1"},
                {"path": "src/app.py"},
            ),
            _observation("test_run", {}, {"returncode": 0}),
        ],
    )
    assert decision.eligible
    assert decision.changed_files == ("src/app.py",)


def test_failed_verification_blocks_all_evolution_outputs() -> None:
    decision = evaluate_turn(
        "Updated, but tests failed.",
        [
            _observation(
                "file_write",
                {"path": "src/app.py", "content": "broken"},
                {"path": "src/app.py"},
            ),
            _observation("test_run", {}, {"returncode": 1}),
        ],
    )
    assert not decision.eligible
    assert "verification_failed:test_run" in decision.reasons


def test_chatter_and_tool_errors_are_not_evolution_candidates() -> None:
    chatter = evaluate_turn("hello", [])
    failed = evaluate_turn(
        "Could not update.",
        [
            _observation(
                "file_write",
                {"path": "src/app.py", "content": "x"},
                {"error": "OSError"},
            )
        ],
    )
    assert not chatter.eligible
    assert chatter.reasons == ("no_durable_change",)
    assert not failed.eligible
    assert "tool_error:file_write" in failed.reasons
