"""Verify the shared gate used by experience memory and Skill learning."""

from myagent.evolution import ToolObservation, evaluate_turn
from myagent.evolution.engine import build_memory_card
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


def test_final_passing_verification_allows_red_green_trajectory() -> None:
    decision = evaluate_turn(
        "Reproduced the failure, repaired it, and verified the suite.",
        [
            _observation("test_run", {}, {"returncode": 1}),
            _observation(
                "file_write",
                {"path": "src/app.py", "content": "fixed = True"},
                {"path": "src/app.py"},
            ),
            _observation("test_run", {}, {"returncode": 0}),
        ],
    )
    assert decision.eligible


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


def test_recovered_read_only_tool_error_does_not_block_verified_change() -> None:
    decision = evaluate_turn(
        "Used bounded reads after shell exploration was denied, then verified.",
        [
            _observation(
                "terminal_exec",
                {"command": "find ."},
                {"error": "PermissionError"},
            ),
            _observation(
                "file_write",
                {"path": "src/app.py", "content": "fixed = True"},
                {"path": "src/app.py"},
            ),
            _observation("test_run", {}, {"returncode": 0}),
        ],
    )
    assert decision.eligible


def test_memory_card_is_structured_and_bounded() -> None:
    observations = [
        _observation(
            "file_write",
            {"path": "src/retry.py", "content": "value = 1"},
            {"path": "src/retry.py"},
        ),
        _observation("test_run", {}, {"returncode": 0}),
    ]
    decision = evaluate_turn("Fixed retry boundaries. " * 80, observations)

    card = build_memory_card(
        "Repair retry accounting. " * 80,
        "Fixed retry boundaries. " * 80,
        observations,
    )

    assert card.startswith("[经验卡]\ntask: Repair retry accounting.")
    assert "task: Repair retry accounting." in card
    assert "lesson: Fixed retry boundaries." in card
    assert "evidence: test_run=passed" in card
    assert "scope:" not in card
    assert "files:" not in card
    assert len(card) <= 440
