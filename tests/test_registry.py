"""Verify deterministic schemas and exception-safe tool execution."""

from myagent.tools import registry


def setup_function() -> None:
    registry.TOOLS.clear()


def teardown_function() -> None:
    registry.TOOLS.clear()


def _schema(description: str = "") -> dict:
    return {"description": description, "type": "object", "properties": {}}


def test_schemas_are_sorted_by_name() -> None:
    registry.register("zeta", _schema(), lambda: {"ok": True})
    registry.register("alpha", _schema(), lambda: {"ok": True})
    assert [item["function"]["name"] for item in registry.schemas()] == ["alpha", "zeta"]
    assert [item["function"]["name"] for item in registry.schemas({"zeta"})] == ["zeta"]


def test_execute_catches_tool_exceptions() -> None:
    def fail() -> dict:
        raise RuntimeError("boom")

    registry.register("fail", _schema(), fail)
    assert registry.execute("fail", {}) == {"error": "RuntimeError", "message": "boom"}

