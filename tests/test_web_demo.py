from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web_demo" / "static"


def test_public_demo_assets_are_complete() -> None:
    expected = {
        "index.html",
        "styles.css",
        "landing.js",
        "config.js",
        "terminal.html",
        "terminal.css",
        "terminal.js",
    }
    assert expected <= {item.name for item in STATIC.iterdir()}
    landing = (STATIC / "index.html").read_text(encoding="utf-8")
    terminal = (STATIC / "terminal.js").read_text(encoding="utf-8")
    assert "进入在线 TUI" in landing
    assert "48.83" in landing
    assert "/ws/terminal" in terminal


def test_public_demo_http_routes() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from myagent.web_demo import app

    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "mode": "public-readonly",
        }
        assert client.get("/").status_code == 200
        assert "MyAgent" in client.get("/").text
        assert client.get("/terminal").status_code == 200
