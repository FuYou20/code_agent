"""Public web gateway that gives each browser an isolated MyAgent TUI PTY."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = ROOT / "web_demo" / "static"
WORKSPACE_TEMPLATE = ROOT / "web_demo" / "demo_workspace"
MAX_SESSIONS = max(1, int(os.environ.get("DEMO_MAX_SESSIONS", "2")))
MAX_SESSIONS_PER_HOUR = max(
    1, int(os.environ.get("DEMO_MAX_SESSIONS_PER_HOUR", "3"))
)
SESSION_SECONDS = max(60, int(os.environ.get("DEMO_SESSION_SECONDS", "900")))
SESSION_INPUT_BYTES = max(
    4096, int(os.environ.get("DEMO_SESSION_INPUT_BYTES", "100000"))
)

app = FastAPI(
    title="MyAgent Public Demo",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.mount("/assets", StaticFiles(directory=STATIC_ROOT), name="assets")

_active_sessions = 0
_active_lock = asyncio.Lock()
_recent_sessions: dict[str, deque[float]] = defaultdict(deque)


@app.get("/")
async def landing() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/terminal")
async def terminal() -> FileResponse:
    return FileResponse(STATIC_ROOT / "terminal.html")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "mode": "public-readonly"})


def _client_ip(websocket: WebSocket) -> str:
    forwarded = websocket.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
    if forwarded:
        return forwarded
    return websocket.client.host if websocket.client else "unknown"


async def _reserve_session(ip: str) -> tuple[bool, str]:
    global _active_sessions
    async with _active_lock:
        now = time.time()
        history = _recent_sessions[ip]
        while history and now - history[0] >= 3600:
            history.popleft()
        if len(history) >= MAX_SESSIONS_PER_HOUR:
            return False, "该IP已达到每小时体验次数上限。"
        if _active_sessions >= MAX_SESSIONS:
            return False, "在线体验席位已满，请稍后重试。"
        history.append(now)
        _active_sessions += 1
        return True, ""


async def _release_session() -> None:
    global _active_sessions
    async with _active_lock:
        _active_sessions = max(0, _active_sessions - 1)


def _write_demo_config(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    provider = os.environ.get("DEMO_LLM_PROVIDER", "deepseek")
    model = os.environ.get("DEMO_LLM_MODEL", "deepseek-flash")
    key_env = os.environ.get("DEMO_LLM_API_KEY_ENV", "DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEMO_LLM_BASE_URL", "https://api.deepseek.com")
    content = (
        "llm:\n"
        f"  provider: {provider}\n"
        f"  model: {model}\n"
        f"  base_url: {base_url}\n"
        f"  api_key_env: {key_env}\n"
        "agent:\n"
        "  max_steps: 8\n"
        "skills:\n"
        "  strategy: scene_epoch\n"
        "  top_k: 5\n"
        "  candidate_k: 20\n"
        "  activation_min_score: 80\n"
        "  max_frozen: 8\n"
        "  warm_turns: 3\n"
        "  idle_seconds: 21600\n"
        "evolution:\n"
        "  gc_on_startup: false\n"
        "  gc_interval_seconds: 86400\n"
        "tools:\n"
        "  terminal_whitelist: []\n"
    )
    (home / "config.yaml").write_text(content, encoding="utf-8")


def _prepare_workspace(session_root: Path) -> Path:
    workspace = session_root / "workspace"
    shutil.copytree(WORKSPACE_TEMPLATE, workspace)
    commands = (
        ["git", "init", "-q"],
        ["git", "config", "user.name", "MyAgent Demo"],
        ["git", "config", "user.email", "demo@example.invalid"],
        ["git", "add", "."],
        ["git", "commit", "-qm", "demo workspace"],
    )
    for command in commands:
        subprocess.run(
            command,
            cwd=workspace,
            capture_output=True,
            timeout=10,
            check=False,
        )
    return workspace


def _resize_pty(fd: int, rows: int, cols: int) -> None:
    import fcntl
    import termios

    safe_rows = min(max(rows, 10), 100)
    safe_cols = min(max(cols, 40), 240)
    size = struct.pack("HHHH", safe_rows, safe_cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, size)


async def _bridge_terminal(websocket: WebSocket) -> None:
    if os.name != "posix":
        await websocket.send_json(
            {"type": "error", "message": "Web Terminal仅支持Linux部署环境。"}
        )
        return

    import pty

    with tempfile.TemporaryDirectory(prefix="myagent-public-") as temp:
        session_root = Path(temp)
        workspace = await asyncio.to_thread(_prepare_workspace, session_root)
        home = session_root / "home"
        _write_demo_config(home)
        master_fd, slave_fd = pty.openpty()
        _resize_pty(master_fd, 32, 120)
        env = os.environ.copy()
        env.update(
            {
                "MYAGENT_HOME": str(home),
                "MYAGENT_PUBLIC_DEMO": "readonly",
                "MYAGENT_DEMO_TOKEN_LIMIT": os.environ.get(
                    "DEMO_TOKEN_LIMIT", "30000"
                ),
                "TERM": "xterm-256color",
                "PYTHONUNBUFFERED": "1",
            }
        )
        process = subprocess.Popen(
            [sys.executable, "-m", "myagent.tui"],
            cwd=workspace,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave_fd)

        async def output_loop() -> None:
            while process.poll() is None:
                try:
                    data = await asyncio.to_thread(os.read, master_fd, 8192)
                except OSError:
                    return
                if not data:
                    return
                await websocket.send_bytes(data)

        async def input_loop() -> None:
            consumed = 0
            while process.poll() is None:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    return
                raw = message.get("text")
                if not isinstance(raw, str):
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "resize":
                    _resize_pty(
                        master_fd,
                        int(payload.get("rows", 32)),
                        int(payload.get("cols", 120)),
                    )
                    continue
                data = str(payload.get("data", ""))
                encoded = data.encode("utf-8")
                consumed += len(encoded)
                if len(encoded) > 4096 or consumed > SESSION_INPUT_BYTES:
                    await websocket.send_json(
                        {"type": "error", "message": "会话输入额度已用完。"}
                    )
                    return
                await asyncio.to_thread(os.write, master_fd, encoded)

        output_task = asyncio.create_task(output_loop())
        input_task = asyncio.create_task(input_loop())
        try:
            done, pending = await asyncio.wait(
                {output_task, input_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*done, return_exceptions=True)
        finally:
            try:
                os.close(master_fd)
            except OSError:
                pass
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(process.wait), timeout=3
                    )
                except TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
            await asyncio.gather(output_task, input_task, return_exceptions=True)


@app.websocket("/ws/terminal")
async def terminal_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    allowed, reason = await _reserve_session(_client_ip(websocket))
    if not allowed:
        await websocket.send_json({"type": "error", "message": reason})
        await websocket.close(code=1013)
        return
    try:
        await asyncio.wait_for(_bridge_terminal(websocket), timeout=SESSION_SECONDS)
    except TimeoutError:
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {"type": "error", "message": "在线体验已到达时间上限。"}
            )
    except WebSocketDisconnect:
        pass
    finally:
        await _release_session()
        with contextlib.suppress(Exception):
            await websocket.close()


# Imported late only to keep the process/session code above visually linear.
import contextlib  # noqa: E402
