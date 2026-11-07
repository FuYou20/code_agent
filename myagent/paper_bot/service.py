"""Long-running local Feishu bot for paper recommendation and focused reading."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import os
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

from ..config import load_config
from ..gateway import AgentGateway, AgentTask
from ..home import get_home
from ..llm.client import OpenAIClient
from .runtime import PaperAgentRuntime
from .store import PaperBotStore, StoredPaper


SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class BotCommand:
    name: str
    label: str = ""
    focus: str = "实验"


def parse_command(text: str) -> BotCommand:
    value = re.sub(r"\s+", " ", text).strip()
    compact = value.replace(" ", "")
    if compact in {"绑定", "开始绑定", "/bind"}:
        return BotCommand("bind")
    if compact in {"今日推荐", "推荐论文", "论文推荐", "/papers"}:
        return BotCommand("digest")
    if compact in {"帮助", "使用说明", "/help"}:
        return BotCommand("help")

    label_match = re.search(r"\b(P[12])\b", value, re.I)
    label = label_match.group(1).upper() if label_match else ""
    if any(word in value for word in ("局限", "不足", "未来工作")):
        focus = "局限"
    elif any(word in value for word in ("方法", "模型结构", "算法")):
        focus = "方法"
    else:
        focus = "实验"
    if label or any(
        word in value
        for word in (
            "精读",
            "仔细看",
            "详细看",
            "详情",
            "详细信息",
            "深入",
            "实验部分",
            "消融",
        )
    ):
        return BotCommand("read", label=label, focus=focus)
    return BotCommand("chat")


def _persistent_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value or os.name != "nt":
        return value
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            stored, _ = winreg.QueryValueEx(key, name)
        return str(stored).strip()
    except (OSError, ImportError):
        return ""


def _require_credential(name: str) -> str:
    value = _persistent_environment(name)
    if not value:
        raise SystemExit(f"缺少环境变量：{name}")
    return value


def format_feishu_digest(items: list[StoredPaper], digest_date: date) -> str:
    labels = {"pathology": "病理图像", "continual": "持续学习"}
    parts = [f"📚 **{digest_date.isoformat()} 每日论文推荐**"]
    for item in items:
        topic = labels.get(item.topic, item.topic)
        if item.is_intersection:
            topic += " × 交叉方向"
        parts.extend(
            (
                f"\n**{item.label}｜{topic}**",
                f"**{item.title}**",
                f"发表时间：{item.venue}；arXiv：{item.arxiv_month}",
                f"主要问题：{item.problem}",
                f"[PaperNotes 原文解读]({item.url})",
            )
        )
    parts.append(
        "\n回复 `精读 P1 实验`、`精读 P2 方法` 或 `总结 P1 局限`，"
        "我会继续读取公开 PDF。"
    )
    return "\n".join(parts)


HELP_TEXT = """**Lulu 使用方法**

你也可以直接和我正常聊天，不需要使用命令。

- `绑定`：将当前飞书会话设为每日推送目标
- `今日推荐`：立即生成或重发今天的两篇论文
- `精读 P1 实验`：读取 P1 的实验、结果与消融
- `精读 P2 方法`：读取 P2 的方法部分
- `总结 P1 局限`：读取作者讨论与局限

PDF 不公开时会明确告知，不会把摘要冒充全文。"""


class PaperBotService:
    def __init__(
        self,
        *,
        channel,
        store: PaperBotStore,
        history_path: Path,
        cache_dir: Path,
        push_hour: int = 8,
    ):
        self.channel = channel
        self.store = store
        self.history_path = history_path
        self.cache_dir = cache_dir
        self.push_hour = push_hour
        self.agent = PaperAgentRuntime(
            store=store,
            history_path=history_path,
            cache_dir=cache_dir,
            workspace=cache_dir.parent,
        )
        self.gateway = AgentGateway(
            self._execute_agent_task,
            claim_task=self.store.reserve_message,
        )
        channel.on("message", self.handle_message)

    async def _send(self, chat_id: str, text: str, *, reply_to: str = "") -> None:
        opts = {"reply_to": reply_to} if reply_to else None
        result = await self.channel.send(chat_id, {"markdown": text}, opts)
        if not result.success:
            raise RuntimeError(f"飞书发送失败：{result.error}")

    async def publish_today(self, chat_id: str) -> str:
        today = datetime.now(SHANGHAI).date()
        task = AgentTask(
            task_id=f"scheduler:paper-digest:{chat_id}:{today.isoformat()}",
            session_id=f"feishu:{chat_id}",
            source="scheduler",
            user_id="system",
            content=(
                f"[系统定时事件｜{today.isoformat()}] 现在执行每日论文推荐。"
                "必须调用 paper_recommend_today 获取真实结果，然后整理成适合飞书阅读的推荐消息。"
            ),
            reply_target=chat_id,
            metadata={"trigger": "daily-paper-digest"},
        )
        result = await self.gateway.submit(task)
        if result.duplicate:
            return ""
        if not result.success:
            raise RuntimeError(f"{result.error_type}: {result.error_message}")
        await self._send(chat_id, result.content)
        return result.content

    def _make_llm_client(self) -> OpenAIClient:
        config = load_config()
        llm = config["llm"]
        key_name = str(llm["api_key_env"])
        api_key = _persistent_environment(key_name)
        if not api_key:
            raise RuntimeError(f"调用模型需要环境变量 {key_name}")
        provider = str(llm.get("provider", "openai"))
        base_url = llm.get("base_url")
        if provider == "deepseek" and not base_url:
            base_url = "https://api.deepseek.com"
        return OpenAIClient(
            model=str(llm["model"]),
            api_key=api_key,
            base_url=str(base_url) if base_url else None,
        )

    def _execute_agent_task(self, task: AgentTask) -> str:
        client = self._make_llm_client()
        return self.agent.run(task.reply_target, task.content, client)

    async def _chat(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        *,
        sender_id: str = "unknown",
    ) -> None:
        task = AgentTask(
            task_id=message_id,
            session_id=f"feishu:{chat_id}",
            source="feishu",
            user_id=sender_id,
            content=text,
            reply_target=chat_id,
            metadata={"message_id": message_id},
        )
        result = await self.gateway.submit(task)
        if result.duplicate:
            return
        if result.success:
            self.store.append_chat_message(chat_id, "user", text)
            self.store.append_chat_message(chat_id, "assistant", result.content)
            await self._send(chat_id, result.content, reply_to=message_id)
        else:
            print(
                f"[paper-bot] Agent 回复失败 chat={chat_id[:8]}… "
                f"{result.error_type}: {result.error_message}"
            )
            await self._send(
                chat_id,
                f"这次回复失败了：{result.error_type}: {result.error_message}",
                reply_to=message_id,
            )

    async def handle_message(self, message) -> None:
        if getattr(message.sender, "is_bot", False):
            return
        chat_id = str(message.conversation.chat_id)
        sender_id = str(message.sender.open_id)
        text = (message.safe_content_text or message.content_text or "").strip()
        command = parse_command(text)

        if command.name == "bind":
            if not self.store.reserve_message(str(message.id)):
                return
            ok, detail = self.store.bind(chat_id, sender_id)
            suffix = "\n\n发送 `今日推荐` 可以立即测试。" if ok else ""
            await self._send(chat_id, detail + suffix, reply_to=str(message.id))
            return
        if not self.store.is_bound(chat_id):
            if not self.store.reserve_message(str(message.id)):
                return
            await self._send(
                chat_id,
                "请先发送 `绑定`，确认这是你的私人论文推送会话。",
                reply_to=str(message.id),
            )
            return
        if not self.store.is_owner(sender_id):
            return
        if command.name == "help":
            if not self.store.reserve_message(str(message.id)):
                return
            await self._send(chat_id, HELP_TEXT, reply_to=str(message.id))
            return
        await self._chat(
            chat_id,
            str(message.id),
            text,
            sender_id=sender_id,
        )

    async def publish_bound_chats(self) -> None:
        for chat_id in self.store.enabled_chats():
            try:
                await self.publish_today(chat_id)
            except Exception as exc:
                print(f"[paper-bot] 每日推送失败 chat={chat_id[:8]}…: {exc}")

    async def daily_loop(self) -> None:
        while True:
            now = datetime.now(SHANGHAI)
            target = now.replace(
                hour=self.push_hour, minute=0, second=0, microsecond=0
            )
            if target <= now:
                target += timedelta(days=1)
            wait_seconds = max(1.0, (target - now).total_seconds())
            print(f"[paper-bot] 下次推送：{target.isoformat()}")
            await asyncio.sleep(wait_seconds)
            await self.publish_bound_chats()


def _build_channel(app_id: str, app_secret: str):
    try:
        from lark_channel import FeishuChannel, SecurityConfig
    except ImportError as exc:
        raise SystemExit(
            '缺少飞书依赖，请执行：python -m pip install -e ".[paperbot]"'
        ) from exc
    return FeishuChannel(
        app_id=app_id,
        app_secret=app_secret,
        security=SecurityConfig(mode="strict"),
    )


async def _run(args: argparse.Namespace, channel) -> None:
    home = get_home() / "paper_bot"
    store = PaperBotStore(home / "paper_bot.sqlite3")
    service = PaperBotService(
        channel=channel,
        store=store,
        history_path=home / "recommendation_history.json",
        cache_dir=home / "pdf_cache",
        push_hour=args.hour,
    )
    try:
        await channel.start_background(timeout=20)
        snapshot = channel.connection_snapshot()
        if not snapshot.ready:
            raise RuntimeError("飞书长连接尚未就绪")
        print("[paper-bot] 飞书长连接成功，凭证有效。")
        if args.check:
            return
        if args.push_now:
            if not store.enabled_chats():
                print("[paper-bot] 尚未绑定会话，请先在飞书中发送：绑定")
            else:
                await service.publish_bound_chats()
        scheduler = asyncio.create_task(service.daily_loop())
        print("[paper-bot] 已启动。请在飞书中发送：绑定")
        try:
            await asyncio.Event().wait()
        finally:
            scheduler.cancel()
    finally:
        await channel.disconnect()
        # Give the SDK's private loop one turn to finish cancelled ping/sweep
        # tasks before interpreter shutdown (notably needed on Python 3.13).
        await asyncio.sleep(0.25)
        store.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地飞书论文助理")
    parser.add_argument("--check", action="store_true", help="只验证飞书连接后退出")
    parser.add_argument(
        "--push-now", action="store_true", help="启动后立即向已绑定会话推送"
    )
    parser.add_argument("--hour", type=int, default=8, help="北京时间每日推送小时")
    args = parser.parse_args()
    if not 0 <= args.hour <= 23:
        parser.error("--hour 必须在 0 到 23 之间")
    return args


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parse_args()
    # lark-channel-sdk creates its transport loop during import.  Import and
    # construct it before asyncio.run(), otherwise Python 3.13 may hand the SDK
    # the already-running application loop and its worker thread cannot own it.
    app_id = _require_credential("FEISHU_APP_ID")
    app_secret = _require_credential("FEISHU_APP_SECRET")
    channel = _build_channel(app_id, app_secret)
    asyncio.run(_run(args, channel))


if __name__ == "__main__":
    main()
