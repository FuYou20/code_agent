"""Paper capabilities exposed to the generic Agent loop as bounded tools."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

from ..paper_digest import recommend_daily, record_recommendations
from ..tools.registry import register
from .reader import download_pdf, extract_focus_excerpt
from .store import PaperBotStore, StoredPaper


SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class PaperToolContext:
    store: PaperBotStore
    chat_id: str
    history_path: Path
    cache_dir: Path


_CONTEXT: ContextVar[PaperToolContext | None] = ContextVar(
    "paper_tool_context",
    default=None,
)


def _context() -> PaperToolContext:
    value = _CONTEXT.get()
    if value is None:
        raise RuntimeError("paper tool called outside an active Feishu Agent turn")
    return value


@contextmanager
def bind_paper_tool_context(context: PaperToolContext) -> Iterator[None]:
    token = _CONTEXT.set(context)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def _paper_payload(paper: StoredPaper) -> dict:
    return {
        "label": paper.label,
        "topic": paper.topic,
        "title": paper.title,
        "venue": paper.venue,
        "arxiv_month": paper.arxiv_month,
        "arxiv_id": paper.arxiv_id,
        "problem": paper.problem,
        "url": paper.url,
        "is_intersection": paper.is_intersection,
    }


def _recommend_today() -> dict:
    context = _context()
    today = datetime.now(SHANGHAI).date()
    papers = context.store.digest_for_day(context.chat_id, today)
    reused = bool(papers)
    if not papers:
        recommendations = recommend_daily(context.history_path)
        if not recommendations:
            return {"date": today.isoformat(), "papers": [], "message": "没有找到候选论文"}
        context.store.save_digest(context.chat_id, today, recommendations)
        record_recommendations(
            context.history_path,
            recommendations,
            recommended_on=today,
        )
        papers = context.store.digest_for_day(context.chat_id, today)
    if papers:
        context.store.set_active_label(context.chat_id, papers[0].label)
    return {
        "date": today.isoformat(),
        "reused": reused,
        "papers": [_paper_payload(paper) for paper in papers],
    }


def _read_paper(label: str = "", focus: str = "实验") -> dict:
    context = _context()
    normalized_label = label.strip().upper() or context.store.active_label(context.chat_id)
    paper = context.store.latest_paper(context.chat_id, normalized_label)
    if paper is None:
        return {
            "error": "paper_not_found",
            "message": "没有可读取的推荐论文，请先调用 paper_recommend_today",
        }
    normalized_focus = focus if focus in {"实验", "方法", "局限"} else "实验"
    context.store.set_active_label(context.chat_id, paper.label)
    pdf_path, pdf_url = download_pdf(paper, context.cache_dir)
    source = extract_focus_excerpt(pdf_path, normalized_focus)
    return {
        "paper": _paper_payload(paper),
        "focus": normalized_focus,
        "pdf_url": pdf_url,
        "selected_pages": list(source.selected_pages),
        "excerpt": source.excerpt,
    }


def register_paper_tools() -> None:
    register(
        "paper_recommend_today",
        {
            "description": (
                "检索并返回今天的病理图像与持续学习论文推荐；同一天重复调用复用已保存结果"
            ),
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        _recommend_today,
        risk="low",
    )
    register(
        "paper_read",
        {
            "description": (
                "读取已推荐论文的公开 PDF，按实验、方法或局限提取带页码的原文证据"
            ),
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "论文编号，如 P1；省略时使用当前论文",
                },
                "focus": {
                    "type": "string",
                    "enum": ["实验", "方法", "局限"],
                    "description": "希望阅读的部分",
                },
            },
            "additionalProperties": False,
        },
        _read_paper,
        risk="low",
    )
