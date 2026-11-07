"""Build a small daily digest from the public PaperNotes search index.

The implementation intentionally avoids a browser.  PaperNotes is a static
MkDocs site whose public search index already contains the metadata needed for
candidate retrieval.  Only the selected paper pages are downloaded afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Iterable
from urllib.parse import urljoin
from urllib.request import Request, urlopen


BASE_URL = "https://papernotes.org/"
SEARCH_INDEX_URL = urljoin(BASE_URL, "search/search_index.json")
USER_AGENT = "MyAgentPaperDigest/0.1 (+https://github.com/FuYou20/code_agent)"

PATHOLOGY_TERMS = (
    "pathology",
    "histopathology",
    "histopathological",
    "histology",
    "whole slide",
    "whole-slide",
    "wsi",
    "digital pathology",
    "computational pathology",
    "病理",
    "组织病理",
    "全切片",
)

CONTINUAL_TERMS = (
    "continual learning",
    "incremental learning",
    "class-incremental",
    "domain-incremental",
    "lifelong learning",
    "catastrophic forgetting",
    "持续学习",
    "增量学习",
    "灾难性遗忘",
)


@dataclass(frozen=True)
class PaperCandidate:
    location: str
    title: str
    text: str
    tags: tuple[str, ...]
    topics: frozenset[str]

    @property
    def url(self) -> str:
        return urljoin(BASE_URL, self.location)

    @property
    def year(self) -> int:
        match = re.search(r"(?:19|20)\d{2}", self.location)
        return int(match.group(0)) if match else 0


@dataclass(frozen=True)
class PaperRecommendation:
    topic: str
    title: str
    venue: str
    arxiv_month: str
    problem: str
    url: str
    is_intersection: bool
    arxiv_id: str = ""


class _ArticleTextParser(HTMLParser):
    """Extract visible text only from the main MkDocs article."""

    _BLOCK_TAGS = {
        "article",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "li",
        "p",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._article_depth = 0
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = dict(attrs).get("class", "") or ""
        if tag == "article" and "md-content__inner" in classes.split():
            self._article_depth = 1
            return
        if not self._article_depth:
            return
        self._article_depth += 1
        if tag in {"script", "style", "nav"}:
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS and not self._skip_depth:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if not self._article_depth:
            return
        if tag in {"script", "style", "nav"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in self._BLOCK_TAGS and not self._skip_depth:
            self._parts.append("\n")
        self._article_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._article_depth and not self._skip_depth:
            value = data.strip()
            if value:
                self._parts.append(value)

    def text(self) -> str:
        return "\n".join(
            line.strip()
            for line in "".join(self._parts).splitlines()
            if line.strip()
        )


def _fetch(url: str, *, timeout: float = 30.0) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def load_search_index() -> list[dict]:
    payload = json.loads(_fetch(SEARCH_INDEX_URL).decode("utf-8"))
    docs = payload.get("docs", [])
    if not isinstance(docs, list):
        raise ValueError("PaperNotes 搜索索引缺少 docs 列表")
    return docs


def _contains_term(text: str, term: str) -> bool:
    if term.isascii() and re.fullmatch(r"[a-z0-9_-]+", term):
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None
    return term in text


def _matches(text: str, terms: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(_contains_term(lowered, term.lower()) for term in terms)


def build_candidates(docs: Iterable[dict]) -> list[PaperCandidate]:
    candidates: list[PaperCandidate] = []
    seen_locations: set[str] = set()
    for raw in docs:
        location = str(raw.get("location", "")).strip()
        title = str(raw.get("title", "")).strip()
        if not location or not title or location in seen_locations:
            continue
        tags = tuple(str(tag) for tag in raw.get("tags", []) if str(tag).strip())
        text = str(raw.get("text", ""))
        searchable = " ".join((location, title, text, " ".join(tags))).lower()
        topics: set[str] = set()
        if _matches(searchable, PATHOLOGY_TERMS):
            topics.add("pathology")
        if _matches(searchable, CONTINUAL_TERMS):
            topics.add("continual")
        if not topics:
            continue
        seen_locations.add(location)
        candidates.append(
            PaperCandidate(
                location=location,
                title=title,
                text=text,
                tags=tags,
                topics=frozenset(topics),
            )
        )
    return candidates


def load_history(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def select_daily_candidates(
    candidates: Iterable[PaperCandidate],
    history: dict[str, dict],
) -> list[tuple[str, PaperCandidate]]:
    pool = list(candidates)

    def rank(item: PaperCandidate, topic: str) -> tuple[int, int, int, str]:
        searchable = " ".join((item.location, item.title, item.text, " ".join(item.tags)))
        terms = PATHOLOGY_TERMS if topic == "pathology" else CONTINUAL_TERMS
        return (
            1 if len(item.topics) > 1 else 0,
            item.year,
            _matches(searchable.lower(), terms),
            item.location,
        )

    selected: list[tuple[str, PaperCandidate]] = []
    selected_urls: set[str] = set()
    for topic in ("pathology", "continual"):
        eligible = [
            item
            for item in pool
            if topic in item.topics
            and item.url not in history
            and item.url not in selected_urls
        ]
        if not eligible:
            eligible = [
                item
                for item in pool
                if topic in item.topics and item.url not in selected_urls
            ]
        if not eligible:
            continue
        choice = max(eligible, key=lambda item: rank(item, topic))
        selected.append((topic, choice))
        selected_urls.add(choice.url)
    return selected


def _scholarly_metadata(html: str) -> dict:
    for raw in re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        try:
            value = json.loads(unescape(raw))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("@type") == "ScholarlyArticle":
            return value
    return {}


def _clean_title(value: str) -> str:
    return re.sub(r"^\[(?:论文解读|论文笔记|Paper Note)\]\s*", "", value).strip()


def _clip(value: str, limit: int = 260) -> str:
    value = re.sub(r"\s+", " ", value).strip(" ：:")
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip("，,；;。 ") + "…"


def _extract_problem(article_text: str, description: str) -> str:
    flattened = re.sub(r"\s+", " ", article_text)
    # Related-paper summaries near the bottom can contain the same labels.  A
    # problem statement must come from the motivation portion before methods.
    for method_heading in ("方法详解", "方法概述", "本文方法"):
        if method_heading in flattened:
            flattened = flattened.split(method_heading, 1)[0]
            break
    labels = (
        "现有痛点",
        "核心问题",
        "研究问题",
        "问题定义",
        "现有方法的不足",
        "主要挑战",
    )
    stops = (
        "核心矛盾",
        "本文目标",
        "切入角度",
        "核心 idea",
        "核心思想",
        "方法详解",
        "方法概述",
        "本文方法",
    )
    stop_pattern = "|".join(re.escape(stop) for stop in stops)
    for label in labels:
        match = re.search(
            rf"{re.escape(label)}\s*[：:]?\s*(.*?)(?={stop_pattern}|$)",
            flattened,
            flags=re.IGNORECASE,
        )
        if match and len(match.group(1).strip()) >= 20:
            return _clip(match.group(1))

    summary_match = re.search(
        r"一句话总结\s*¶?\s*(.*?)(?=研究背景与动机|背景与动机|$)",
        flattened,
        flags=re.IGNORECASE,
    )
    summary = summary_match.group(1).strip() if summary_match else ""

    def problem_clause(value: str) -> str:
        value = re.sub(r"^(?:\[[^]]+\])+\s*", "", value).strip()
        if not value:
            return ""
        # PaperNotes one-line summaries commonly use "针对 X，提出 Y".  The
        # digest wants X, not a compressed retelling of the whole method.
        match = re.search(r"^(.*?)(?:，|,)(?:本文)?提出", value)
        if match and len(match.group(1).strip()) >= 15:
            return _clip(match.group(1))
        return ""

    concise_problem = problem_clause(summary) or problem_clause(description)
    if concise_problem:
        return concise_problem
    cleaned = re.sub(r"^(?:\[[^]]+\])+\s*", "", description)
    return _clip(summary or cleaned or "PaperNotes 页面未提供可解析的问题摘要")


def parse_recommendation(topic: str, candidate: PaperCandidate, html: str) -> PaperRecommendation:
    metadata = _scholarly_metadata(html)
    parser = _ArticleTextParser()
    parser.feed(html)
    article_text = parser.text()
    description = str(metadata.get("description", candidate.text))
    title = _clean_title(str(metadata.get("headline", candidate.title)))

    venue_match = re.search(r"\[((?:19|20)\d{2}|[^]]*(?:19|20)\d{2})\]", description)
    if venue_match:
        venue = venue_match.group(1).strip()
    else:
        location_match = re.match(r"([A-Za-z]+)((?:19|20)\d{2})/", candidate.location)
        venue = (
            f"{location_match.group(1)} {location_match.group(2)}"
            if location_match
            else str(candidate.year or "时间待核实")
        )

    arxiv_match = re.search(r"arXiv\s*[：:]?\s*(\d{4})\.(\d{4,5})", article_text, re.I)
    if arxiv_match:
        prefix = arxiv_match.group(1)
        arxiv_month = f"20{prefix[:2]}-{prefix[2:]}"
        arxiv_id = f"{arxiv_match.group(1)}.{arxiv_match.group(2)}"
    else:
        arxiv_month = "未标注"
        arxiv_id = ""

    return PaperRecommendation(
        topic=topic,
        title=title,
        venue=venue,
        arxiv_month=arxiv_month,
        problem=_extract_problem(article_text, description),
        url=candidate.url,
        is_intersection=len(candidate.topics) > 1,
        arxiv_id=arxiv_id,
    )


def recommend_daily(history_path: Path) -> list[PaperRecommendation]:
    history = load_history(history_path)
    candidates = build_candidates(load_search_index())
    selected = select_daily_candidates(candidates, history)
    recommendations: list[PaperRecommendation] = []
    for topic, candidate in selected:
        html = _fetch(candidate.url).decode("utf-8", errors="replace")
        recommendations.append(parse_recommendation(topic, candidate, html))
    return recommendations


def record_recommendations(
    history_path: Path,
    recommendations: Iterable[PaperRecommendation],
    *,
    recommended_on: date | None = None,
) -> None:
    history = load_history(history_path)
    day = (recommended_on or date.today()).isoformat()
    for item in recommendations:
        history[item.url] = {
            "title": item.title,
            "topic": item.topic,
            "recommended_on": day,
        }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def format_digest(
    recommendations: Iterable[PaperRecommendation],
    *,
    digest_date: date | None = None,
) -> str:
    labels = {"pathology": "病理图像", "continual": "持续学习"}
    blocks = [f"📚 {(digest_date or date.today()).isoformat()} 每日论文推荐"]
    for index, item in enumerate(recommendations, start=1):
        label = labels.get(item.topic, item.topic)
        if item.is_intersection:
            label += " × 交叉方向"
        blocks.append(
            "\n".join(
                (
                    f"\n【{index}. {label}】",
                    item.title,
                    f"发表时间：{item.venue}；arXiv 首次公开月份：{item.arxiv_month}",
                    f"主要问题：{item.problem}",
                    f"原文解读：{item.url}",
                )
            )
        )
    return "\n".join(blocks)
