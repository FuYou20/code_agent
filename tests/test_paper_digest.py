from __future__ import annotations

from datetime import date

from myagent.paper_digest import (
    PaperCandidate,
    _contains_term,
    build_candidates,
    format_digest,
    parse_recommendation,
    select_daily_candidates,
)


def candidate(location: str, title: str, topics: set[str]) -> PaperCandidate:
    return PaperCandidate(
        location=location,
        title=title,
        text="",
        tags=(),
        topics=frozenset(topics),
    )


def test_short_wsi_term_does_not_match_browsing() -> None:
    assert not _contains_term("a browsing agent", "wsi")
    assert _contains_term("whole-slide image (wsi)", "wsi")


def test_build_candidates_detects_bilingual_topics() -> None:
    docs = [
        {
            "location": "AAAI2026/medical_imaging/example/",
            "title": "Continual Learning for Whole-Slide Pathology",
            "text": "灾难性遗忘",
            "tags": ["全切片图像"],
        }
    ]
    items = build_candidates(docs)
    assert len(items) == 1
    assert items[0].topics == {"pathology", "continual"}


def test_selection_prefers_intersection_and_keeps_two_distinct() -> None:
    both = candidate(
        "AAAI2026/medical_imaging/both/",
        "Both",
        {"pathology", "continual"},
    )
    pathology = candidate(
        "CVPR2026/medical_imaging/path/", "Path", {"pathology"}
    )
    continual = candidate(
        "NeurIPS2026/optimization/cl/", "CL", {"continual"}
    )
    chosen = select_daily_candidates([pathology, continual, both], {})
    assert chosen[0] == ("pathology", both)
    assert chosen[1] == ("continual", continual)


def test_parse_and_format_recommendation() -> None:
    item = candidate(
        "AAAI2026/medical_imaging/example/",
        "Example",
        {"pathology", "continual"},
    )
    html = """
    <html><head>
      <script type="application/ld+json">
      {"@type":"ScholarlyArticle","headline":"[论文解读] Example Paper",
       "description":"[AAAI 2026][医学图像] 示例摘要"}
      </script>
    </head><body>
      <article class="md-content__inner">
        <p>arXiv: 2511.01234</p>
        <h2>研究背景与动机</h2>
        <p>现有痛点：模型在顺序学习新医院数据时会遗忘旧医院知识，且原始病理数据不能长期保存。</p>
        <p>核心矛盾：需要兼顾稳定性和可塑性。</p>
        <h2>方法详解</h2>
      </article>
    </body></html>
    """
    parsed = parse_recommendation("pathology", item, html)
    assert parsed.title == "Example Paper"
    assert parsed.venue == "AAAI 2026"
    assert parsed.arxiv_month == "2025-11"
    assert parsed.arxiv_id == "2511.01234"
    assert "遗忘旧医院知识" in parsed.problem

    rendered = format_digest([parsed], digest_date=date(2026, 11, 6))
    assert "2026-11-06" in rendered
    assert "病理图像 × 交叉方向" in rendered
    assert "arXiv 首次公开月份：2025-11" in rendered


def test_problem_extraction_ignores_labels_after_method_section() -> None:
    item = candidate(
        "NeurIPS2026/anomaly_detection/example/",
        "Benchmark",
        {"continual"},
    )
    html = """
    <html><head><script type="application/ld+json">
    {"@type":"ScholarlyArticle","headline":"Benchmark",
     "description":"[NeurIPS 2026] 针对产线组件持续更替导致模型遗忘历史产品的瓶颈，本文提出一个新基准。"}
    </script></head><body><article class="md-content__inner">
    <h2>一句话总结</h2>
    <p>针对产线组件持续更替导致模型遗忘历史产品的瓶颈，本文提出一个新基准。</p>
    <h2>研究背景与动机</h2><p>背景内容。</p>
    <h2>方法详解</h2><p>相关论文的核心问题：这段文字不能成为摘要。</p>
    </article></body></html>
    """
    parsed = parse_recommendation("continual", item, html)
    assert parsed.problem == "针对产线组件持续更替导致模型遗忘历史产品的瓶颈"
