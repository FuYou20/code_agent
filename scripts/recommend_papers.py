"""Run the minimal PaperNotes daily recommendation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from myagent.home import get_home
from myagent.paper_digest import (
    format_digest,
    recommend_daily,
    record_recommendations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 PaperNotes 推荐病理图像与持续学习论文"
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=get_home() / "paper_digest_history.json",
        help="已推荐论文记录；默认位于 MYAGENT_HOME",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="输出成功后写入历史；手工预览时默认不写入",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    recommendations = recommend_daily(args.history)
    if not recommendations:
        raise SystemExit("PaperNotes 中没有找到匹配候选")
    print(format_digest(recommendations))
    if args.record:
        record_recommendations(args.history, recommendations)


if __name__ == "__main__":
    main()
