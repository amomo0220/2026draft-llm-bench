"""src/report.py の表を、note に貼る PNG 画像にする（note は Markdown の表を使えないため）。

HTML を組み立て、Chromium（Playwright）で表ごとにスクリーンショットを撮る。

    uv run python -m src.report_images                 # results/images/ に出力
    CHROMIUM_PATH=/opt/pw-browsers/chromium uv run python -m src.report_images   # クラウド環境

手元で初めて使うときは `uv run playwright install chromium` が必要。
"""

import argparse
import html
import os
import sys
from pathlib import Path

from src.candidates import load_candidates
from src.matching import Master
from src.report import Section, load_logs, sections
from src.scorer import load_predictions

CREDIT = "AI 3社のドラフト予想（2026年10月15日に事前登録） github.com/amomo0220/2026draft-llm-bench"
# 読みやすい日本語フォント。取得できない環境では CSS の後ろのフォントを使う
FONT_LINK = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap">'
WIDTH = 1200  # 画像の幅（CSS ピクセル）。出力は2倍の解像度

# 記事のイラストに合わせた紺・白・赤
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: "Noto Sans JP", "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Noto Sans JP", "Noto Sans CJK JP",
    "IPAGothic", "Yu Gothic", sans-serif;
  background: #ffffff; color: #1b2a4a; width: %(width)dpx;
}
.card { padding: 40px 48px 28px; border-top: 10px solid #1b2a4a; }
h1 { font-size: 30px; font-weight: 700; margin-bottom: 10px; }
.note { font-size: 17px; color: #4a5872; margin-bottom: 22px; line-height: 1.6; }
table { width: 100%%; border-collapse: collapse; font-size: 20px; }
th { background: #1b2a4a; color: #ffffff; font-weight: 700; padding: 12px 14px; text-align: left; }
td { padding: 11px 14px; border-bottom: 1px solid #d9dfea; }
td:first-child { font-weight: 700; white-space: nowrap; }
tr.hl td { background: #fdecec; }
tr.hl td:first-child { box-shadow: inset 6px 0 0 #d6332e; }
.empty { font-size: 20px; padding: 16px 0; }
.credit { margin-top: 20px; font-size: 14px; color: #8a94a8; text-align: right; }
"""


def section_html(sec: Section) -> str:
    t = sec.table
    if t.rows:
        head = "".join(f"<th>{html.escape(h)}</th>" for h in t.header)
        body = ""
        for i, row in enumerate(t.rows):
            cls = ' class="hl"' if i < len(t.highlight) and t.highlight[i] else ""
            body += f"<tr{cls}>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>"
        content = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    else:
        content = f'<p class="empty">{html.escape(t.empty or "（該当なし）")}</p>'
    return (
        f'<!doctype html><html lang="ja"><head><meta charset="utf-8">{FONT_LINK}<style>{CSS % {"width": WIDTH}}</style></head>'
        f'<body><div class="card" id="card"><h1>{html.escape(sec.title)}</h1><p class="note">{html.escape(sec.note)}</p>'
        f'{content}<p class="credit">{html.escape(CREDIT)}</p></div></body></html>'
    )


def render(secs: list[Section], out_dir: Path, chromium: str | None = None) -> list[Path]:
    # 画像を作るときだけ使う（開発用の依存）
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium) if chromium else p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": 800}, device_scale_factor=2)
        for sec in secs:
            page.set_content(section_html(sec), wait_until="load", timeout=30_000)
            page.evaluate("document.fonts.ready")
            path = out_dir / f"{sec.key}.png"
            page.locator("#card").screenshot(path=str(path))
            paths.append(path)
        browser.close()
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, default=Path("predictions"))
    ap.add_argument("--runs", type=Path, default=Path("runs/runs.jsonl"))
    ap.add_argument("--candidates", type=Path, default=Path("data/candidates.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/images"))
    ap.add_argument("--chromium", default=os.environ.get("CHROMIUM_PATH"), help="Chromium の実行ファイル（既定: Playwright のもの）")
    args = ap.parse_args(argv)

    preds = load_predictions(args.predictions, include_late=False)
    if not preds:
        print(f"{args.predictions} に予測がありません", file=sys.stderr)
        return 1
    secs = sections(preds, load_logs(args.runs), Master(load_candidates(args.candidates)))
    for path in render(secs, args.out, args.chromium):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
