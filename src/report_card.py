"""SNS 用の結論カード（PNG、1200×675）。「どの AI か」より「何を渡すか」を1枚で見せる。

1位入札の的中数（12球団中、各3回の平均）を、AI ごとに条件B → C → D の棒で並べ、メディア予想の多数決を基準線で引く。
数字は採点（src/scorer.py）から計算する。手で書かない。

    CHROMIUM_PATH=/opt/pw-browsers/chromium uv run python -m src.report_card   # results/images/conclusion_card.png
"""

import argparse
import html
import os
import sys
from collections import defaultdict
from pathlib import Path

from src.actual import ActualDraft, load_actual
from src.baselines import consensus, load_media_picks
from src.candidates import load_candidates
from src.matching import Master, normalize_name
from src.report import label
from src.scorer import load_name_checks, load_predictions, score_run, top_candidate

CONDITIONS = [("B", "記憶＋名簿"), ("C", "＋Web検索"), ("D", "＋名簿で裏取り（エージェント）")]
# 条件の順に薄い青 → 濃い青（一色の順序付きの色。dataviz の検証スクリプトで確認済み）
RAMP = ["#86b6ef", "#3987e5", "#184f95"]
ACCENT = "#d6332e"  # メディア予想の基準線
INK, INK2, MUTED, GRID = "#1b2a4a", "#4a5872", "#8a94a8", "#e3e7ee"


def hit_means(preds, actual: ActualDraft, master: Master, checks) -> dict[str, dict[str, float]]:
    """モデル → 条件 → 1位入札の的中数の平均。"""
    by: dict[tuple[str, str], list[int]] = defaultdict(list)
    for p in preds:
        by[(p.model_id, p.condition)].append(score_run(p, actual, master, checks).first_round_hits)
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (m, c), hits in by.items():
        out[m][c] = sum(hits) / len(hits)
    return out


def baseline_hits(bids, actual: ActualDraft) -> int:
    bt = actual.by_team()
    return sum(normalize_name(top_candidate(b).player_name) == normalize_name(bt[b.team].bid.player_name) for b in bids)


def card_html(means: dict[str, dict[str, float]], media: int | None) -> str:
    # 棒グラフの座標（SVG 内）
    w, h, left, right, top, bottom = 760, 440, 44, 16, 16, 56
    ymax = 4.0
    plot_w, plot_h = w - left - right, h - top - bottom
    models = sorted(means, key=lambda m: -means[m].get("D", 0))
    group_w = plot_w / len(models)
    bar_w, gap = 50, 4  # 隣り合う棒の間に 2px 以上の隙間

    def y(v: float) -> float:
        return top + plot_h * (1 - v / ymax)

    parts = []
    for t in range(int(ymax) + 1):  # 目盛りと補助線（控えめに）
        parts.append(f'<line x1="{left}" x2="{w - right}" y1="{y(t)}" y2="{y(t)}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y(t) + 5}" text-anchor="end" font-size="14" fill="{MUTED}">{t}</text>')
    for gi, m in enumerate(models):
        cx = left + group_w * (gi + 0.5)
        x0 = cx - (bar_w * 3 + gap * 2) / 2
        for ci, (c, _) in enumerate(CONDITIONS):
            v = means[m].get(c)
            if v is None:
                continue
            x = x0 + ci * (bar_w + gap)
            top_y, base = y(v), y(0)
            r = min(4, base - top_y)  # 値の側だけ角を丸め、基線は直角にする
            d = (
                f"M{x},{base} V{top_y + r} Q{x},{top_y} {x + r},{top_y} H{x + bar_w - r} "
                f"Q{x + bar_w},{top_y} {x + bar_w},{top_y + r} V{base} Z"
            )
            parts.append(f'<path d="{d}" fill="{RAMP[ci]}"/>')
            parts.append(
                f'<text x="{x + bar_w / 2}" y="{top_y - 8}" text-anchor="middle" font-size="17" font-weight="700" fill="{INK}">{v:.1f}</text>'
            )
        parts.append(
            f'<text x="{cx}" y="{h - bottom + 30}" text-anchor="middle" font-size="18" font-weight="700" fill="{INK}">{html.escape(label(m))}</text>'
        )
    if media is not None:  # 基準線（メディア予想の多数決）
        parts.append(
            f'<line x1="{left}" x2="{w - right}" y1="{y(media)}" y2="{y(media)}" stroke="{ACCENT}" stroke-width="2" stroke-dasharray="7 5"/>'
        )
        parts.append(
            f'<text x="{w - right}" y="{y(media) - 8}" text-anchor="end" font-size="14" fill="{INK2}">メディア予想の多数決：{media}</text>'
        )
    svg = f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'

    legend = "".join(
        f'<div class="lg"><span class="sw" style="background:{RAMP[i]}"></span>{html.escape(name)}</div>' for i, (_, name) in enumerate(CONDITIONS)
    )
    # 見出しの数字：同じ AI の B → D の伸び（最大の AI）
    best = max(models, key=lambda m: means[m].get("D", 0) / max(means[m].get("B", 0), 1e-9))
    b, d = means[best].get("B", 0), means[best].get("D", 0)
    ratio = d / b if b else 0
    spread = max(max(means[m].get(c, 0) for m in models) - min(means[m].get(c, 0) for m in models) for c, _ in CONDITIONS)
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700;900&display=swap">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ width: 1200px; height: 675px; background: #ffffff; color: {INK};
  font-family: "Noto Sans JP", "Hiragino Sans", "IPAGothic", sans-serif; }}
#card {{ width: 1200px; height: 675px; border-top: 12px solid {INK}; padding: 34px 56px 0; position: relative; }}
.kicker {{ font-size: 20px; color: {INK2}; font-weight: 700; }}
h1 {{ font-size: 50px; font-weight: 900; margin-top: 6px; letter-spacing: 0.01em; }}
h1 em {{ font-style: normal; color: {ACCENT}; }}
.row {{ display: flex; gap: 36px; margin-top: 18px; }}
.side {{ width: 300px; padding-top: 12px; }}
.big {{ font-size: 64px; font-weight: 900; line-height: 1; }}
.big small {{ font-size: 26px; font-weight: 700; }}
.cap {{ font-size: 17px; color: {INK2}; margin-top: 10px; line-height: 1.6; }}
.cap2 {{ font-size: 17px; color: {INK}; font-weight: 700; margin-top: 14px; line-height: 1.6; }}
.lgs {{ margin-top: 26px; }}
.lg {{ font-size: 16px; color: {INK2}; display: flex; align-items: center; gap: 10px; margin-top: 8px; }}
.sw {{ width: 18px; height: 18px; border-radius: 4px; display: inline-block; }}
.foot {{ position: absolute; left: 56px; right: 56px; bottom: 20px; font-size: 14px; color: {MUTED}; display: flex; justify-content: space-between; }}
</style></head><body><div id="card">
<div class="kicker">AI 3社のドラフト1位入札予想・答え合わせ</div>
<h1>「どのAIか」より、<em>「何を渡すか」</em></h1>
<div class="row">
  <div>{svg}</div>
  <div class="side">
    <div class="big">{ratio:.0f}<small>倍</small></div>
    <div class="cap">同じ {html.escape(label(best))} でも、<br>渡し方を変えると<br>的中は {b:.1f} → {d:.1f} 球団に。</div>
    <div class="cap2">同じ渡し方での AI どうしの差は<br>最大 {spread:.1f} 球団</div>
    <div class="lgs">{legend}</div>
  </div>
</div>
<div class="foot"><span>1位入札の的中数（12球団中、各3回の平均）。予想は2026年10月15日に事前登録</span><span>github.com/amomo0220/2026draft-llm-bench</span></div>
</div></body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actual", type=Path, default=Path("data/actual/2026.json"))
    ap.add_argument("--predictions", type=Path, default=Path("predictions"))
    ap.add_argument("--candidates", type=Path, default=Path("data/candidates.csv"))
    ap.add_argument("--name-checks", type=Path, default=Path("data/name_checks.csv"))
    ap.add_argument("--baselines", type=Path, default=Path("data/baselines/media_picks.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/images/conclusion_card.png"))
    ap.add_argument("--chromium", default=os.environ.get("CHROMIUM_PATH"))
    args = ap.parse_args(argv)

    actual = load_actual(args.actual)
    preds = load_predictions(args.predictions, include_late=False)
    means = hit_means(preds, actual, Master(load_candidates(args.candidates)), load_name_checks(args.name_checks))
    media = baseline_hits(consensus(load_media_picks(args.baselines)), actual) if args.baselines.exists() else None
    page_html = card_html(means, media)

    from playwright.sync_api import sync_playwright

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium) if args.chromium else p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 675}, device_scale_factor=2)
        page.set_content(page_html, wait_until="load", timeout=30_000)
        page.evaluate("document.fonts.ready")
        page.locator("#card").screenshot(path=str(args.out))
        browser.close()
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
