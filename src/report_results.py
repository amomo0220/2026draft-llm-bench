"""答え合わせの表（後編・X・Slack 用）。Markdown と、note に貼る PNG 画像を作る。

採点そのものは src/scorer.py が行う。ここでは採点結果と実際の結果を、読みやすい表に並べるだけ。

    uv run python -m src.report_results                      # Markdown を表示
    CHROMIUM_PATH=/opt/pw-browsers/chromium uv run python -m src.report_results --images   # results/images/ に PNG
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

from src.actual import ActualDraft, load_actual
from src.baselines import consensus, load_media_picks
from src.candidates import load_candidates
from src.matching import Master, normalize_name
from src.report import (
    CONDITION_NAMES,
    TEAM_NAMES,
    TEAM_ORDER,
    Section,
    Table,
    _mode,
    group,
    label,
)
from src.schema import FirstRoundBid, Prediction, TeamCode
from src.scorer import (
    RunScore,
    brier,
    load_name_checks,
    load_predictions,
    score_run,
    top_candidate,
)

BASELINE_LABEL = "メディア予想（多数決）"


def _same(a: str, b: str) -> bool:
    return normalize_name(a) == normalize_name(b)


def _range(values: list[float], fmt: str = "{:.1f}") -> str:
    """平均（最小〜最大）。"""
    mean = sum(values) / len(values)
    lo, hi = min(values), max(values)
    return fmt.format(mean) if lo == hi else f"{fmt.format(mean)}（{lo:g}〜{hi:g}）"


def summary_table(scores: list[RunScore], baseline: list[FirstRoundBid], actual: ActualDraft) -> Table:
    by: dict[tuple[str, str], list[RunScore]] = defaultdict(list)
    for s in scores:
        by[(s.condition, s.model_id)].append(s)
    rows = []
    for (cond, m), ss in sorted(by.items()):
        rows.append(
            [
                f"{cond}「{CONDITION_NAMES.get(cond, cond)}」",
                label(m),
                _range([s.first_round_hits for s in ss]),
                f"{sum(s.first_round_brier for s in ss) / len(ss):.2f}",
                _range([s.first_round_position_hits for s in ss]),
                _range([s.lotte_round_hits for s in ss]),
                _range([s.lotte_top5_hits for s in ss]),
            ]
        )
    if baseline:
        bt = actual.by_team()
        hits = sum(_same(top_candidate(b).player_name, bt[b.team].bid.player_name) for b in baseline)
        pos = sum(top_candidate(b).position == bt[b.team].bid.position for b in baseline)
        mean_brier = sum(brier(b, bt[b.team].bid.player_name) for b in baseline) / len(baseline)
        n = len(baseline)
        rows.append(["—", BASELINE_LABEL, f"{hits}（{n}球団中）", f"{mean_brier:.2f}", f"{pos}（{n}球団中）", "—", "—"])
    header = ["条件", "予想した AI", "1位的中", "ブライアー", "ポジション", "ロッテ順位", "ロッテ獲得"]
    return Table(header, rows)


def first_round_result_table(runs_by_model: dict[str, list[Prediction]], actual: ActualDraft, baseline: list[FirstRoundBid]) -> Table:
    """球団ごとの答え合わせ。当たったら「○ 回数/実行数」、外れたら「× 最も多く挙げた選手」。"""
    models = sorted(runs_by_model)
    base = {b.team: b for b in baseline}
    bt = actual.by_team()
    rows, highlight = [], []
    for team in TEAM_ORDER:
        truth = bt[team].bid
        row = [TEAM_NAMES[team], f"{truth.player_name}・{truth.position.value}"]
        any_hit = False
        for m in models:
            refs = [top_candidate(b) for p in runs_by_model[m] for b in p.output.first_round if b.team == team]
            k = sum(_same(r.player_name, truth.player_name) for r in refs)
            any_hit |= k > 0
            row.append(f"○ {k}/{len(refs)}" if k else f"× {_mode(refs)[0].player_name}")
        if baseline:
            b = base.get(team)
            if b is None:
                row.append("—")
            else:
                top = top_candidate(b)
                row.append("○" if _same(top.player_name, truth.player_name) else f"× {top.player_name}")
        rows.append(row)
        highlight.append(any_hit)
    header = ["球団", "実際の1位入札", *map(label, models), *([BASELINE_LABEL] if baseline else [])]
    return Table(header, rows, highlight)


def lotte_result_table(runs_by_model: dict[str, list[Prediction]], actual: ActualDraft) -> Table:
    """ロッテの答え合わせ。○＝その順位で的中、△＝別の順位で挙げていた、×＝外れ（最も多く挙げた選手）。"""
    truth = {1: actual.by_team()[TeamCode.M].bid}
    truth.update({p.round: p for p in actual.lotte})
    models = sorted(runs_by_model)
    rows = []
    for rnd in range(1, 6):
        t = truth.get(rnd)
        row = [f"{rnd}位", f"{t.player_name}・{t.position.value}" if t else "（指名なし）"]
        for m in models:
            runs = runs_by_model[m]
            picks = [pk for p in runs for pk in p.output.lotte if pk.round == rnd]
            if t is None:
                row.append("—")
                continue
            k = sum(_same(pk.player_name, t.player_name) for pk in picks)
            other = sum(_same(pk.player_name, t.player_name) for p in runs for pk in p.output.lotte if pk.round != rnd)
            if k:
                row.append(f"○ {k}/{len(picks)}")
            elif other:
                row.append(f"△ 別の順位で{other}回")
            else:
                row.append(f"× {_mode(picks)[0].player_name}")
        rows.append(row)
    return Table(["順位", "実際の指名", *map(label, models)], rows)


def sections(preds: list[Prediction], actual: ActualDraft, master: Master, checks, baseline: list[FirstRoundBid]) -> list[Section]:
    scores = [score_run(p, actual, master, checks) for p in preds]
    groups = group(preds)
    out = [
        Section(
            "results_summary",
            "答え合わせ：条件・AI ごとの成績",
            "各3回の平均（最小〜最大）。1位的中・ポジション＝12球団中、本命の選手／ポジションが実際の1位入札と一致した数。"
            "ブライアー＝確率の当たり具合（0が最良）。ロッテ順位＝1〜5位のうち順位どおり当たった数。ロッテ獲得＝予想5人のうち実際に指名された人数。"
            "メディア予想は10/13〜18の2媒体の多数決で、AI（10/15）より後の記事を含む。",
            summary_table(scores, baseline, actual),
            style="results",
        )
    ]
    for cond in sorted(groups):
        name = f"条件{cond}「{CONDITION_NAMES.get(cond, cond)}」"
        out.append(
            Section(
                f"results_first_round_{cond}",
                f"1位入札の答え合わせ（{name}）",
                "○ は的中した回数 / 予想の回数。× は外れで、最も多く挙げた選手。色付きの行は、どれかの AI が当てた球団。",
                first_round_result_table(groups[cond], actual, baseline),
                style="results",
            )
        )
        out.append(
            Section(
                f"results_lotte_{cond}",
                f"ロッテの答え合わせ（{name}）",
                "○ はその順位で的中（回数 / 予想の回数）、△ は別の順位で挙げていた、× は外れで、最も多く挙げた選手。1位は1回目の入札。",
                lotte_result_table(groups[cond], actual),
                style="results",
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actual", type=Path, default=Path("data/actual/2026.json"))
    ap.add_argument("--predictions", type=Path, default=Path("predictions"))
    ap.add_argument("--candidates", type=Path, default=Path("data/candidates.csv"))
    ap.add_argument("--name-checks", type=Path, default=Path("data/name_checks.csv"))
    ap.add_argument("--baselines", type=Path, default=Path("data/baselines/media_picks.csv"))
    ap.add_argument("--images", action="store_true", help="results/images/ に PNG を作る")
    ap.add_argument("--out", type=Path, default=Path("results/images"))
    ap.add_argument("--chromium", default=os.environ.get("CHROMIUM_PATH"))
    args = ap.parse_args(argv)

    preds = load_predictions(args.predictions, include_late=False)
    baseline = consensus(load_media_picks(args.baselines)) if args.baselines.exists() else []
    secs = sections(
        preds, load_actual(args.actual), Master(load_candidates(args.candidates)), load_name_checks(args.name_checks), baseline
    )
    if args.images:
        from src.report_images import render

        for path in render(secs, args.out, args.chromium):
            print(path)
    else:
        for s in secs:
            print(f"## {s.title}\n\n{s.note}\n\n{s.table}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
