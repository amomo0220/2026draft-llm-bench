"""事前登録した予測を、記事に貼る Markdown の表にまとめる（docs/PLAN.md 6章）。

記事の数字は手で写さず、このスクリプトの出力から貼る。採点（正解との比較）は src/scorer.py が行う。

    uv run python -m src.report > results/prereg_report.md
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.candidates import load_candidates
from src.matching import Master, MatchStatus, normalize_name
from src.run import RunLog
from src.schema import Prediction, TeamCode
from src.scorer import _mentions, load_predictions, top_candidate

TEAM_NAMES = {
    TeamCode.G: "巨人",
    TeamCode.T: "阪神",
    TeamCode.DB: "DeNA",
    TeamCode.C: "広島",
    TeamCode.S: "ヤクルト",
    TeamCode.D: "中日",
    TeamCode.H: "ソフトバンク",
    TeamCode.F: "日本ハム",
    TeamCode.B: "オリックス",
    TeamCode.E: "楽天",
    TeamCode.L: "西武",
    TeamCode.M: "ロッテ",
}
# 記事に載せる順（セ・リーグ → パ・リーグ）
TEAM_ORDER = [TeamCode.G, TeamCode.T, TeamCode.DB, TeamCode.C, TeamCode.S, TeamCode.D,
              TeamCode.H, TeamCode.F, TeamCode.B, TeamCode.E, TeamCode.L, TeamCode.M]  # fmt: skip
CONDITION_NAMES = {"A": "情報なし", "B": "候補者名簿のみ", "C": "Web検索", "D": "エージェント"}

Groups = dict[str, dict[str, list[Prediction]]]  # condition → model_id → 実行の一覧


def group(preds: list[Prediction]) -> Groups:
    out: Groups = defaultdict(lambda: defaultdict(list))
    for p in sorted(preds, key=lambda p: p.run_id):
        out[p.condition][p.model_id].append(p)
    return out


def _mode(names: list[str]) -> tuple[str, int]:
    """最も多い名前と、その回数。同数なら先に出た方（実行番号の小さい方）。"""
    counts = Counter(normalize_name(n) for n in names)
    best = max(counts.values())
    name = next(n for n in names if counts[normalize_name(n)] == best)
    return name, best


def _cell(names: list[str]) -> str:
    if not names:
        return "—"
    name, k = _mode(names)
    return f"{name}（{k}/{len(names)}）"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def first_round_table(runs_by_model: dict[str, list[Prediction]]) -> str:
    """球団ごとの1位入札の予想。セルは「最も多く1位に挙げた選手（その回数/実行数）」。"""
    models = sorted(runs_by_model)
    rows = []
    for team in TEAM_ORDER:
        row = [TEAM_NAMES[team]]
        for m in models:
            names = [top_candidate(b).player_name for p in runs_by_model[m] for b in p.output.first_round if b.team == team]
            row.append(_cell(names))
        rows.append(row)
    return _table(["球団", *models], rows)


def contest_table(runs_by_model: dict[str, list[Prediction]], min_teams: float = 2.0) -> str:
    """1位で入札が重なると予想された選手。値は、その選手を最有力に挙げた球団数の平均（実行ごとに数える）。"""
    models = sorted(runs_by_model)
    per_model: dict[str, dict[str, float]] = {}
    display: dict[str, str] = {}
    for m in models:
        runs = runs_by_model[m]
        total: Counter[str] = Counter()
        for p in runs:
            for b in p.output.first_round:
                name = top_candidate(b).player_name
                display.setdefault(normalize_name(name), name)
                total[normalize_name(name)] += 1
        per_model[m] = {k: v / len(runs) for k, v in total.items()}
    keys = [k for k in display if max(per_model[m].get(k, 0.0) for m in models) >= min_teams]
    keys.sort(key=lambda k: -sum(per_model[m].get(k, 0.0) for m in models))
    rows = [[display[k], *(f"{per_model[m].get(k, 0.0):.1f}" for m in models)] for k in keys]
    if not rows:
        return "（入札が重なると予想された選手はいません）"
    return _table(["選手", *models], rows)


def lotte_table(runs_by_model: dict[str, list[Prediction]]) -> str:
    models = sorted(runs_by_model)
    rows = []
    for rnd in range(1, 6):
        row = [f"{rnd}位"]
        for m in models:
            row.append(_cell([pick.player_name for p in runs_by_model[m] for pick in p.output.lotte if pick.round == rnd]))
        rows.append(row)
    return _table(["順位", *models], rows)


def match_table(groups: Groups, master: Master) -> str:
    """予測に出てきた選手（実行ごとに重複を除く）のうち、候補者マスタにない名前の数。"""
    rows = []
    for cond in sorted(groups):
        for m in sorted(groups[cond]):
            runs = groups[cond][m]
            refs = [r for p in runs for r in _mentions(p)]
            unmatched = sum(master.match(r.player_name, r.affiliation).status == MatchStatus.UNMATCHED for r in refs)
            rows.append([cond, m, str(len(runs)), str(len(refs)), str(unmatched)])
    return _table(["条件", "モデル", "実行数", "登場した選手（延べ）", "うちマスタにない名前"], rows)


def cost_table(logs: list[RunLog]) -> str:
    by: dict[tuple[str, str], list[RunLog]] = defaultdict(list)
    for log in logs:
        by[(log.condition, log.model_id)].append(log)
    rows = []
    for (cond, m), ls in sorted(by.items()):
        n = len(ls)
        rows.append(
            [
                cond,
                m,
                f"{sum(log.status == 'ok' for log in ls)}/{n}",
                f"${sum(log.cost_usd for log in ls) / n:.2f}",
                f"{sum(log.duration_sec for log in ls) / n:.0f}秒",
                f"{sum(log.tool_calls for log in ls) / n:.1f}",
                f"{sum(log.function_calls for log in ls) / n:.1f}",
            ]
        )
    total = sum(log.cost_usd for log in logs)
    table = _table(["条件", "モデル", "成功", "コスト（1回平均）", "時間（1回平均）", "Web検索（平均）", "名簿ツール（平均）"], rows)
    return f"{table}\n\n合計 {len(logs)} 回、${total:.2f}"


def build_report(preds: list[Prediction], logs: list[RunLog], master: Master) -> str:
    groups = group(preds)
    parts = ["# 事前登録した予測のまとめ（自動生成: `uv run python -m src.report`）", ""]
    for cond in sorted(groups):
        title = f"条件{cond}（{CONDITION_NAMES.get(cond, cond)}）"
        parts += [
            f"## {title}",
            "",
            "### 12球団の1位入札",
            "",
            "各モデルが最も確率が高いとした選手。（ ）は、その選手を挙げた回数 / 実行数。",
            "",
            first_round_table(groups[cond]),
            "",
            "### 1位入札が重なると予想された選手",
            "",
            "その選手を最有力に挙げた球団数（実行ごとの平均）。どれかのモデルで2球団以上のものだけ。",
            "",
            contest_table(groups[cond]),
            "",
            "### ロッテの1〜5位",
            "",
            lotte_table(groups[cond]),
            "",
        ]
    parts += ["## 候補者マスタとの照合", "", match_table(groups, master), ""]
    if logs:
        parts += ["## コスト・時間", "", cost_table(logs), ""]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, default=Path("predictions"))
    ap.add_argument("--runs", type=Path, default=Path("runs/runs.jsonl"))
    ap.add_argument("--candidates", type=Path, default=Path("data/candidates.csv"))
    args = ap.parse_args(argv)

    preds = load_predictions(args.predictions, include_late=False)
    if not preds:
        print(f"{args.predictions} に予測がありません", file=sys.stderr)
        return 1
    logs = []
    if args.runs.exists():
        lines = args.runs.read_text(encoding="utf-8").splitlines()
        logs = [RunLog.model_validate_json(line) for line in lines if line.strip()]
    print(build_report(preds, logs, Master(load_candidates(args.candidates))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
