"""事前登録した予測を、記事に貼る Markdown の表にまとめる（docs/PLAN.md 6章）。

記事の数字は手で写さず、このスクリプトの出力から貼る。採点（正解との比較）は src/scorer.py が行う。

    uv run python -m src.report > results/prereg_report.md
"""

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.candidates import load_candidates
from src.matching import Master, MatchStatus, normalize_name
from src.run import RunLog
from src.schema import Prediction, TeamCode, _PlayerRef
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
CONDITION_NAMES = {"A": "情報なし", "B": "記憶＋名簿", "C": "Web検索", "D": "エージェント"}
MODEL_LABELS = {"claude-sonnet-5-5": "Claude Sonnet 5.5", "gpt-5.6-terra": "GPT-5.6 Terra", "gemini-3.8-flash": "Gemini 3.8 Flash"}


def label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


@dataclass
class Table:
    """記事に載せる表。Markdown（str）と画像（src/report_images.py）の両方に使う。"""

    header: list[str]
    rows: list[list[str]]
    highlight: list[bool] = field(default_factory=list)  # 行ごと。画像で色を付ける（例: 全モデルの予想が一致）
    empty: str = ""  # 行がないときに表の代わりに出す文

    def __str__(self) -> str:
        if not self.rows and self.empty:
            return self.empty
        lines = ["| " + " | ".join(self.header) + " |", "| " + " | ".join("---" for _ in self.header) + " |"]
        lines += ["| " + " | ".join(r) + " |" for r in self.rows]
        return "\n".join(lines)

Groups = dict[str, dict[str, list[Prediction]]]  # condition → model_id → 実行の一覧


def group(preds: list[Prediction]) -> Groups:
    out: Groups = defaultdict(lambda: defaultdict(list))
    for p in sorted(preds, key=lambda p: p.run_id):
        out[p.condition][p.model_id].append(p)
    return out


def _mode[R: _PlayerRef](refs: list[R]) -> tuple[R, int]:
    """最も多く挙がった選手（最初に挙げた実行の記述）と、その回数。同数なら先に出た方（実行番号の小さい方）。"""
    counts = Counter(normalize_name(r.player_name) for r in refs)
    best = max(counts.values())
    ref = next(r for r in refs if counts[normalize_name(r.player_name)] == best)
    return ref, best


def _cell(refs: list[_PlayerRef]) -> str:
    """「選手名・ポジション（回数/実行数）」。"""
    if not refs:
        return "—"
    ref, k = _mode(refs)
    return f"{ref.player_name}・{ref.position.value}（{k}/{len(refs)}）"


def first_round_table(runs_by_model: dict[str, list[Prediction]]) -> Table:
    """球団ごとの1位入札の予想。セルは「最も多く1位に挙げた選手（その回数/実行数）」。全モデルが同じ選手なら強調する。"""
    models = sorted(runs_by_model)
    rows, highlight = [], []
    for team in TEAM_ORDER:
        row = [TEAM_NAMES[team]]
        modes = set()
        for m in models:
            refs = [top_candidate(b) for p in runs_by_model[m] for b in p.output.first_round if b.team == team]
            row.append(_cell(refs))
            modes.add(normalize_name(_mode(refs)[0].player_name) if refs else None)
        rows.append(row)
        highlight.append(len(models) > 1 and len(modes) == 1 and None not in modes)
    return Table(["球団", *map(label, models)], rows, highlight)


def contest_table(runs_by_model: dict[str, list[Prediction]], min_teams: float = 2.0) -> Table:
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
    return Table(["選手", *map(label, models)], rows, empty="（入札が重なると予想された選手はいません）")


def lotte_table(runs_by_model: dict[str, list[Prediction]]) -> Table:
    models = sorted(runs_by_model)
    rows = []
    for rnd in range(1, 6):
        row = [f"{rnd}位"]
        for m in models:
            row.append(_cell([pick for p in runs_by_model[m] for pick in p.output.lotte if pick.round == rnd]))
        rows.append(row)
    return Table(["順位", *map(label, models)], rows)


def lotte_reason_table(runs: list[Prediction]) -> Table:
    """1つのモデルのロッテ1〜5位。順位ごとに最も多く挙がった選手と、その選手を挙げた最初の実行の理由。"""
    rows = []
    for rnd in range(1, 6):
        picks = [pick for p in runs for pick in p.output.lotte if pick.round == rnd]
        if not picks:
            rows.append([f"{rnd}位", "—", ""])
            continue
        pick, k = _mode(picks)
        rows.append([f"{rnd}位", f"{pick.player_name}・{pick.position.value}（{k}/{len(picks)}）", pick.reason])
    return Table(["順位", "予想した選手", "AI が書いた理由"], rows)


def match_table(groups: Groups, master: Master) -> Table:
    """予測に出てきた選手（実行ごとに重複を除く）のうち、候補者マスタにない名前の数。"""
    rows = []
    for cond in sorted(groups):
        for m in sorted(groups[cond]):
            runs = groups[cond][m]
            refs = [r for p in runs for r in _mentions(p)]
            unmatched = sum(master.match(r.player_name, r.affiliation).status == MatchStatus.UNMATCHED for r in refs)
            rows.append([cond, label(m), str(len(runs)), str(len(refs)), str(unmatched)])
    return Table(["条件", "モデル", "実行数", "登場した選手（延べ）", "うちマスタにない名前"], rows)


def cost_table(logs: list[RunLog]) -> Table:
    by: dict[tuple[str, str], list[RunLog]] = defaultdict(list)
    for log in logs:
        by[(log.condition, log.model_id)].append(log)
    rows = []
    for (cond, m), ls in sorted(by.items()):
        n = len(ls)
        rows.append(
            [
                cond,
                label(m),
                f"{sum(log.status == 'ok' for log in ls)}/{n}",
                f"${sum(log.cost_usd for log in ls) / n:.2f}",
                f"{sum(log.duration_sec for log in ls) / n:.0f}秒",
                f"{sum(log.tool_calls for log in ls) / n:.1f}",
                f"{sum(log.function_calls for log in ls) / n:.1f}",
            ]
        )
    header = ["条件", "モデル", "成功", "コスト（1回平均）", "時間（1回平均）", "Web検索（平均）", "名簿ツール（平均）"]
    return Table(header, rows)


def cost_total(logs: list[RunLog]) -> str:
    return f"合計 {len(logs)} 回、${sum(log.cost_usd for log in logs):.2f}"


@dataclass
class Section:
    """画像1枚に対応する単位（見出し・説明・表）。"""

    key: str  # 画像のファイル名に使う
    title: str
    note: str
    table: Table
    style: str = ""  # 画像の表の見た目（"reasons": 理由の長い文を折り返す）


def sections(preds: list[Prediction], logs: list[RunLog], master: Master) -> list[Section]:
    groups = group(preds)
    out: list[Section] = []
    for cond in sorted(groups):
        name = f"条件{cond}「{CONDITION_NAMES.get(cond, cond)}」"
        by_model = groups[cond]
        out += [
            Section(
                f"{cond}_first_round",
                f"12球団の1位入札予想（{name}）",
                "各AIがいちばん確率が高いとした選手とポジション。（ ）は、その選手を挙げた回数 / 予想の回数。色付きの行は全AIの予想が一致。",
                first_round_table(by_model),
            ),
            Section(
                f"{cond}_contest",
                f"1位入札が重なると予想された選手（{name}）",
                "その選手を1位の本命に挙げた球団の数（予想ごとの平均）。どれかのAIで2球団以上のものだけ。",
                contest_table(by_model),
            ),
            Section(f"{cond}_lotte", f"ロッテの1〜5位予想（{name}）", "選手とポジション。（ ）は、その選手を挙げた回数 / 予想の回数。", lotte_table(by_model)),
        ]
        out += [
            Section(
                f"{cond}_lotte_reasons_{m}",
                f"ロッテの1〜5位予想と理由：{label(m)}（{name}）",
                "順位ごとに最も多く挙がった選手と、その選手を挙げた回の理由（AI の出力をそのまま掲載）。（ ）は、挙げた回数 / 予想の回数。",
                lotte_reason_table(by_model[m]),
                style="reasons",
            )
            for m in sorted(by_model)
        ]
    out.append(
        Section("match", "候補者名簿との照合", "予想に出てきた選手（予想ごとに重複を除いて数えた延べ人数）のうち、名簿にない名前の数。", match_table(groups, master))
    )
    if logs:
        out.append(Section("cost", "かかった費用と時間", cost_total(logs) + "（失敗してやり直した回も含む）", cost_table(logs)))
    return out


def build_report(preds: list[Prediction], logs: list[RunLog], master: Master) -> str:
    parts = ["# 事前登録した予測のまとめ（自動生成: `uv run python -m src.report`）", ""]
    for sec in sections(preds, logs, master):
        parts += [f"## {sec.title}", "", sec.note, "", str(sec.table), ""]
    return "\n".join(parts)


def load_logs(path: Path) -> list[RunLog]:
    if not path.exists():
        return []
    return [RunLog.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
    print(build_report(preds, load_logs(args.runs), Master(load_candidates(args.candidates))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
