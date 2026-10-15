"""予測を実際の結果と照合して採点する（docs/PLAN.md 3.6、最小構成）。

対象: 12球団の1位入札、ロッテ1〜5位、ハルシネーション（3.4 の分類）。
未実装: 競合予測精度、指名順の順位相関、出典の裏付け率、安定性、丸写し度。

採点ロジックは事前登録後も修正してよい。予測ファイルは読むだけで変更しない。

使い方:
    uv run python -m src.scorer --actual data/actual/2026.json
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from datetime import date
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, HttpUrl

from src.actual import ActualDraft, load_actual
from src.candidates import load_candidates
from src.matching import Master, MatchStatus, normalize_name, same_affiliation, to_positions
from src.schema import FirstRoundBid, NonEmptyStr, PlayerProb, Prediction, TeamCode, _PlayerRef


class Verdict(StrEnum):
    """マスタにない名前を一次情報で確認した結果（PLAN.md 3.4）。実在・指名可能ならマスタに追加する。"""

    INELIGIBLE = "ineligible"  # 実在・指名不可（志望届未提出の高校生・大学生、NPB所属選手など）
    NOT_FOUND = "not_found"  # 実在を確認できない（ハルシネーション）


class NameCheck(BaseModel):
    """data/name_checks.csv の1行。"""

    model_config = ConfigDict(extra="forbid")

    name: NonEmptyStr
    verdict: Verdict
    checked_on: date
    source_url: HttpUrl | None = None
    note: str = ""


def load_name_checks(path: Path) -> dict[str, NameCheck]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        rows = [NameCheck.model_validate({k: v for k, v in r.items() if v != ""}) for r in csv.DictReader(f)]
    return {normalize_name(r.name): r for r in rows}


class RunScore(BaseModel):
    run_id: str
    model_id: str
    condition: str
    # 1位入札（12球団）
    first_round_hits: int  # 最尤候補が実際の入札と一致した球団数
    first_round_brier: float  # 12球団の平均（0が最良、2が最悪）
    first_round_position_hits: int  # 最尤候補のポジションが実際の入札選手と一致した球団数
    # ロッテ1〜5位（1位は入札、2〜5位は実際に指名があった順位のみ）
    lotte_rounds_scored: int
    lotte_round_hits: int
    lotte_position_hits: int
    lotte_top5_hits: int  # 予測5人のうち、ロッテが実際に獲得した選手（1位は最終的な1位）
    # ハルシネーション（予測に出た選手を名前で重複除去して数える）
    names_total: int
    names_matched: int
    names_ambiguous: int
    names_pending: int  # マスタになく、まだ確認していない
    names_ineligible: int
    names_hallucinated: int
    affiliation_errors: int  # マスタと突合できた選手のうち、所属が違うもの
    position_errors: int  # マスタにポジションがある選手のうち、ポジションが違うもの


def top_candidate(bid: FirstRoundBid) -> PlayerProb:
    """確率最大の候補。同率なら先に書かれたもの。"""
    return max(bid.candidates, key=lambda c: c.prob)


def brier(bid: FirstRoundBid, actual_name: str) -> float:
    """多クラスのブライアースコア。候補にない選手が入札された場合は「その他」（確率0）が正解とみなす。"""
    probs: dict[str, float] = defaultdict(float)
    for c in bid.candidates:
        probs[normalize_name(c.player_name)] += c.prob
    total = sum(probs.values())
    y = normalize_name(actual_name)
    score = sum((p / total - (1.0 if k == y else 0.0)) ** 2 for k, p in probs.items())
    return score + (0.0 if y in probs else 1.0)


def _same_player(a: str, b: str) -> bool:
    return normalize_name(a) == normalize_name(b)


def _mentions(pred: Prediction) -> list[_PlayerRef]:
    refs: list[_PlayerRef] = [c for b in pred.output.first_round for c in b.candidates]
    refs += pred.output.contested + pred.output.lotte
    seen: set[str] = set()
    unique: list[_PlayerRef] = []
    for r in refs:
        key = normalize_name(r.player_name)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def score_run(pred: Prediction, actual: ActualDraft, master: Master, checks: dict[str, NameCheck]) -> RunScore:
    by_team = actual.by_team()

    fr_hits = fr_pos = 0
    briers: list[float] = []
    for bid in pred.output.first_round:
        truth = by_team[bid.team].bid
        top = top_candidate(bid)
        fr_hits += _same_player(top.player_name, truth.player_name)
        fr_pos += top.position == truth.position
        briers.append(brier(bid, truth.player_name))

    lotte_truth = {1: by_team[TeamCode.M].bid, **{p.round: p for p in actual.lotte}}
    acquired = [p.player_name for p in actual.lotte_acquired()]
    scored = l_hits = l_pos = top5 = 0
    for pick in pred.output.lotte:
        if any(_same_player(pick.player_name, a) for a in acquired):
            top5 += 1
        truth_pick = lotte_truth.get(pick.round)
        if truth_pick is None:
            continue
        scored += 1
        l_hits += _same_player(pick.player_name, truth_pick.player_name)
        l_pos += pick.position == truth_pick.position

    counts = defaultdict(int)
    mentions = _mentions(pred)
    for ref in mentions:
        m = master.match(ref.player_name, ref.affiliation)
        if m.status == MatchStatus.MATCHED and m.candidate is not None:
            counts["matched"] += 1
            counts["affiliation_errors"] += not same_affiliation(m.candidate.team, ref.affiliation)
            positions = to_positions(m.candidate.position)
            counts["position_errors"] += bool(positions) and ref.position not in positions
        elif m.status == MatchStatus.AMBIGUOUS:
            counts["ambiguous"] += 1
        else:
            check = checks.get(normalize_name(ref.player_name))
            if check is None:
                counts["pending"] += 1
            elif check.verdict == Verdict.INELIGIBLE:
                counts["ineligible"] += 1
            else:
                counts["hallucinated"] += 1

    return RunScore(
        run_id=pred.run_id,
        model_id=pred.model_id,
        condition=pred.condition,
        first_round_hits=fr_hits,
        first_round_brier=statistics.fmean(briers),
        first_round_position_hits=fr_pos,
        lotte_rounds_scored=scored,
        lotte_round_hits=l_hits,
        lotte_position_hits=l_pos,
        lotte_top5_hits=top5,
        names_total=len(mentions),
        names_matched=counts["matched"],
        names_ambiguous=counts["ambiguous"],
        names_pending=counts["pending"],
        names_ineligible=counts["ineligible"],
        names_hallucinated=counts["hallucinated"],
        affiliation_errors=counts["affiliation_errors"],
        position_errors=counts["position_errors"],
    )


def pending_names(preds: list[Prediction], master: Master, checks: dict[str, NameCheck]) -> list[dict[str, str | int]]:
    """マスタになく、まだ確認していない名前の一覧（PLAN.md 3.4 の確認作業用）。出現回数の多い順。"""
    found: dict[str, dict[str, str | int]] = {}
    for pred in preds:
        for ref in _mentions(pred):
            key = normalize_name(ref.player_name)
            if key in checks or master.match(ref.player_name, ref.affiliation).status != MatchStatus.UNMATCHED:
                continue
            row = found.setdefault(
                key,
                {"name": ref.player_name, "affiliation": ref.affiliation, "position": ref.position.value, "runs": 0},
            )
            row["runs"] = int(row["runs"]) + 1
    return sorted(found.values(), key=lambda r: (-int(r["runs"]), str(r["name"])))


SUMMARY_KEYS = [k for k, f in RunScore.model_fields.items() if f.annotation in (int, float)]


def summarize(scores: list[RunScore]) -> list[dict[str, str | int | float]]:
    """モデル × 条件ごとに、各指標の平均と標準偏差（実行が2回以上のとき）を出す。"""
    groups: dict[tuple[str, str], list[RunScore]] = defaultdict(list)
    for s in scores:
        groups[(s.model_id, s.condition)].append(s)
    rows: list[dict[str, str | int | float]] = []
    for (model_id, condition), runs in sorted(groups.items()):
        row: dict[str, str | int | float] = {"model_id": model_id, "condition": condition, "n_runs": len(runs)}
        for key in SUMMARY_KEYS:
            values = [float(getattr(r, key)) for r in runs]
            row[f"{key}_mean"] = round(statistics.fmean(values), 4)
            row[f"{key}_sd"] = round(statistics.stdev(values), 4) if len(values) > 1 else ""
        rows.append(row)
    return rows


def load_predictions(root: Path, include_late: bool) -> list[Prediction]:
    # 条件 D の自己検証前の下書き（<run_id>.draft.json）は予測ではないので読まない
    paths = sorted(
        p
        for p in root.rglob("*.json")
        if not p.name.endswith(".draft.json") and (include_late or "late" not in p.relative_to(root).parts)
    )
    return [Prediction.model_validate_json(p.read_text(encoding="utf-8")) for p in paths]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actual", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, default=Path("predictions"))
    ap.add_argument("--candidates", type=Path, default=Path("data/candidates.csv"))
    ap.add_argument("--name-checks", type=Path, default=Path("data/name_checks.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--include-late", action="store_true", help="predictions/late/ も採点する")
    args = ap.parse_args(argv)

    actual = load_actual(args.actual)
    master = Master(load_candidates(args.candidates))
    checks = load_name_checks(args.name_checks)
    preds = load_predictions(args.predictions, args.include_late)
    if not preds:
        print(f"{args.predictions} に予測がありません", file=sys.stderr)
        return 1
    scores = [score_run(p, actual, master, checks) for p in preds]

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "run_scores.jsonl").open("w", encoding="utf-8") as f:
        for s in scores:
            f.write(s.model_dump_json() + "\n")
    summary = summarize(scores)
    with (args.out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(summary)

    pending = pending_names(preds, master, checks)
    with (args.out / "pending_names.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "affiliation", "position", "runs"], lineterminator="\n")
        w.writeheader()
        w.writerows(pending)

    print(f"{len(scores)}件を採点しました → {args.out}/")
    if pending:
        print(
            f"注意: マスタになく未確認の名前が {len(pending)} 人います（{args.out}/pending_names.csv）。"
            "PLAN.md 3.4 の手順で確認してください",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
