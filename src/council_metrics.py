"""予想会議の観察（docs/COUNCIL.md 7.2）。結論には使わず、記事では「観察」として付録に置く。

主な比較（M vs S、M vs M-avg、コスト）は、本体と同じ src/scorer.py の採点で行う（条件 M・M-avg・S として集計される）。
ここで扱うのは、runs/council.jsonl に残した途中の案（Round 0・Round 1・S の各回）から計算する値。

使い方:
    uv run python -m src.council_metrics                                  # 正解なしで計算できるものだけ
    uv run python -m src.council_metrics --actual data/actual/2026.json   # 変化の当否も計算する
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from src.actual import ActualDraft, load_actual
from src.candidates import load_candidates
from src.council import ROLES, TurnLog
from src.matching import Master, MatchStatus, normalize_name
from src.schema import PredictionOutput, TeamCode
from src.scorer import top_candidate


def top_names(pred: PredictionOutput) -> dict[TeamCode, str]:
    """球団ごとの1位入札の最有力候補（正規化した名前）。"""
    return {b.team: normalize_name(top_candidate(b).player_name) for b in pred.first_round}


def opinion_changes(round0: dict[str, PredictionOutput], round1: dict[str, PredictionOutput]) -> list[tuple[str, TeamCode]]:
    """Round 0 → Round 1 で、1位入札の最有力候補が変わった (役割, 球団)。"""
    changed = []
    for role in round0.keys() & round1.keys():
        before, after = top_names(round0[role]), top_names(round1[role])
        changed += [(role, t) for t in TeamCode if before[t] != after[t]]
    return sorted(changed)


def opinion_change_rate(round0: dict[str, PredictionOutput], round1: dict[str, PredictionOutput]) -> float:
    roles = round0.keys() & round1.keys()
    return len(opinion_changes(round0, round1)) / (len(roles) * len(TeamCode)) if roles else 0.0


def change_correctness(
    round0: dict[str, PredictionOutput], round1: dict[str, PredictionOutput], actual: ActualDraft
) -> dict[str, int]:
    """意見が変わったもののうち、正解に近づいた（改善）・遠ざかった（悪化）・どちらでもない の件数（採点後）。"""
    truth = {f.team: normalize_name(f.bid.player_name) for f in actual.first_round}
    counts = {"improved": 0, "worsened": 0, "neutral": 0}
    for role, team in opinion_changes(round0, round1):
        before, after = top_names(round0[role])[team], top_names(round1[role])[team]
        key = "improved" if after == truth[team] else "worsened" if before == truth[team] else "neutral"
        counts[key] += 1
    return counts


def agreement(a: PredictionOutput, b: PredictionOutput) -> float:
    """12球団のうち、1位入札の最有力候補が一致した割合。"""
    ta, tb = top_names(a), top_names(b)
    return sum(ta[t] == tb[t] for t in TeamCode) / len(TeamCode)


def order_sensitivity(finals: list[PredictionOutput]) -> float:
    """発言順の違う run の最終予測で、12球団のうち最有力候補が全 run で一致した割合（1.0 なら発言順に影響されていない）。"""
    if len(finals) < 2:
        return 1.0
    tops = [top_names(p) for p in finals]
    return sum(len({t[team] for t in tops}) == 1 for team in TeamCode) / len(TeamCode)


def unique_candidates(preds: list[PredictionOutput]) -> int:
    """1位入札の候補に出てきた選手のユニーク数（全球団・全案の合計）。"""
    return len({normalize_name(c.player_name) for p in preds for b in p.first_round for c in b.candidates})


def unmatched_names(pred: PredictionOutput, master: Master) -> set[str]:
    refs = [c for b in pred.first_round for c in b.candidates] + list(pred.contested) + list(pred.lotte)
    return {normalize_name(r.player_name) for r in refs if master.match(r.player_name, r.affiliation).status == MatchStatus.UNMATCHED}


def name_corrections(round0: list[PredictionOutput], final: PredictionOutput, master: Master) -> int:
    """Round 0 に出たマスタにない名前のうち、最終予測から消えたものの数。"""
    before = set().union(*(unmatched_names(p, master) for p in round0)) if round0 else set()
    return len(before - unmatched_names(final, master))


def load_turns(path: Path) -> list[TurnLog]:
    return [TurnLog.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def observe(turns: list[TurnLog], master: Master, actual: ActualDraft | None = None) -> dict:
    """council.jsonl から観察をまとめる。run ごとの値と、M の run をまたいだ発言順感度。"""
    by_run: dict[str, list[TurnLog]] = defaultdict(list)
    for t in turns:
        by_run[t.run_id].append(t)

    def pred(t: TurnLog) -> PredictionOutput | None:
        return PredictionOutput.model_validate(t.prediction_json) if t.prediction_json is not None else None

    runs, finals = [], []
    for run_id, ts in by_run.items():
        condition, run_no = ts[0].condition, ts[0].run_no
        row: dict = {"run_id": run_id, "condition": condition, "run_no": run_no}
        if condition == "M":
            r0 = {t.role: p for t in ts if t.round == "round0" and (p := pred(t))}
            r1_turns = sorted((t for t in ts if t.round == "round1" and pred(t)), key=lambda t: t.speaker_index or 0)
            r1 = {t.role: p for t in r1_turns if (p := pred(t))}
            final = next((p for t in ts if t.round == "final" and (p := pred(t))), None)
            row["opinion_change_rate"] = round(opinion_change_rate(r0, r1), 4)
            row["unique_candidates_round0"] = unique_candidates(list(r0.values()))
            if actual is not None and r0 and r1:
                row["change_correctness"] = change_correctness(r0, r1, actual)
            if final is not None:
                finals.append(final)
                row["name_corrections"] = name_corrections(list(r0.values()), final, master)
                if r1_turns:
                    first = r1_turns[0]
                    row["first_speaker"] = first.role
                    row["first_speaker_agreement"] = round(agreement(final, r1[first.role]), 4)
                    others = [r for r in ROLES if r != first.role and r in r1]
                    row["other_speakers_agreement"] = round(sum(agreement(final, r1[r]) for r in others) / len(others), 4) if others else None
        elif condition == "S":
            samples = [p for t in ts if t.round == "sample" and (p := pred(t))]
            row["k"] = len(samples)
            row["unique_candidates_samples"] = unique_candidates(samples)
            summary = next((t for t in ts if t.round == "summary"), None)
            if summary is not None:
                row["budget"] = summary.content
        runs.append(row)
    return {"runs": runs, "m_order_sensitivity": round(order_sensitivity(finals), 4) if finals else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--council-log", type=Path, default=Path("runs/council.jsonl"))
    ap.add_argument("--candidates", type=Path, default=Path("data/candidates.csv"))
    ap.add_argument("--actual", type=Path, help="実際の結果（あれば変化の当否も計算する）")
    ap.add_argument("--out", type=Path, default=Path("results/council_observations.json"))
    args = ap.parse_args(argv)

    if not args.council_log.exists():
        print(f"{args.council_log} がありません", file=sys.stderr)
        return 1
    master = Master(load_candidates(args.candidates))
    actual = load_actual(args.actual) if args.actual else None
    report = observe(load_turns(args.council_log), master, actual)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
