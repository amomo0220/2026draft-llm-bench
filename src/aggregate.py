"""複数の予測を機械的に1つにまとめる（docs/COUNCIL.md 3.4）。LLM は呼ばない。

予想会議の M-avg（Round 1 の3案）と、同予算シングル S（k 回の予測）で同じ関数を使う。

- 1位入札: 球団ごとに、選手ごとの確率を全案で平均する（案に出てこない選手は確率 0 として数える）
- 競合予測: 平均後の1位入札で、最有力候補が重なる球団を数えて作り直す
- ロッテ1〜5位: 順位ごとの多数決。同数なら、その選手のロッテ1位入札の平均確率が高い方、
  それも同じなら案の並び順で先に出てきた方。上の順位で使った選手は除く
"""

from collections import Counter

from src.matching import normalize_name
from src.schema import (
    LOTTE_ROUNDS,
    ContestPrediction,
    FirstRoundBid,
    LottePick,
    PlayerProb,
    PredictionOutput,
    TeamCode,
    _PlayerRef,
)
from src.scorer import top_candidate


def _ref(r: _PlayerRef) -> dict[str, str]:
    return {"player_name": r.player_name, "affiliation": r.affiliation, "position": r.position.value}


def average_first_round(preds: list[PredictionOutput]) -> list[FirstRoundBid]:
    bids = []
    for team in TeamCode:
        probs: dict[str, float] = {}
        refs: dict[str, _PlayerRef] = {}  # 選手の所属・ポジションは、最初に出てきた案のものを使う
        for p in preds:
            bid = next(b for b in p.first_round if b.team == team)
            total = sum(c.prob for c in bid.candidates)  # 許容誤差の範囲で 1.0 からずれた分を直す
            for c in bid.candidates:
                key = normalize_name(c.player_name)
                refs.setdefault(key, c)
                probs[key] = probs.get(key, 0.0) + c.prob / total / len(preds)
        ranked = sorted(probs, key=lambda k: -probs[k])  # 同率なら先に出てきた選手（sorted は安定）
        bids.append(FirstRoundBid(team=team, candidates=[PlayerProb(**_ref(refs[k]), prob=probs[k]) for k in ranked]))
    return bids


def contested_from(first_round: list[FirstRoundBid]) -> list[ContestPrediction]:
    tops = [top_candidate(b) for b in first_round]
    counts = Counter(normalize_name(t.player_name) for t in tops)
    out, seen = [], set()
    for t in tops:
        key = normalize_name(t.player_name)
        if counts[key] >= 2 and key not in seen:
            seen.add(key)
            out.append(ContestPrediction(**_ref(t), num_teams=counts[key]))
    return out


def vote_lotte(preds: list[PredictionOutput], first_round: list[FirstRoundBid]) -> list[LottePick]:
    lotte_bid = next(b for b in first_round if b.team == TeamCode.M)
    bid_prob = {normalize_name(c.player_name): c.prob for c in lotte_bid.candidates}
    order: dict[str, int] = {}  # 案の並び順（案ごと・順位ごとに、最初に出てきた順）
    refs: dict[str, _PlayerRef] = {}
    for p in preds:
        for pick in sorted(p.lotte, key=lambda x: x.round):
            key = normalize_name(pick.player_name)
            order.setdefault(key, len(order))
            refs.setdefault(key, pick)

    def rank(votes: Counter[str]) -> list[str]:
        return sorted(votes, key=lambda k: (-votes[k], -bid_prob.get(k, 0.0), order[k]))

    used: set[str] = set()
    picks = []
    for rnd in range(1, LOTTE_ROUNDS + 1):
        votes = Counter(normalize_name(x.player_name) for p in preds for x in p.lotte if x.round == rnd)
        chosen = next((k for k in rank(votes) if k not in used), None)
        if chosen is None:  # その順位の候補がすべて上で使われた: 全順位の票から選ぶ（各案は5人なので必ず残る）
            all_votes = Counter(normalize_name(x.player_name) for p in preds for x in p.lotte)
            chosen = next(k for k in rank(all_votes) if k not in used)
        used.add(chosen)
        picks.append(
            LottePick(
                **_ref(refs[chosen]),
                round=rnd,
                reason=f"{len(preds)}案の集約（この順位で {votes[chosen]} 票）",
            )
        )
    return picks


def aggregate(preds: list[PredictionOutput]) -> PredictionOutput:
    if not preds:
        raise ValueError("集約する予測がありません")
    first_round = average_first_round(preds)
    return PredictionOutput(first_round=first_round, contested=contested_from(first_round), lotte=vote_lotte(preds, first_round))
