import pytest

from src.aggregate import aggregate
from src.schema import TeamCode
from tests.council_fixtures import output


def _bid(pred, team: TeamCode):
    return next(b for b in pred.first_round if b.team == team)


def test_average_probabilities_over_all_proposals() -> None:
    a = output({"G": "甲"})  # G: 甲 0.6 / 次点G 0.4
    b = output({"G": "乙"})  # G: 乙 0.6 / 次点G 0.4
    agg = aggregate([a, b])
    probs = {c.player_name: c.prob for c in _bid(agg, TeamCode.G).candidates}
    assert probs == pytest.approx({"次点G": 0.4, "甲": 0.3, "乙": 0.3})
    assert _bid(agg, TeamCode.G).candidates[0].player_name == "次点G"  # 確率の高い順
    assert sum(probs.values()) == pytest.approx(1.0)


def test_names_are_merged_after_normalization() -> None:
    a = output({"G": "競合 一郎"})
    b = output({"G": "競合　一郎"})  # 全角空白
    names = [c.player_name for c in _bid(aggregate([a, b]), TeamCode.G).candidates]
    assert names == ["競合 一郎", "次点G"]


def test_contested_rebuilt_from_averaged_tops() -> None:
    a = output({"G": "人気", "T": "人気", "DB": "人気"})
    b = output({"G": "人気", "T": "人気"})
    agg = aggregate([a, b])
    # DB は 人気 0.3 / 選手DB 0.3 / 次点DB 0.4 → 最有力は 次点DB。G・T の2球団が重なる
    assert [(c.player_name, c.num_teams) for c in agg.contested] == [("人気", 2)]


def test_lotte_majority_vote_per_round() -> None:
    names = ["A", "B", "C", "D", "E"]
    preds = [output(lotte=names), output(lotte=names), output(lotte=["X", "B", "C", "D", "E"])]
    agg = aggregate(preds)
    assert [p.player_name for p in agg.lotte] == names
    assert agg.lotte[0].reason == "3案の集約（この順位で 2 票）"


def test_lotte_tie_broken_by_first_round_probability() -> None:
    # 1位の票は X と Y が1票ずつ。ロッテの1位入札の確率は Y の方が高い
    a = output({"M": "X"}, lotte=["X", "B", "C", "D", "E"], top_prob=0.55)
    b = output({"M": "Y"}, lotte=["Y", "B", "C", "D", "E"], top_prob=0.9)
    assert aggregate([a, b]).lotte[0].player_name == "Y"


def test_lotte_tie_without_probability_uses_first_appearance() -> None:
    a = output(lotte=["X", "B", "C", "D", "E"])
    b = output(lotte=["Y", "B", "C", "D", "E"])
    assert aggregate([a, b]).lotte[0].player_name == "X"


def test_lotte_never_repeats_a_player() -> None:
    # 2位の多数決は A（1位で使用済み）→ 次点の B を使う。最後は全順位の票から残りを選ぶ
    preds = [
        output(lotte=["A", "B", "C", "D", "E"]),
        output(lotte=["F", "A", "C", "D", "E"]),
        output(lotte=["A", "A2", "C", "D", "E"]),
    ]
    picks = [p.player_name for p in aggregate(preds).lotte]
    assert picks[0] == "A" and len(set(picks)) == 5


def test_aggregate_requires_predictions() -> None:
    with pytest.raises(ValueError):
        aggregate([])
