"""採点テスト用のダミーの正解データと予測（実在の結果ではない）。

- G と T が「競合 一郎」に入札し、G が交渉権、T は外れ1位で「外れ 次郎」
- ロッテ（M）は「ロッテ 一位」を単独入札。2〜4位まで指名（5位は指名なし）
- それ以外の球団は「選手X」（X は球団コード）を単独入札
"""

from typing import Any

from src.schema import TeamCode


def player(name: str, affiliation: str = "ダミー高", position: str = "投手") -> dict[str, str]:
    return {"player_name": name, "affiliation": affiliation, "position": position}


def bid_name(team: TeamCode) -> str:
    if team in (TeamCode.G, TeamCode.T):
        return "競合 一郎"
    if team == TeamCode.M:
        return "ロッテ 一位"
    return f"選手{team.value}"


def actual_payload() -> dict[str, Any]:
    first_round = []
    for team in TeamCode:
        bid = player(bid_name(team))
        won = team != TeamCode.T
        final = bid if won else player("外れ 次郎", position="内野手")
        first_round.append({"team": team.value, "bid": bid, "won_lottery": won, "final": final})
    lotte = [
        {**player("ロッテ 二位", "ダミー大学", "捕手"), "round": 2},
        {**player("ロッテ 三位", "ダミー製作所", "内野手"), "round": 3},
        {**player("ロッテ 四位", "ダミー高", "外野手"), "round": 4},
    ]
    return {"first_round": first_round, "lotte": lotte}


def prediction_payload(run_id: str = "run-1", model_id: str = "dummy", condition: str = "B") -> dict[str, Any]:
    """全球団の最尤候補が実際の入札と一致し、ロッテは1〜4位が一致する予測。"""
    first_round = [
        {
            "team": team.value,
            "candidates": [
                {**player(bid_name(team)), "prob": 0.6},
                {**player(f"次点{team.value}"), "prob": 0.4},
            ],
        }
        for team in TeamCode
    ]
    names = ["ロッテ 一位", "ロッテ 二位", "ロッテ 三位", "ロッテ 四位", "ロッテ 五位"]
    positions = ["投手", "捕手", "内野手", "外野手", "投手"]
    lotte = [
        {**player(n, position=p), "round": i + 1, "reason": "ダミー"} for i, (n, p) in enumerate(zip(names, positions))
    ]
    return {
        "run_id": run_id,
        "model_id": model_id,
        "condition": condition,
        "output": {"first_round": first_round, "contested": [], "lotte": lotte},
    }
