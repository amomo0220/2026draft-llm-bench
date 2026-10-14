"""予想会議のテスト用の予測（実在の選手・結果ではない）。"""

import json
from typing import Any

from src.schema import PredictionOutput, TeamCode
from tests.dummy_draft import player, prediction_payload


def output(tops: dict[str, str] | None = None, lotte: list[str] | None = None, top_prob: float = 0.6) -> PredictionOutput:
    """dummy_draft の予測を元に、指定した球団の最有力候補とロッテ1〜5位を差し替える。"""
    payload: dict[str, Any] = prediction_payload()["output"]
    for bid in payload["first_round"]:
        name = (tops or {}).get(bid["team"])
        if name is not None:
            bid["candidates"][0]["player_name"] = name
        bid["candidates"][0]["prob"] = top_prob
        bid["candidates"][1]["prob"] = round(1 - top_prob, 6)
    if lotte is not None:
        payload["lotte"] = [{**player(n), "round": i + 1, "reason": "ダミー"} for i, n in enumerate(lotte)]
    return PredictionOutput.model_validate(payload)


def turn_json(pred: PredictionOutput, argument: str = "ダミーの発言") -> str:
    return json.dumps({"argument": argument, "prediction": pred.model_dump(mode="json")}, ensure_ascii=False)


ALL_TEAMS = [t.value for t in TeamCode]
