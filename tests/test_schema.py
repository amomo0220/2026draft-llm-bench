import copy
from typing import Any

import pytest
from pydantic import ValidationError

from src.schema import PROB_SUM_TOLERANCE, FirstRoundBid, Prediction, PredictionOutput, TeamCode


def _valid_payload() -> dict[str, Any]:
    """検証を通る最小の LLM 出力（ダミーの選手名）。"""
    return {
        "first_round": [
            {
                "team": team.value,
                "candidates": [
                    {"player_name": f"選手{team.value}A", "prob": 0.6},
                    {"player_name": f"選手{team.value}B", "prob": 0.4},
                ],
            }
            for team in TeamCode
        ],
        "contested": [{"player_name": "選手GA", "num_teams": 3}],
        "lotte": [
            {
                "round": r,
                "player_name": f"ロッテ{r}位",
                "reason": "ダミー",
                "sources": ["https://example.com/a"],
            }
            for r in range(1, 6)
        ],
    }


def _expect_invalid(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PredictionOutput.model_validate(payload)


# --- 正常系 ---


def test_valid_prediction() -> None:
    pred = PredictionOutput.model_validate(_valid_payload())
    assert len(pred.first_round) == 12
    assert {b.team for b in pred.first_round} == set(TeamCode)
    assert [p.round for p in pred.lotte] == [1, 2, 3, 4, 5]


def test_json_roundtrip() -> None:
    pred = PredictionOutput.model_validate(_valid_payload())
    assert PredictionOutput.model_validate_json(pred.model_dump_json()) == pred


def test_contested_and_sources_are_optional() -> None:
    payload = _valid_payload()
    del payload["contested"]
    for pick in payload["lotte"]:
        del pick["sources"]
    pred = PredictionOutput.model_validate(payload)
    assert pred.contested == []
    assert all(p.sources == [] for p in pred.lotte)


def test_lotte_order_does_not_matter() -> None:
    payload = _valid_payload()
    payload["lotte"].reverse()
    PredictionOutput.model_validate(payload)


def test_prob_sum_within_tolerance() -> None:
    probs = [0.333, 0.333, 0.333]  # 合計 0.999
    FirstRoundBid.model_validate(
        {"team": "M", "candidates": [{"player_name": f"x{i}", "prob": p} for i, p in enumerate(probs)]}
    )


# --- 確率 ---


@pytest.mark.parametrize("probs", [[0.5, 0.3], [0.7, 0.7], [0.5, 0.5 + 2 * PROB_SUM_TOLERANCE]])
def test_prob_sum_must_be_one(probs: list[float]) -> None:
    with pytest.raises(ValidationError, match="確率の合計"):
        FirstRoundBid.model_validate(
            {"team": "G", "candidates": [{"player_name": f"x{i}", "prob": p} for i, p in enumerate(probs)]}
        )


@pytest.mark.parametrize("bad_prob", [-0.1, 1.5, float("nan"), float("inf")])
def test_prob_out_of_range(bad_prob: float) -> None:
    payload = _valid_payload()
    payload["first_round"][0]["candidates"] = [{"player_name": "x", "prob": bad_prob}]
    _expect_invalid(payload)


def test_empty_candidates() -> None:
    payload = _valid_payload()
    payload["first_round"][0]["candidates"] = []
    _expect_invalid(payload)


def test_duplicate_candidate_in_team() -> None:
    payload = _valid_payload()
    payload["first_round"][0]["candidates"] = [
        {"player_name": "山田 太郎", "prob": 0.5},
        {"player_name": "山田太郎", "prob": 0.5},
    ]
    _expect_invalid(payload)


# --- 12球団 ---


def test_unknown_team_code() -> None:
    payload = _valid_payload()
    payload["first_round"][0]["team"] = "XX"
    _expect_invalid(payload)


def test_missing_team() -> None:
    payload = _valid_payload()
    payload["first_round"].pop()
    with pytest.raises(ValidationError, match="12球団"):
        PredictionOutput.model_validate(payload)


def test_duplicate_team() -> None:
    payload = _valid_payload()
    payload["first_round"][-1] = copy.deepcopy(payload["first_round"][0])
    with pytest.raises(ValidationError, match="12球団"):
        PredictionOutput.model_validate(payload)


def test_extra_team() -> None:
    payload = _valid_payload()
    payload["first_round"].append(copy.deepcopy(payload["first_round"][0]))
    _expect_invalid(payload)


# --- ロッテ1〜5位 ---


@pytest.mark.parametrize("n", [4, 6])
def test_lotte_must_have_five_picks(n: int) -> None:
    payload = _valid_payload()
    base = payload["lotte"][0]
    payload["lotte"] = [{**base, "round": r, "player_name": f"p{r}"} for r in range(1, n + 1)]
    _expect_invalid(payload)


@pytest.mark.parametrize("bad_round", [0, 6])
def test_lotte_round_out_of_range(bad_round: int) -> None:
    payload = _valid_payload()
    payload["lotte"][0]["round"] = bad_round
    _expect_invalid(payload)


def test_lotte_duplicate_round() -> None:
    payload = _valid_payload()
    payload["lotte"][1]["round"] = 1
    with pytest.raises(ValidationError, match="1〜5位"):
        PredictionOutput.model_validate(payload)


def test_lotte_duplicate_player() -> None:
    payload = _valid_payload()
    payload["lotte"][1]["player_name"] = payload["lotte"][0]["player_name"]
    with pytest.raises(ValidationError, match="同じ選手"):
        PredictionOutput.model_validate(payload)


def test_lotte_invalid_source_url() -> None:
    payload = _valid_payload()
    payload["lotte"][0]["sources"] = ["not a url"]
    _expect_invalid(payload)


# --- その他の不正出力 ---


def test_player_id_rejected_in_llm_output() -> None:
    """player_id はコードで突合するため、LLM が出力したら拒否する。"""
    for mutate in (
        lambda p: p["first_round"][0]["candidates"][0].update(player_id="P-1"),
        lambda p: p["lotte"][0].update(player_id="P-1"),
        lambda p: p["contested"][0].update(player_id="P-1"),
    ):
        payload = _valid_payload()
        mutate(payload)
        _expect_invalid(payload)


def test_fullwidth_space_duplicate_in_lotte() -> None:
    payload = _valid_payload()
    payload["lotte"][0]["player_name"] = "山田\u3000太郎"
    payload["lotte"][1]["player_name"] = "山田太郎"
    with pytest.raises(ValidationError, match="同じ選手"):
        PredictionOutput.model_validate(payload)


def test_extra_field_rejected() -> None:
    payload = _valid_payload()
    payload["comment"] = "自由文"
    _expect_invalid(payload)


def _valid_prediction() -> dict[str, Any]:
    return {"run_id": "run-0001", "model_id": "dummy-model", "condition": "B", "output": _valid_payload()}


def test_prediction_wraps_output() -> None:
    pred = Prediction.model_validate(_valid_prediction())
    assert pred.condition == "B"
    assert len(pred.output.first_round) == 12


@pytest.mark.parametrize("field", ["run_id", "model_id", "condition"])
def test_blank_metadata_rejected(field: str) -> None:
    payload = _valid_prediction()
    payload[field] = "  "
    with pytest.raises(ValidationError):
        Prediction.model_validate(payload)


@pytest.mark.parametrize("field", ["run_id", "model_id", "condition"])
def test_llm_output_cannot_contain_metadata(field: str) -> None:
    payload = _valid_payload()
    payload[field] = "x"
    _expect_invalid(payload)


def test_invalid_output_rejected_in_prediction() -> None:
    payload = _valid_prediction()
    payload["output"]["lotte"].pop()
    with pytest.raises(ValidationError):
        Prediction.model_validate(payload)


def test_empty_player_name_rejected() -> None:
    payload = _valid_payload()
    payload["lotte"][0]["player_name"] = ""
    _expect_invalid(payload)


@pytest.mark.parametrize("num_teams", [1, 13])
def test_contest_num_teams_range(num_teams: int) -> None:
    payload = _valid_payload()
    payload["contested"][0]["num_teams"] = num_teams
    _expect_invalid(payload)
