import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.actual import ActualDraft, main
from src.schema import TeamCode
from tests.dummy_draft import actual_payload, player


def _invalid(payload: dict[str, Any], match: str | None = None) -> None:
    with pytest.raises(ValidationError, match=match):
        ActualDraft.model_validate(payload)


def _team(payload: dict[str, Any], team: TeamCode) -> dict[str, Any]:
    return next(r for r in payload["first_round"] if r["team"] == team.value)


def test_valid_actual() -> None:
    actual = ActualDraft.model_validate(actual_payload())
    assert actual.by_team()[TeamCode.T].final.player_name == "外れ 次郎"
    assert [p.player_name for p in actual.lotte_acquired()] == ["ロッテ 一位", "ロッテ 二位", "ロッテ 三位", "ロッテ 四位"]


def test_lotte_can_be_empty() -> None:
    payload = actual_payload()
    payload["lotte"] = []
    assert ActualDraft.model_validate(payload).lotte == []


def test_won_lottery_requires_same_final() -> None:
    payload = actual_payload()
    _team(payload, TeamCode.G)["final"] = player("別人")
    _invalid(payload, "won_lottery=true")


def test_lost_lottery_requires_different_final() -> None:
    payload = actual_payload()
    t = _team(payload, TeamCode.T)
    t["final"] = t["bid"]
    _invalid(payload, "抽選で外れた")


def test_missing_team() -> None:
    payload = actual_payload()
    payload["first_round"].pop()
    _invalid(payload, "12球団")


def test_duplicate_final() -> None:
    payload = actual_payload()
    t = _team(payload, TeamCode.T)
    t["final"] = player("選手H")  # H の最終的な1位と重複
    _invalid(payload, "最終的な1位")


@pytest.mark.parametrize("rounds", [[2, 2], [2, 4], [3]])
def test_lotte_rounds_must_be_contiguous_from_two(rounds: list[int]) -> None:
    payload = actual_payload()
    payload["lotte"] = [{**player(f"p{i}"), "round": r} for i, r in enumerate(rounds)]
    _invalid(payload)


@pytest.mark.parametrize("bad_round", [1, 6])
def test_lotte_round_out_of_range(bad_round: int) -> None:
    payload = actual_payload()
    payload["lotte"] = [{**player("p"), "round": bad_round}]
    _invalid(payload)


def test_unknown_position() -> None:
    payload = actual_payload()
    payload["lotte"][0]["position"] = "遊撃手"
    _invalid(payload)


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "2026.json"
    path.write_text(json.dumps(actual_payload(), ensure_ascii=False), encoding="utf-8")
    assert main(["actual", str(path)]) == 0
    assert "抽選で外れた球団: T" in capsys.readouterr().out
    path.write_text("{}", encoding="utf-8")
    assert main(["actual", str(path)]) == 1
