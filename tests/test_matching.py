import pytest

from src.candidates import Candidate, Category
from src.matching import Master, MatchStatus, normalize_name, same_affiliation, to_positions
from src.schema import Position


def _cand(pid: str, name: str, team: str, category: Category = Category.HIGH_SCHOOL) -> Candidate:
    return Candidate(player_id=pid, name=name, category=category, team=team, source_url="https://example.com/")


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("𠮷井 優都己", "吉井優都己"),  # 公式一覧の異体字
        ("髙橋 将大", "高橋将大"),
        ("岡﨑 來", "岡崎來"),
        ("藤澤 涼介", "藤沢涼介"),
        ("八木　月輝", "八木 月輝"),  # 全角空白
        ("ささき・だいすけ", "ささきだいすけ"),  # 中黒
        ("ＡＢＣ", "ABC"),  # 全角英字（NFKC）
    ],
)
def test_normalize_name_equal(a: str, b: str) -> None:
    assert normalize_name(a) == normalize_name(b)


def test_normalize_name_keeps_different_names() -> None:
    assert normalize_name("鈴木 悠人") != normalize_name("鈴木 悠悟")


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("横浜高", "横浜", True),
        ("横浜高校", "横浜", True),
        ("青山学院大", "青山学院大学", True),
        ("高岡第一高", "高岡第一", True),
        ("高岡第一", "未来富山", False),
        ("NTT東日本", "ＮＴＴ東日本", True),
    ],
)
def test_same_affiliation(a: str, b: str, same: bool) -> None:
    assert same_affiliation(a, b) is same


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("投手", {Position.PITCHER}),
        ("遊撃手", {Position.INFIELDER}),
        ("投手、内野手", {Position.PITCHER, Position.INFIELDER}),
        ("中堅手", {Position.OUTFIELDER}),
        ("", set()),
        (None, set()),
        ("指名打者", set()),
    ],
)
def test_to_positions(text: str | None, expected: set[Position]) -> None:
    assert to_positions(text) == expected


def test_master_match() -> None:
    master = Master([_cand("HS-001", "織田 翔希", "横浜"), _cand("HS-002", "𠮷井 優都己", "松陰")])
    assert master.match("織田翔希").candidate is not None
    assert master.match("吉井 優都己").status == MatchStatus.MATCHED
    assert master.match("架空 太郎").status == MatchStatus.UNMATCHED


def test_master_same_name_resolved_by_affiliation() -> None:
    master = Master(
        [
            _cand("HS-001", "田中 彰人", "鵬翔"),
            _cand("UN-001", "田中 彰人", "ダミー大学", Category.UNIVERSITY),
        ]
    )
    assert master.match("田中彰人").status == MatchStatus.AMBIGUOUS
    m = master.match("田中彰人", "ダミー大")
    assert m.status == MatchStatus.MATCHED and m.candidate is not None and m.candidate.player_id == "UN-001"
    # 所属でも絞れない場合は曖昧のまま
    assert master.match("田中彰人", "別の学校").status == MatchStatus.AMBIGUOUS
