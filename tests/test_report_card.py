"""結論カード（HTML の組み立てと数字の計算。画像の出力はブラウザが要るので確かめない）。"""

import copy

from src.actual import ActualDraft
from src.matching import Master
from src.report_card import card_html, hit_means
from src.schema import Prediction
from tests.dummy_draft import actual_payload, prediction_payload

ACTUAL = ActualDraft.model_validate(actual_payload())


def _pred(model: str, cond: str, miss: int) -> Prediction:
    """最初の miss 球団の本命を外した予想（ダミーは全球団的中が基本）。"""
    payload = copy.deepcopy(prediction_payload(f"{cond}-{model}", model, cond))
    for b in payload["output"]["first_round"][:miss]:
        b["candidates"][0]["prob"], b["candidates"][1]["prob"] = 0.4, 0.6
    return Prediction.model_validate(payload)


def test_hit_means_and_card() -> None:
    preds = [_pred("m1", "B", 11), _pred("m1", "D", 9), _pred("m2", "B", 11), _pred("m2", "D", 10)]
    means = hit_means(preds, ACTUAL, Master([]), {})
    assert means["m1"] == {"B": 1.0, "D": 3.0} and means["m2"] == {"B": 1.0, "D": 2.0}
    page = card_html(means, media=3)
    assert "3<small>倍</small>" in page  # m1 の B → D
    assert "1.0 → 3.0" in page and "最大 1.0 球団" in page  # 条件D での AI どうしの差
    assert "メディア予想の多数決：3" in page
