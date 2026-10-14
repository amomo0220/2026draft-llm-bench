"""実験に使うモデルと料金（docs/PLAN.md 3.2）。

モデルIDは 2026-10-14 に固定した。料金は二次情報で確認したもの。予算を確定する前に各社の公式料金ページで確かめ、
変わっていたらここを更新する（料金は見積もりとコスト記録にだけ使い、予測には影響しない）。
"""

from dataclasses import dataclass
from enum import StrEnum


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    model_id: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float  # 推論（thinking）トークンを含む出力の単価
    search_usd_per_call: float  # Web 検索1回あたり（条件 C）。要確認: 公式料金ページで未確認の仮の値（高めに置いている）

    def cost_usd(self, input_tokens: int, output_tokens: int, searches: int = 0) -> float:
        tokens = input_tokens * self.input_usd_per_mtok + output_tokens * self.output_usd_per_mtok
        return tokens / 1_000_000 + searches * self.search_usd_per_call


MODELS: dict[str, ModelSpec] = {
    m.model_id: m
    for m in [
        ModelSpec(Provider.ANTHROPIC, "claude-sonnet-5-5", 2.00, 10.00, 0.01),
        ModelSpec(Provider.OPENAI, "gpt-5.6-terra", 2.00, 12.00, 0.025),
        ModelSpec(Provider.GEMINI, "gemini-3.8-flash", 0.75, 3.75, 0.035),  # トークン単価は2026年末までの導入価格
    ]
}
