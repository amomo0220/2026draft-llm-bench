"""実験に使うモデルと料金（docs/PLAN.md 3.2）。

モデルIDは 2026-10-14 に固定した。料金は見積もりとコスト記録にだけ使い、予測には影響しない。

料金の確認（2026-10-14）:
- Anthropic: 公式（https://platform.claude.com/docs/en/about-claude/pricing）で確認。Sonnet 5.5 は $2 / $10、
  Web 検索は $10 / 1,000回（検索結果は入力トークンとして別に課金）。動的フィルタリングのコード実行は無料
- Google: Vertex AI の公式（https://cloud.google.com/vertex-ai/generative-ai/pricing）で確認。Gemini 3.8 Flash は
  2026-12-31 まで $0.75 / $3.75。Google 検索のグラウンディングは検索語1つごとに $14 / 1,000（Gemini 3 以降の合計で
  月5,000件まで無料。無料枠は見積もりに入れない）。Gemini API（ai.google.dev）の料金ページはこの環境から開けず、
  検索結果の抜粋で同じ料金であることだけ確かめた
- OpenAI: 公式の料金ページ（https://platform.openai.com/docs/pricing）をユーザーが保存した HTML で確認。
  Web 検索は全モデル $10 / 1,000回（検索結果のトークンはモデルの単価で別に課金）。gpt-5.6-terra のトークン単価は
  このページに載っておらず未確認（$2 / $12 は二次情報）
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
    search_usd_per_call: float  # Web 検索1回あたり（条件 C・D）。Gemini は検索語1つあたり

    def cost_usd(self, input_tokens: int, output_tokens: int, searches: int = 0) -> float:
        tokens = input_tokens * self.input_usd_per_mtok + output_tokens * self.output_usd_per_mtok
        return tokens / 1_000_000 + searches * self.search_usd_per_call


MODELS: dict[str, ModelSpec] = {
    m.model_id: m
    for m in [
        ModelSpec(Provider.ANTHROPIC, "claude-sonnet-5-5", 2.00, 10.00, 0.01),
        ModelSpec(Provider.OPENAI, "gpt-5.6-terra", 2.00, 12.00, 0.01),  # トークン単価は要確認（モジュールの説明）
        ModelSpec(Provider.GEMINI, "gemini-3.8-flash", 0.75, 3.75, 0.014),  # トークン単価は2026年末までの導入価格
    ]
}
