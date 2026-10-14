"""Anthropic Messages API（条件 A・B 用。検索なし）。

APIキーは BENCH_ANTHROPIC_API_KEY から読む。ANTHROPIC_API_KEY はクラウド環境で Claude Code 自身の認証に
予約されていて使えないため。接続先も環境変数（ANTHROPIC_BASE_URL）に左右されないよう明示する。
"""

import os

import anthropic

from src.runners.base import RunnerResult

API_KEY_ENV = "BENCH_ANTHROPIC_API_KEY"
BASE_URL = "https://api.anthropic.com"


def call(model_id: str, system: str, user: str, max_output_tokens: int) -> RunnerResult:
    client = anthropic.Anthropic(api_key=os.environ[API_KEY_ENV], base_url=BASE_URL)
    # 出力が長いので、SDK のタイムアウトを避けるためストリーミングで受け取る。thinking・effort は既定のまま
    with client.messages.stream(
        model=model_id,
        max_tokens=max_output_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        msg = stream.get_final_message()
    text = "".join(b.text for b in msg.content if b.type == "text")
    return RunnerResult(
        text=text,
        input_tokens=msg.usage.input_tokens,
        output_tokens=msg.usage.output_tokens,
        stop_reason=msg.stop_reason,
    )
