"""各社 API 呼び出しの共通の型。"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RunnerResult:
    text: str
    input_tokens: int
    output_tokens: int  # 推論（thinking）トークンを含む
    reasoning_tokens: int | None = None  # 内訳が取れる場合のみ
    tool_calls: int = 0
    stop_reason: str | None = None


class Runner(Protocol):
    def __call__(self, model_id: str, system: str, user: str, max_output_tokens: int) -> RunnerResult: ...
