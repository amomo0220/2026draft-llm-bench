"""各社 API 呼び出しの共通の型。"""

from dataclasses import dataclass
from typing import Protocol

# 条件 C（Web 検索）で、1回の呼び出しあたりに許す検索回数の目安（制限できる API でのみ効く）
MAX_SEARCHES = 10


@dataclass(frozen=True)
class RunnerResult:
    text: str
    input_tokens: int
    output_tokens: int  # 推論（thinking）トークンを含む
    reasoning_tokens: int | None = None  # 内訳が取れる場合のみ
    tool_calls: int = 0  # Web 検索などのツール呼び出し回数
    stop_reason: str | None = None


class Runner(Protocol):
    def __call__(
        self, model_id: str, system: str, user: str, max_output_tokens: int, web_search: bool = False
    ) -> RunnerResult: ...
