"""各社 API 呼び出しの共通の型。"""

from dataclasses import dataclass
from typing import Any, Protocol

# 条件 C（Web 検索）で、1回の呼び出しあたりに許す検索回数の目安（制限できる API でのみ効く）
MAX_SEARCHES = 10
# 条件 D で、1回の呼び出しの中で関数呼び出し（draft-db のツール）を往復する上限
MAX_TOOL_TURNS = 20
# 上限を超えた関数呼び出しには、実行せずにこれを返し、もう1回だけ応答させる（本文を書かずに終わるのを防ぐ）
TOOL_LIMIT_RESULT = '{"error": "ツール呼び出しの上限に達しました。これ以上ツールは使えません。ここまでの情報で回答してください。"}'


class ToolSpecLike(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolBox(Protocol):
    """関数呼び出しで渡すツール（src/draft_db_mcp/tools.py の DraftDB）。"""

    @property
    def specs(self) -> list[ToolSpecLike]: ...

    def call(self, name: str, args: dict[str, Any]) -> str: ...


@dataclass(frozen=True)
class RunnerResult:
    text: str
    input_tokens: int
    output_tokens: int  # 推論（thinking）トークンを含む
    reasoning_tokens: int | None = None  # 内訳が取れる場合のみ
    tool_calls: int = 0  # Web 検索の回数（検索の料金の計算に使う）
    function_calls: int = 0  # 関数呼び出し（draft-db のツール）の回数
    max_request_input_tokens: int = 0  # 1回のリクエストの入力トークンの最大（長いコンテキストの割増の判定に使う）
    stop_reason: str | None = None


class Runner(Protocol):
    def __call__(
        self,
        model_id: str,
        system: str,
        user: str,
        max_output_tokens: int,
        web_search: bool = False,
        tools: ToolBox | None = None,
    ) -> RunnerResult: ...
