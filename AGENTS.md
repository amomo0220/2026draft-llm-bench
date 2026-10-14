# AGENTS.md

このリポジトリで作業するAIコーディングエージェント共通の指示書です。
（Codex は直接読み込み、Claude Code は `CLAUDE.md` の `@AGENTS.md` 経由、Gemini CLI は `contextFileName` の設定で読み込む）
詳細な企画・実験設計は [`docs/PLAN.md`](docs/PLAN.md) にあります。新しいタスクを始める前に必ず読むこと。

## プロジェクトの目的

2026年プロ野球ドラフト会議（2026-10-22）を題材にした、**LLM／AIエージェントのライブ予測ベンチマーク**。

- 各球団の1位入札、競合の有無、千葉ロッテマリーンズの1〜5位指名を、
  「モデル × 情報条件 × ハーネス」の組み合わせで予測させる
- 予測はドラフト前に **事前登録（GitHub タグで固定）** し、ドラフト後に **自動採点** する
- 問いは「どのAIが賢いか」ではなく、
  **検索・コンテキスト・ハーネスで予測がどう変わり、どこでハルシネーションが起き、ベースラインにどれだけ勝てるか**

## 絶対に守るルール

1. **事前登録の固定性**
   - `predictions/` 配下のファイルは、事前登録タグ（`prereg-2026-10-15`）を打った後は **絶対に変更・削除しない**
   - 締切後に作った予測は、別ディレクトリ（`predictions/late/`）に、条件名に `late` を付けて保存する
2. **出力は構造化する**
   - LLMの出力は必ず Pydantic スキーマで検証する。自由文の予測を採点対象にしない
   - スキーマを変えるときは `src/schema.py` とテストを同時に更新する
3. **実験条件は再現できるようにする**
   - 1回の実行ごとに、モデルID・条件名・seed／実行番号・プロンプトのハッシュ・日時・トークン数・コスト・所要時間・ツール呼び出し回数を `runs/*.jsonl` に記録する
   - プロンプトは `prompts/` にファイルとして置き、コード内に直書きしない
4. **データの扱い**
   - 新聞・Web記事の本文や、スクレイピングで取得した生データは **コミットしない**（`data/raw/` は `.gitignore` 済み）
   - コミットしてよいのは、自分で構造化したデータ（選手名・所属・ポジション・出典URL など）とスクリプトのみ
   - 各データ行には出典URLを持たせる
   - 一次情報（公式発表・公式名簿など）を優先する。まとめ記事などの二次情報は、一次情報が存在しない項目にだけ使う
5. **選手への配慮**
   - 候補には高校生（未成年）が含まれる。選手個人を貶めるような評価文を生成・公開しない
   - 記事やREADMEの主役は「システムと検証結果」であって、選手の品評ではない
6. **秘密情報**
   - APIキーは `.env` で管理し、コミットしない（`.env.example` のみコミット）
   - ログやトレースにAPIキーやトークンが混ざらないよう注意する
7. **予算**
   - 1回の実験バッチを流す前に、見積もりコストを表示し、`BUDGET_USD` を超える場合は停止する

## ディレクトリ規約

```
.
├─ AGENTS.md              # このファイル
├─ CLAUDE.md              # "@AGENTS.md" の1行のみ
├─ docs/PLAN.md           # 企画書・実験設計
├─ data/
│  ├─ candidates.csv      # 候補者マスタ（正解判定の基準）
│  ├─ teams.csv           # 12球団の補強ポイント等
│  ├─ baselines/          # メディア予想の多数決など（集計値のみ）
│  ├─ actual/             # ドラフト当日の実際の結果
│  └─ raw/                # 取得した生データ（gitignore）
├─ prompts/               # 条件ごとのプロンプト
├─ predictions/           # 事前登録する予測（タグ後は不変）
├─ runs/                  # 実行ログ（JSONL）
├─ results/               # 採点結果・図表
├─ src/
│  ├─ schema.py           # Pydantic スキーマ
│  ├─ runners/            # openai / anthropic / gemini / cli（claude, codex, gemini）
│  ├─ draft_db_mcp/       # 候補者マスタを公開する MCP サーバー
│  ├─ baselines.py
│  └─ scorer.py
└─ tests/
```

## 開発コマンド（初期案。実装に合わせて更新すること）

```bash
uv sync                                  # 依存関係のインストール
uv run pytest                            # テスト
uv run python -m src.run --condition B --model <model-id> --n 3   # 実験の実行
uv run python -m src.scorer --actual data/actual/2026.json        # 採点
uv run python -m src.actual data/actual/2026.json  # 実際の結果の入力チェック
uv run python -m src.build_candidates    # 候補者マスタの生成（data/raw/ の公式一覧から）
uv run python -m src.candidates          # 候補者マスタの検証
uv run python -m src.draft_db_mcp        # MCP サーバー起動
uv run python -m src.prereg_guard        # 事前登録した predictions/ が変わっていないか（CI でも実行）
uv run python -m src.council --predictions <保存先> --dry-run   # 予想会議（追加実験。docs/COUNCIL.md）
uv run python -m src.council_metrics     # 予想会議の観察の計算
```

## コーディング規約

- Python 3.12+、パッケージ管理は `uv`
- 型ヒント必須。I/O境界（LLM出力・CSV読み込み）は必ず Pydantic で検証する
- 抽選・指名済み除外・ウェーバー順などの **ルールはコードで決定的に処理** し、LLMには判断だけをさせる
- 乱数を使う処理は seed を引数で受け取る
- 採点ロジック（`src/scorer.py`）には必ずユニットテストを付ける。ダミーの正解データで検証する
- フレームワーク（LangGraph 等）は導入しない。必要になったら PLAN.md に理由を書いてから

## 作業の進め方

- 1タスク＝1ブランチ＝1PRを基本とする
- 実験条件・評価指標を変えるときは、先に `docs/PLAN.md` を更新する
- 完了の定義: テストが通る / `runs/` にログが残る / README か PLAN に変更点を反映済み
