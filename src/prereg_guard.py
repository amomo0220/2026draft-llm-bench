"""事前登録の保護（AGENTS.md ルール1、docs/PLAN.md 3.7）。

`prereg-*` タグがあるとき、タグの時点から `predictions/` の `late/` 以外が変わっていないかを確かめる。
変更・削除・追加のどれも違反とする（締切後の予測は `predictions/late/` に置く）。
GitHub Actions（`.github/workflows/prereg-guard.yml`）で、push と PR のたびに実行する。

    uv run python -m src.prereg_guard            # HEAD を確かめる
    uv run python -m src.prereg_guard --rev <rev>
"""

import argparse
import subprocess
import sys
from pathlib import Path

TAG_PATTERN = "prereg-*"
PROTECTED_DIR = "predictions"
LATE_DIR = "predictions/late"


def _git(repo: Path, *args: str) -> str:
    # core.quotepath=false: 日本語のパスをエスケープせずに表示する
    out = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return out.stdout


def prereg_tags(repo: Path) -> list[str]:
    return _git(repo, "tag", "--list", TAG_PATTERN).split()


def changed_paths(repo: Path, tag: str, rev: str = "HEAD") -> list[str]:
    """タグの時点から rev までに変わった、保護対象のパス（`<状態>\\t<パス>` の形）。"""
    out = _git(
        repo, "diff", "--name-status", "--no-renames", tag, rev, "--", PROTECTED_DIR, f":(exclude){LATE_DIR}"
    )
    return [line for line in out.splitlines() if line.strip()]


def violations(repo: Path, rev: str = "HEAD") -> dict[str, list[str]]:
    """タグごとの違反。タグがなければ空。"""
    found = {tag: changed_paths(repo, tag, rev) for tag in prereg_tags(repo)}
    return {tag: paths for tag, paths in found.items() if paths}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="事前登録した predictions/ が変わっていないかを確かめる")
    parser.add_argument("--rev", default="HEAD", help="確かめるリビジョン（既定: HEAD）")
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    tags = prereg_tags(args.repo)
    if not tags:
        print(f"{TAG_PATTERN} のタグはありません（事前登録の前なので確認しません）")
        return 0
    bad = violations(args.repo, args.rev)
    if not bad:
        print(f"OK: {', '.join(tags)} の時点から {PROTECTED_DIR}/ は変わっていません（{LATE_DIR}/ を除く）")
        return 0
    for tag, paths in bad.items():
        print(f"NG: {tag} の時点から、事前登録した予測が変わっています:", file=sys.stderr)
        for p in paths:
            print(f"  {p}", file=sys.stderr)
    print(f"締切後の予測は {LATE_DIR}/ に置いてください（AGENTS.md ルール1）", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
