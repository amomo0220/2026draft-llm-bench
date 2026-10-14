"""事前登録の保護（一時的な git リポジトリで確かめる）。"""

import subprocess
from pathlib import Path

import pytest

from src.prereg_guard import main, violations

TAG = "prereg-2026-10-20"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false",
         "-c", "tag.gpgsign=false", *args],
        cwd=repo, check=True, capture_output=True,
    )  # fmt: skip


def _write(repo: Path, rel: str, text: str = "{}") -> None:
    f = repo / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text, encoding="utf-8")


def _commit(repo: Path) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "c")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _write(tmp_path, "predictions/D/gpt/D-01.json")
    _write(tmp_path, "predictions/late/.gitkeep", "")
    _write(tmp_path, "README.md", "x")
    _commit(tmp_path)
    return tmp_path


def _tag(repo: Path) -> None:
    _git(repo, "tag", "-a", TAG, "-m", "事前登録")


def test_no_tag_allows_changes(repo: Path) -> None:
    _write(repo, "predictions/D/gpt/D-01.json", '{"changed": 1}')
    _commit(repo)
    assert violations(repo) == {}
    assert main(["--repo", str(repo)]) == 0


def test_unchanged_after_tag(repo: Path) -> None:
    _tag(repo)
    _write(repo, "README.md", "y")  # predictions/ の外は変えてよい
    _commit(repo)
    assert main(["--repo", str(repo)]) == 0


def test_late_dir_is_allowed(repo: Path) -> None:
    _tag(repo)
    _write(repo, "predictions/late/D-late/gpt/D-late-01.json")
    _commit(repo)
    assert violations(repo) == {}


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda r: _write(r, "predictions/D/gpt/D-01.json", '{"changed": 1}'), "M\tpredictions/D/gpt/D-01.json"),
        (lambda r: (r / "predictions/D/gpt/D-01.json").unlink(), "D\tpredictions/D/gpt/D-01.json"),
        (lambda r: _write(r, "predictions/D/gpt/D-02.json"), "A\tpredictions/D/gpt/D-02.json"),
    ],
    ids=["modify", "delete", "add"],
)
def test_changes_after_tag_are_rejected(repo: Path, change, expected: str, capsys: pytest.CaptureFixture[str]) -> None:
    _tag(repo)
    change(repo)
    _commit(repo)
    assert violations(repo) == {TAG: [expected]}
    assert main(["--repo", str(repo)]) == 1
    assert "predictions/late" in capsys.readouterr().err


def test_rename_counts_as_delete_and_add(repo: Path) -> None:
    _tag(repo)
    _git(repo, "mv", "predictions/D/gpt/D-01.json", "predictions/D/gpt/D-09.json")
    _commit(repo)
    assert violations(repo) == {TAG: ["D\tpredictions/D/gpt/D-01.json", "A\tpredictions/D/gpt/D-09.json"]}


def test_rev_option_checks_given_revision(repo: Path) -> None:
    _tag(repo)
    _write(repo, "predictions/D/gpt/D-01.json", '{"changed": 1}')
    _commit(repo)
    assert main(["--repo", str(repo), "--rev", TAG]) == 0
    assert main(["--repo", str(repo), "--rev", "HEAD"]) == 1


def test_other_tags_are_ignored(repo: Path) -> None:
    _git(repo, "tag", "v0.1")
    _write(repo, "predictions/D/gpt/D-01.json", '{"changed": 1}')
    _commit(repo)
    assert violations(repo) == {}
