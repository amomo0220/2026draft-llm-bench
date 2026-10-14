"""表の画像化（HTML の組み立てはブラウザなしで、画像の出力は Chromium があるときだけ確かめる）。"""

import os
import shutil
from pathlib import Path

import pytest

from src.report import Section, Table
from src.report_images import render, section_html

SEC = Section("D_first_round", "12球団の1位入札予想", "説明 <注>", Table(["球団", "A"], [["巨人", "選手（3/3）"], ["阪神", "x"]], [True, False]))


def test_section_html_escapes_and_highlights() -> None:
    page = section_html(SEC)
    assert "説明 &lt;注&gt;" in page
    assert '<tr class="hl"><td>巨人</td>' in page and "<tr><td>阪神</td>" in page


def test_section_html_empty_table() -> None:
    page = section_html(Section("x", "t", "n", Table(["a"], [], empty="なし")))
    assert '<p class="empty">なし</p>' in page and "<table>" not in page


def _chromium() -> str | None:
    path = os.environ.get("CHROMIUM_PATH") or "/opt/pw-browsers/chromium"
    return path if Path(path).exists() or shutil.which(path) else None


@pytest.mark.skipif(_chromium() is None, reason="Chromium がない環境では画像の出力を確かめない")
def test_render_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("playwright")
    [path] = render([SEC], tmp_path, _chromium())
    assert path.name == "D_first_round.png"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
