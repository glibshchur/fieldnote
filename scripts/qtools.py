"""Helpers for writing and auditing eval questions against the corpus.

Usage (from the repo root):
  uv run python scripts/qtools.py render <file.pdf> <page> [dpi]          # page -> PNG
  uv run python scripts/qtools.py crop <file.pdf> <page> x0 y0 x1 y1 [dpi] # region -> PNG
  uv run python scripts/qtools.py text <file.pdf> <page>                   # text layer
  uv run python scripts/qtools.py grep <file.pdf> <regex>                  # search a manual

Renders go to artifacts/qtools/. Coordinates are PDF points (a letter page is 612 x 792).
A visual-only question needs its gold answer to be readable in the render and absent
from `text`/`grep` output.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pymupdf

pymupdf.TOOLS.mupdf_display_errors(False)
MANUALS = Path("data/manuals")
OUT = Path("artifacts/qtools")


def render(f: str, page: str, dpi: str = "110") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{Path(f).stem}_p{int(page):03d}.png"
    pymupdf.open(MANUALS / f)[int(page) - 1].get_pixmap(dpi=int(dpi)).save(out)
    print(out)


def crop(f: str, page: str, x0: str, y0: str, x1: str, y1: str, dpi: str = "200") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = pymupdf.open(MANUALS / f)[int(page) - 1]
    rect = pymupdf.Rect(*map(float, (x0, y0, x1, y1)))
    out = OUT / f"{Path(f).stem}_p{int(page):03d}_crop_{int(rect.x0)}_{int(rect.y0)}.png"
    p.get_pixmap(clip=rect, dpi=int(dpi)).save(out)
    print(out, "page size:", p.rect)


def text(f: str, page: str) -> None:
    print(pymupdf.open(MANUALS / f)[int(page) - 1].get_text())


def grep(f: str, pattern: str) -> None:
    rx = re.compile(pattern, re.I)
    for p in pymupdf.open(MANUALS / f):
        t = " ".join(p.get_text().split())
        for m in rx.finditer(t):
            s = max(0, m.start() - 80)
            print(f"p{p.number + 1}: ...{t[s : m.end() + 80]}...")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"render": render, "crop": crop, "text": text, "grep": grep}[cmd](*args)
