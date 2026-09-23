"""Corpus fetcher. Manuals are not committed (they are the manufacturers'
copyright); data/manifest.yaml records where each one came from and its hash."""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_manifest(manifest: Path, out_dir: Path) -> Iterator[str]:
    spec = yaml.safe_load(manifest.read_text()) or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in spec.get("manuals", []):
        dest = out_dir / item["file"]
        if not dest.exists():
            req = urllib.request.Request(item["url"], headers={"User-Agent": "fieldnote/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                dest.write_bytes(resp.read())
        digest = sha256(dest)
        expected = item.get("sha256")
        if expected and digest != expected:
            yield f"CHECKSUM MISMATCH {dest.name}: {digest}"
        else:
            yield f"ok {dest.name} {digest[:12]}"
