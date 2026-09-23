"""Corpus fetcher. Manuals are not committed (they are the manufacturers'
copyright); data/manifest.yaml records where each one came from, and
data/manuals.sha256 pins the exact bytes so the eval is reproducible."""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import yaml

# Some manufacturer CDNs reject non-browser user agents.
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) fieldnote/0.1"
PDF_MAGIC = b"%PDF"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_checksums(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def write_checksums(path: Path, sums: dict[str, str]) -> None:
    path.write_text("".join(f"{d}  {n}\n" for n, d in sorted(sums.items())))


def download(url: str, dest: Path, timeout: int = 120) -> None:
    import truststore

    # Verify TLS with the OS trust store; uv's standalone Python ships no CA roots.
    truststore.inject_into_ssl()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    if not tmp.read_bytes()[:4] == PDF_MAGIC:
        tmp.unlink()
        raise ValueError("response is not a PDF")
    tmp.rename(dest)


def fetch_manifest(manifest: Path, out_dir: Path, checksums: Path | None = None) -> Iterator[str]:
    """Download missing manuals, then verify against (or record into) the checksum file."""
    spec = yaml.safe_load(manifest.read_text()) or {}
    checksums = checksums or manifest.parent / "manuals.sha256"
    pinned = read_checksums(checksums)
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in spec.get("manuals", []):
        name = item["file"]
        dest = out_dir / name
        if not dest.exists():
            try:
                download(item["url"], dest)
            except Exception as e:  # keep going; report every failure at the end
                yield f"FAILED {name}: {e}"
                continue
        digest = sha256(dest)
        if name in pinned and pinned[name] != digest:
            yield f"CHECKSUM MISMATCH {name}: expected {pinned[name][:12]}, got {digest[:12]}"
            continue
        pinned[name] = digest
        yield f"ok {name} {dest.stat().st_size / 1e6:.1f} MB"
    write_checksums(checksums, pinned)
