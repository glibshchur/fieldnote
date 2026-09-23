"""Render the support-call scripts to audio with macOS `say` (two voices per call).

Usage: uv run python scripts/tts_calls.py [--only call_03_mitsubishi_u1]
Writes data/calls/<script name>.wav (16 kHz mono PCM). macOS only.

The recordings are synthetic: they stand in for real support-line calls so the
audio modality has something to transcribe. Voices rotate per call so Whisper
sees more than one speaker profile.
"""

from __future__ import annotations

import argparse
import random
import re
import subprocess
import tempfile
import wave
from pathlib import Path

SCRIPTS = Path("data/calls/scripts")
OUT = Path("data/calls")
RATE = 16000

# (tech voice, rep voice), rotated across calls
VOICE_PAIRS = [
    ("Reed (English (US))", "Samantha"),
    ("Eddy (English (US))", "Shelley (English (US))"),
    ("Rocko (English (US))", "Karen"),
    ("Ralph", "Sandy (English (US))"),
    ("Reed (English (US))", "Flo (English (US))"),
    ("Eddy (English (US))", "Daniel"),
    ("Rocko (English (US))", "Samantha"),
    ("Ralph", "Shelley (English (US))"),
]
LINE_RE = re.compile(r"^(TECH|REP):\s*(.+)$")


def parse(script: Path) -> list[tuple[str, str]]:
    return [
        (m.group(1), m.group(2).strip())
        for line in script.read_text().splitlines()
        if (m := LINE_RE.match(line.strip()))
    ]


def say(text: str, voice: str, out: Path, words_per_min: int) -> None:
    subprocess.run(
        [
            "say",
            "-v",
            voice,
            "-r",
            str(words_per_min),
            "-o",
            str(out),
            "--file-format=WAVE",
            f"--data-format=LEI16@{RATE}",
            text,
        ],
        check=True,
    )


def render_call(script: Path, voices: tuple[str, str], seed: int) -> Path:
    rng = random.Random(seed)
    turns = parse(script)
    frames: list[bytes] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (speaker, text) in enumerate(turns):
            voice = voices[0] if speaker == "TECH" else voices[1]
            clip = Path(tmp) / f"{i:03d}.wav"
            say(text, voice, clip, words_per_min=rng.randint(172, 190))
            with wave.open(str(clip)) as w:
                assert w.getframerate() == RATE and w.getnchannels() == 1
                frames.append(w.readframes(w.getnframes()))
            gap_s = rng.uniform(0.35, 0.8)  # natural pause between speakers
            frames.append(b"\x00\x00" * int(RATE * gap_s))
    out = OUT / f"{script.stem}.wav"
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(frames))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="render a single script by name")
    args = ap.parse_args()
    scripts = sorted(SCRIPTS.glob("call_*.md"))
    for i, script in enumerate(scripts):
        if args.only and script.stem != args.only:
            continue
        voices = VOICE_PAIRS[i % len(VOICE_PAIRS)]
        out = render_call(script, voices, seed=i)
        with wave.open(str(out)) as w:
            secs = w.getnframes() / w.getframerate()
        print(f"{out}  {secs / 60:.1f} min  tech={voices[0]}  rep={voices[1]}")


if __name__ == "__main__":
    main()
