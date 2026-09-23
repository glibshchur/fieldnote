"""Support-call ingestion: Whisper transcription into timestamped windows.

Speech content is language, so calls are indexed as transcript text with the
audio modality tag. Citations point at a timestamp in the recording.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from fieldnote.types import Chunk

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
WINDOW_S = 30.0
STRIDE_S = 20.0  # 10s overlap so an answer spanning a boundary is not split

Segment = tuple[float, float, str]
Transcriber = Callable[[Path], list[Segment]]


class FasterWhisperTranscriber:
    def __init__(self, model_size: str = "small", device: str = "auto"):
        from faster_whisper import WhisperModel  # heavy import, loaded on demand

        # CTranslate2 has no MPS backend; "auto" picks CUDA when present, else CPU.
        self.model = WhisperModel(model_size, device=device, compute_type="auto")

    def __call__(self, path: Path) -> list[Segment]:
        segments, _info = self.model.transcribe(str(path), vad_filter=True)
        return [(s.start, s.end, s.text.strip()) for s in segments]


def transcribe_cached(path: Path, cache_dir: Path, transcriber: Transcriber) -> list[Segment]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{path.stem}.json"
    if cached.exists():
        return [tuple(s) for s in json.loads(cached.read_text())]  # type: ignore[misc]
    segments = transcriber(path)
    cached.write_text(json.dumps(segments, indent=1))
    return segments


def window_segments(
    segments: list[Segment], window_s: float = WINDOW_S, stride_s: float = STRIDE_S
) -> list[Segment]:
    """Group Whisper segments into overlapping windows of roughly `window_s` seconds."""
    if not segments:
        return []
    windows: list[Segment] = []
    start_idx = 0
    while start_idx < len(segments):
        w_start = segments[start_idx][0]
        texts, end_idx, w_end = [], start_idx, segments[start_idx][1]
        while end_idx < len(segments) and segments[end_idx][0] < w_start + window_s:
            texts.append(segments[end_idx][2])
            w_end = segments[end_idx][1]
            end_idx += 1
        windows.append((w_start, w_end, " ".join(texts)))
        if end_idx >= len(segments):
            break
        nxt = start_idx + 1
        while nxt < end_idx and segments[nxt][0] < w_start + stride_s:
            nxt += 1
        start_idx = nxt
    return windows


def ingest_calls(calls_dir: Path, cache_dir: Path, transcriber: Transcriber) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(calls_dir.iterdir()):
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        for start, end, text in window_segments(transcribe_cached(path, cache_dir, transcriber)):
            chunks.append(
                Chunk(
                    modality="audio",
                    source_type="call",
                    source=path.name,
                    text=text,
                    start_s=round(start, 2),
                    end_s=round(end, 2),
                )
            )
    return chunks
