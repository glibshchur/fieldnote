"""Answer synthesis: a vision model reads the retrieved evidence itself.

In native mode, image evidence goes to the model as pixels. In caption mode
(the baseline), the same evidence goes as its caption text only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from fieldnote.answer.prompts import ANSWER_SYSTEM
from fieldnote.llm import LLM, image_block, text_block
from fieldnote.types import Candidate, format_ts

CITE_RE = re.compile(r"\[S(\d+)\]")


@dataclass
class Citation:
    sid: str
    label: str
    candidate: Candidate
    image_path: str | None = None


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation]
    evidence: list[Candidate]
    stop_reason: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)


def build_content(question: str, evidence: list[Candidate], use_images: bool) -> list[dict]:
    blocks: list[dict] = [text_block("Sources:")]
    for i, cand in enumerate(evidence, start=1):
        c = cand.chunk
        header = f"[S{i}] {c.source_type} | {c.citation_label()}"
        if c.source_type == "call" and c.start_s is not None:
            header += f"-{format_ts(c.end_s or c.start_s)}"
        if c.modality == "image" and use_images and c.image_path:
            blocks.append(text_block(header))
            blocks.append(image_block(c.image_path))
        else:
            body = c.text or "(no text)"
            blocks.append(text_block(f"{header}\n{body}"))
    blocks.append(text_block(f"Question: {question}"))
    return blocks


def synthesize(
    llm: LLM, question: str, evidence: list[Candidate], use_images: bool = True
) -> tuple[str, list[str], str | None, dict[str, int]]:
    """Returns (answer text, cited source ids like "S2", stop reason, token usage)."""
    result = llm.complete(ANSWER_SYSTEM, build_content(question, evidence, use_images))
    cited = []
    for n in CITE_RE.findall(result.text):
        sid = f"S{n}"
        if sid not in cited and 1 <= int(n) <= len(evidence):
            cited.append(sid)
    usage = {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens}
    return result.text, cited, result.stop_reason, usage
