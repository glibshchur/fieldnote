"""Scorers. Retrieval and generation are scored separately, on purpose:
a wrong answer from the right evidence is a different bug from a wrong answer
from the wrong evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass

from fieldnote.llm import LLM, text_block
from fieldnote.types import Chunk

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
NOT_FOUND = "not found in the provided sources"


@dataclass
class Question:
    id: str
    question: str
    category: str
    gold_sources: list[dict]
    gold_answer: str
    answer_type: str = "text"  # "numeric" | "text" | "abstain"
    gold_value: float | None = None
    tolerance: float | None = None

    @property
    def visual_only(self) -> bool:
        return self.category.startswith("visual")


def matches_gold(chunk: Chunk, gold: dict) -> bool:
    """A gold source is a manual page, a photo, or a moment in a call."""
    if "doc" in gold:
        return chunk.source == gold["doc"] and chunk.page == gold.get("page")
    if "photo" in gold:
        return chunk.source == gold["photo"]
    if "call" in gold:
        if chunk.source != gold["call"]:
            return False
        t = gold.get("t")
        if t is None:
            return True
        return (chunk.start_s or 0) <= t <= (chunk.end_s or 0)
    return False


def first_relevant_rank(chunks: list[Chunk], golds: list[dict]) -> int | None:
    for rank, c in enumerate(chunks, start=1):
        if any(matches_gold(c, g) for g in golds):
            return rank
    return None


def recall_at(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


# Numbers that are references, not readings: citations, pages, figures, bulletins, tables.
REF_RE = re.compile(
    r"\[S\d+(?:\s*,\s*S?\d+)*\]|\b(?:p|pp|page|pages|fig|figure|table|bulletin|section)\.?\s*"
    r"[\w.-]*\d[\w.-]*",
    re.I,
)


def numeric_correct(answer: str, gold: float, tolerance: float) -> bool:
    """Any stated number within tolerance, ignoring citation/page/figure references and
    never crediting an answer that declined."""
    if abstained(answer):
        return False
    cleaned = REF_RE.sub(" ", answer.replace(",", ""))
    return any(abs(float(n) - gold) <= tolerance for n in NUM_RE.findall(cleaned))


def abstained(answer: str) -> bool:
    return NOT_FOUND in answer.lower()


JUDGE_SYSTEM = """\
You grade answers from an HVAC field-service assistant against a reference answer. \
An answer is CORRECT if it states the same fact or value as the reference (wording \
may differ; extra correct detail is fine). It is INCORRECT if it contradicts the \
reference, omits the key fact, or answers a different question. Reply with CORRECT \
or INCORRECT on the first line, then one sentence of reasoning."""


def judge_correct(llm: LLM, q: Question, answer: str) -> tuple[bool, str]:
    prompt = (
        f"Question: {q.question}\n\nReference answer: {q.gold_answer}\n\nAssistant answer: {answer}"
    )
    res = llm.complete(JUDGE_SYSTEM, [text_block(prompt)], max_tokens=512)
    first = res.text.strip().splitlines()[0].upper() if res.text.strip() else ""
    return first.startswith("CORRECT"), res.text


def answer_correct(q: Question, answer: str, judge: LLM | None) -> tuple[bool, str]:
    if q.answer_type == "abstain":
        return abstained(answer), "abstain check"
    if q.answer_type == "numeric" and q.gold_value is not None:
        return numeric_correct(answer, q.gold_value, q.tolerance or 0.0), "numeric tolerance"
    if abstained(answer):
        return False, "abstained on an answerable question"
    if judge is None:
        return False, "no judge configured"
    return judge_correct(judge, q, answer)


def percentile(values: list[float], p: float) -> float:
    import numpy as np

    return float(np.percentile(values, p)) if values else 0.0
