"""Eval runner: one pass over the question set, one results file per system."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from fieldnote.evaluation.scorers import (
    Question,
    answer_correct,
    first_relevant_rank,
    matches_gold,
    percentile,
    recall_at,
    reciprocal_rank,
)
from fieldnote.llm import LLM
from fieldnote.pipeline import Fieldnote


def load_questions(path: Path) -> list[Question]:
    return [Question(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def evaluate_one(pipe: Fieldnote, q: Question, judge: LLM | None, k_pool: int) -> dict:
    ans = pipe.ask(q.question)
    retrieval = pipe.last_retrieval
    reranked = [c.chunk for c in retrieval.reranked]
    rank = first_relevant_rank(reranked, q.gold_sources)
    pool_hit = first_relevant_rank([c.chunk for c in retrieval.fused], q.gold_sources)
    correct, why = answer_correct(q, ans.text, judge)
    cited_ok = any(
        matches_gold(c.candidate.chunk, g) for c in ans.citations for g in q.gold_sources
    )
    return {
        "id": q.id,
        "category": q.category,
        "visual_only": q.visual_only,
        "question": q.question,
        "gold_answer": q.gold_answer,
        "answer": ans.text,
        "rank": rank,
        "recall@5": recall_at(rank, 5),
        f"recall@{k_pool}": recall_at(pool_hit, k_pool),
        "mrr": reciprocal_rank(rank),
        "correct": correct,
        "judge_note": why,
        "citation_correct": cited_ok,
        "has_citation": bool(ans.citations),
        "citations": [c.label for c in ans.citations],
        "timings": ans.timings,
        "usage": ans.usage,
    }


def summarize(rows: list[dict], k_pool: int) -> dict:
    def agg(sub: list[dict]) -> dict:
        n = len(sub) or 1
        return {
            "n": len(sub),
            "recall@5": sum(r["recall@5"] for r in sub) / n,
            f"recall@{k_pool}": sum(r[f"recall@{k_pool}"] for r in sub) / n,
            "mrr": sum(r["mrr"] for r in sub) / n,
            "answer_acc": sum(r["correct"] for r in sub) / n,
            "citation_acc": sum(r["citation_correct"] for r in sub) / n,
            "cited_rate": sum(r["has_citation"] for r in sub) / n,
        }

    latencies = [r["timings"]["total_s"] for r in rows]
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    return {
        "all": agg(rows),
        "visual_only": agg([r for r in rows if r["visual_only"]]),
        "by_category": {c: agg(v) for c, v in sorted(by_cat.items())},
        "latency": {"p50_s": percentile(latencies, 50), "p95_s": percentile(latencies, 95)},
    }


def run_eval(
    pipe: Fieldnote,
    questions: list[Question],
    judge: LLM | None,
    out_dir: Path,
    system_name: str,
    repeats: int = 1,
) -> tuple[dict, Path]:
    k_pool = pipe.settings.k_total
    rows = [evaluate_one(pipe, q, judge, k_pool) for q in questions]
    # Extra passes only re-measure latency; quality numbers come from the first pass.
    extra_latency = []
    for _ in range(repeats - 1):
        for q in questions:
            extra_latency.append(pipe.ask(q.question).timings["total_s"])

    summary = summarize(rows, k_pool)
    if extra_latency:
        all_lat = [r["timings"]["total_s"] for r in rows] + extra_latency
        summary["latency"] = {
            "p50_s": percentile(all_lat, 50),
            "p95_s": percentile(all_lat, 95),
            "samples": len(all_lat),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"{stamp}_{system_name}.json"
    config = {k: str(v) for k, v in asdict(pipe.settings).items() if not k.endswith("_dir")} | {
        "mode": pipe.mode,
        "reranker": type(pipe.reranker).__name__,
        "k_total": str(k_pool),
    }
    out.write_text(
        json.dumps(
            {"system": system_name, "config": config, "summary": summary, "rows": rows}, indent=2
        )
    )
    return summary, out


def results_table(results: dict[str, dict], k_pool: int = 20) -> str:
    """Markdown table comparing systems: visual-only slice first, then all questions."""
    lines = [
        f"| System | Slice | n | Recall@5 | Recall@{k_pool} | MRR "
        "| Answer acc. | Citation acc. | p95 latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, res in results.items():
        s = res["summary"]
        for slice_name in ("visual_only", "all"):
            a = s[slice_name]
            lines.append(
                f"| {name} | {slice_name.replace('_', '-')} | {a['n']} | {a['recall@5']:.2f} | "
                f"{a.get(f'recall@{k_pool}', 0):.2f} | {a['mrr']:.2f} | {a['answer_acc']:.2f} | "
                f"{a['citation_acc']:.2f} | {s['latency']['p95_s']:.1f}s |"
            )
    return "\n".join(lines)
