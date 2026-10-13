"""Eval runner: one pass over the question set, one results file per system."""

from __future__ import annotations

import json
import time
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


_TRANSIENT = {
    "RemoteProtocolError",
    "ConnectError",
    "ReadError",
    "ReadTimeout",
    "WriteTimeout",
    "WriteError",
    "APIError",
    "APIConnectionError",
    "ServerError",
}


def _transient(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__} {current}".lower()
        if type(current).__name__ in _TRANSIENT or "broken pipe" in text or "disconnect" in text:
            return True
        current = current.__cause__ or current.__context__
    return False


def ask_with_retry(pipe: Fieldnote, question: str, attempts: int = 5):
    """One answer, retried when the provider drops the connection mid-eval."""
    for attempt in range(attempts):
        try:
            return pipe.ask(question)
        except Exception as exc:
            if attempt == attempts - 1 or not _transient(exc):
                raise
            wait = 2 * (attempt + 1)
            print(f"retry in {wait}s after {type(exc).__name__}", flush=True)
            time.sleep(wait)


def evaluate_one(pipe: Fieldnote, q: Question, judge: LLM | None, k_pool: int) -> dict:
    ans = ask_with_retry(pipe, q.question)
    retrieval = pipe.last_retrieval
    reranked = [c.chunk for c in retrieval.reranked]
    rank = first_relevant_rank(reranked, q.gold_sources)
    pool_hit = first_relevant_rank([c.chunk for c in retrieval.fused], q.gold_sources)
    correct, why = answer_correct(q, ans.text, judge)
    # For visual-only questions, the gold page only counts if an *image* of it reached the model.
    evidence_hit = any(
        matches_gold(c.chunk, g) and (c.chunk.modality == "image" or not q.visual_only)
        for c in ans.evidence
        for g in q.gold_sources
    )
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
        "evidence_hit": evidence_hit,
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
            "evidence_recall": sum(r.get("evidence_hit", False) for r in sub) / n,
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
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"{stamp}_{system_name}.json"
    config = {k: str(v) for k, v in asdict(pipe.settings).items() if not k.endswith("_dir")} | {
        "mode": pipe.mode,
        "reranker": type(pipe.reranker).__name__,
        "k_total": str(k_pool),
    }
    rows: list[dict] = []

    def save(summary: dict) -> None:
        record = {"system": system_name, "config": config, "summary": summary, "rows": rows}
        out.write_text(json.dumps(record, indent=2))

    for i, q in enumerate(questions, 1):
        row = evaluate_one(pipe, q, judge, k_pool)
        rows.append(row)
        mark = "ok " if row["correct"] else "MISS"
        print(
            f"[{i:>2}/{len(questions)}] {mark} {q.id} {q.category:<14} rank={row['rank']!s:<4} "
            f"{row['timings']['total_s']:.1f}s",
            flush=True,
        )
        save(summarize(rows, k_pool))

    summary = summarize(rows, k_pool)

    # Models are resident after the quality pass. Further passes measure steady-state latency;
    # a dropped connection is retried so one blip does not discard the run.
    extra_latency = []
    for p in range(repeats - 1):
        for q in questions:
            extra_latency.append(ask_with_retry(pipe, q.question).timings["total_s"])
        print(f"latency pass {p + 2}/{repeats} done", flush=True)
    if extra_latency:
        all_lat = [r["timings"]["total_s"] for r in rows] + extra_latency
        summary["latency"] = {
            "p50_s": percentile(all_lat, 50),
            "p95_s": percentile(all_lat, 95),
            "samples": len(all_lat),
        }
        save(summary)
    return summary, out


def results_table(results: dict[str, dict], k_pool: int = 20) -> str:
    """Markdown table comparing systems: visual-only slice first, then all questions."""
    lines = [
        f"| System | Slice | n | Recall@5 | Recall@{k_pool} | Evidence recall "
        "| Answer acc. | Citation acc. | p95 latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, res in results.items():
        s = res["summary"]
        for slice_name in ("visual_only", "all"):
            a = s[slice_name]
            lines.append(
                f"| {name} | {slice_name.replace('_', '-')} | {a['n']} | {a['recall@5']:.2f} | "
                f"{a.get(f'recall@{k_pool}', 0):.2f} | {a.get('evidence_recall', 0):.2f} | "
                f"{a['answer_acc']:.2f} | "
                f"{a['citation_acc']:.2f} | {s['latency']['p95_s']:.1f}s |"
            )
    return "\n".join(lines)


def rescore(results_path: Path, questions: list[Question]) -> tuple[dict, Path]:
    """Re-apply deterministic scorers (numeric, abstain) to saved answers without new LLM
    calls. Judge verdicts on free-text answers are kept unless the answer abstained."""
    from fieldnote.answer.synthesize import extract_citations
    from fieldnote.evaluation.scorers import abstained, numeric_correct

    data = json.loads(results_path.read_text())
    by_id = {q.id: q for q in questions}
    # top_n evidence plus page-figure expansion; ids past this are model inventions.
    n_evidence = int(data["config"].get("top_n", 5)) + int(
        data["config"].get("expand_page_figures", 2)
    )
    changed = []
    for row in data["rows"]:
        q = by_id[row["id"]]
        cited = extract_citations(row["answer"], n_evidence)
        had = row["has_citation"]
        row["has_citation"] = bool(cited) or bool(row.get("citations"))
        if row["has_citation"] != had:
            changed.append(f"{row['id']}: cited {had} -> {row['has_citation']} ({cited})")
        before = row["correct"]
        if q.answer_type == "abstain":
            row["correct"] = abstained(row["answer"])
        elif q.answer_type == "numeric" and q.gold_value is not None:
            row["correct"] = numeric_correct(row["answer"], q.gold_value, q.tolerance or 0.0)
        elif abstained(row["answer"]):
            row["correct"] = False
        if row["correct"] != before:
            changed.append(f"{row['id']}: {before} -> {row['correct']}")
    k_pool = int(data["config"]["k_total"])
    latency = data["summary"]["latency"]
    data["summary"] = summarize(data["rows"], k_pool) | {"latency": latency}
    data["rescored"] = {"at": datetime.now().isoformat(timespec="seconds"), "changed": changed}
    out = results_path.with_name(results_path.stem + "_rescored.json")
    out.write_text(json.dumps(data, indent=2))
    return data["summary"], out
