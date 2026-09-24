# Eval set

`questions.jsonl`: 40 questions, one JSON object per line. Write them **before** tuning
retrieval, by reading the actual manual pages, looking at the photos and listening to the
calls. Every gold source is checked by hand.

| Category | Target count | Visual-only? |
|---|---|---|
| `visual_chart` | 7 | yes |
| `visual_diagram` (wiring diagrams, tables that exist only as images) | 4 | yes |
| `visual_photo` | 4 | yes |
| `text` | 12 | no |
| `audio` | 7 | no |
| `cross_modal` | 6 | no |

A category starting with `visual` counts toward the 15 visual-only questions.

## Visual-only audit

A visual-only question counts only if its answer cannot be found as text **anywhere in the
corpus**, not just on the gold page. For each one: read the answer off the render
(`scripts/qtools.py render/crop`), then search every manual's text layer for the equipment
identifier together with the answer. On 2026-09-24 this audit retired the original q12 (a
CoreSense run-time limit shown as an image in the Goodman manual but stated as plain text in
the Copeland bulletin). It is kept in `reserve.jsonl` with `kind: retired`.

## Schema

```json
{
  "id": "q07",
  "question": "Per the discharge pressure chart, what head pressure should a 48TC show at 95F outdoor?",
  "category": "visual_chart",
  "gold_sources": [{"doc": "carrier_48tc_service.pdf", "page": 112}],
  "gold_answer": "About 118 psig",
  "answer_type": "numeric",
  "gold_value": 118,
  "tolerance": 5
}
```

`gold_sources` entries take one of three forms:
- `{"doc": "<manual>.pdf", "page": <1-indexed page>}`: any chunk from that page counts
- `{"photo": "<file name>"}`
- `{"call": "<recording file>", "t": <seconds>}`: the retrieved window must contain `t`

`answer_type`:
- `numeric`: correct if any number in the answer is within `tolerance` of `gold_value`
- `text`: graded by an LLM judge against `gold_answer` (spot-check 10 verdicts by hand)
- `abstain`: correct if the system says "Not found in the provided sources."

## Metrics

Retrieval and answer quality are reported separately:
- **Recall@5 / MRR**: over the reranked list (what the answer model sees)
- **Recall@20**: over the fused candidate pool, before reranking (did any lane find it?)
- **Answer acc.**: numeric tolerance, abstain check, or judge verdict
- **Citation acc.**: at least one cited source matches a gold source
- **p50 / p95 latency**: end to end, over `--repeats` passes
