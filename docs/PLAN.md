# Fieldnote: implementation plan

Multimodal RAG for commercial HVAC field technicians.
Project 01 from the baswe "six projects" guide. Effort: two weekends. Level: intermediate.

---

## 1. What it does

A technician on a roof asks: *"Unit 7 keeps tripping on high pressure. What should the head pressure be at 95°F outdoor?"*

The answer isn't in a paragraph. It's a value read off a pressure curve on page 112 of a service manual. Fieldnote:

1. Searches three kinds of evidence at once:
   - **manual pages**: charts, wiring diagrams and tables, embedded as images
   - **job-site photos**: failed parts, burnt contactors, iced coils
   - **recorded support calls**: transcribed with timestamps
2. Merges the candidates and reranks them with a model that can score images as well as text.
3. Has a vision model answer **from the retrieved images**, not from captions of them.
4. Returns the answer along with a **cropped image of the chart**, the page number and document name (or the call timestamp), so the tech can check the number themselves.

What it proves: some evidence exists only inside a picture, and a system that turns pictures into captions loses it. The results table shows this with numbers.

---

## 2. Architecture

```
                 ┌──────────────── INGEST ────────────────┐
 manual PDFs ──► PyMuPDF: page render + figure/table crops ─► image encoder ──┐
            └──► text blocks per page ─────────────────────► text encoder  ──┤
 photos ───────► EXIF/filename metadata ───────────────────► image encoder ──┼─► Qdrant
 call audio ───► faster-whisper (timestamped segments) ────► text encoder  ──┘   (named vectors,
                                                                                   modality tag,
                 ┌──────────────── QUERY ─────────────────┐                        source, page,
 question ──► text→text search  ┐                                                  bbox, timestamp)
          ──► text→image search ├─(parallel, k=20 total)─► multimodal reranker ─► top 5
          ──► text→audio search ┘                                  │
                                                                   ▼
                              vision LLM (Claude) gets question + images + snippets
                                                                   ▼
                              answer + citations [doc, page, crop.png | call, 03:14]
```

**Key decisions:**

| Decision | Choice | Why |
|---|---|---|
| Vector DB | **Qdrant** (Docker) | Named vectors per modality, payload filters on `modality`, simple local setup |
| Text encoder | `BAAI/bge-m3` or `bge-base-en-v1.5` | Strong, free, runs locally |
| Image encoder | **SigLIP** (`google/siglip-so400m-patch14-384`) | Shared text-image space, so a text query can find an image directly |
| Page-image encoder (ablation) | **ColQwen2 / ColPali** | Built for document pages with charts; a strong "what I tried" story |
| Audio | **faster-whisper** → text encoder, keep timestamps | Speech content is language, so transcripts suit calls. Citations point to a time in the call |
| Fusion | **Late fusion**: parallel per-modality search, then merge and rerank | Every modality keeps its own encoder; nothing is squashed into text |
| Reranker | **jina-reranker-m0** (multimodal) with an RRF fallback | A text-only cross-encoder can't score an image candidate |
| Synthesis | **Claude Sonnet 5** (`claude-sonnet-5`), vision | Reads axis values off the actual chart image |
| Crops | Index figure/table regions as their own chunks, bbox stored | The citation crop comes for free and retrieval is more precise than whole pages |
| API / UI | FastAPI + a small Streamlit (or Next.js) demo | Enough for a demo GIF and a portfolio link |

---

## 3. The corpus (and how to be honest about it)

You don't have a real contractor's 300 manuals, so build a **scaled-down corpus from public data** and say so in the README.

| Modality | Target | Source |
|---|---|---|
| Manuals | 25–40 PDFs | Service and installation manuals that Carrier, Trane, Lennox, Daikin, Copeland and others publish publicly. Pick ones with pressure-temperature charts, wiring diagrams, fault-code tables and superheat/subcooling charts |
| Photos | 150–300 | Wikimedia Commons / CC-licensed images of failed contactors, capacitors, iced evaporator coils and burnt wiring, plus your own photos of equipment. Record the license per file in `data/photos/LICENSES.csv` |
| Calls | 10–15 recordings, 2–5 min each | Script realistic tech ↔ manufacturer support calls grounded in the manuals, then record them yourself or with a friend (better) or generate them with TTS. Put a fact in each call that only the call contains |

Don't commit copyrighted PDFs. Commit a `data/manifest.yaml` with URLs and checksums, plus a `make fetch` script.

---

## 4. The eval: build it first

The ship gate asks for **40 questions, 15 of them unanswerable from text alone**.

| Category | Count | Example |
|---|---|---|
| Visual-only: chart | 7 | "Per the P-T chart, what's the expected suction pressure at 40°F evap for R-410A on the 48TC?" |
| Visual-only: diagram/table-as-image | 4 | "On the wiring diagram, which terminal does the high-pressure switch connect to?" |
| Visual-only: photo | 4 | "In job photo 0142, what failed on the contactor?" |
| Text | 12 | Procedure and spec questions answered in prose |
| Audio | 7 | "What did manufacturer support say to check before replacing the TXV?" |
| Cross-modal | 6 | Needs both a call and a manual page |

Each item in `eval/questions.jsonl`:
```json
{"id": "q07", "question": "...", "category": "visual_chart",
 "gold_sources": [{"doc": "carrier_48tc_service.pdf", "page": 112, "region": "fig_3"}],
 "gold_answer": "~118 psig", "answer_type": "numeric", "tolerance": 5}
```

**Metrics**, with retrieval and generation reported separately:
- **Retrieval:** recall@5, recall@20, MRR, per category. A hit means the gold page/region/photo/call segment was retrieved.
- **Answer:** numeric within tolerance for numeric questions; LLM judge with a rubric for free text; spot-check 10 judge verdicts by hand.
- **Citation:** the cited source matches gold, and the crop contains the relevant figure.
- **Latency:** p50 and p95 end-to-end at k=20, over 3 runs.

**The baseline, and the point of the project:**
- **Caption-only:** caption every image and page with a VLM (`claude-haiku-4-5`), embed the captions as text, same pipeline otherwise.
- **Fieldnote:** native image embeddings plus multimodal rerank plus a vision answer.
- Headline table: both systems on the **15 visual-only questions**, then on all 40.

If you built Groundtruth (project 04), reuse its scorers. The two projects are meant to fit together.

---

## 5. Repo layout

```
fieldnote/
├── README.md               # results table first, then how it works, then how to run
├── pyproject.toml          # uv
├── docker-compose.yml      # qdrant (+ api)
├── Makefile                # fetch, ingest, eval, serve, demo
├── data/manifest.yaml
├── fieldnote/
│   ├── ingest/  pdf.py  photos.py  audio.py
│   ├── encoders/ text.py  image.py  colpali.py
│   ├── index/   qdrant_store.py
│   ├── retrieve/ search.py  fuse.py  rerank.py
│   ├── answer/  synthesize.py  crop.py  prompts.py
│   ├── baseline/ caption_pipeline.py
│   └── api.py
├── eval/
│   ├── questions.jsonl
│   ├── run_eval.py         # --system fieldnote|caption --k 20
│   ├── scorers.py
│   └── results/            # dated JSON + markdown tables, committed
├── app/streamlit_app.py
├── tests/                  # unit + a CI smoke eval on 5 questions
└── docs/
    ├── architecture.md
    ├── error_analysis.md
    └── demo.gif
```

---

## 6. Build schedule (two weekends)

### Weekend 1: data, eval, baseline

**Sat AM: scaffold and corpus**
- Repo, `uv`, Docker Qdrant, Makefile, `.env` for `ANTHROPIC_API_KEY`
- Choose and download manuals; collect photos; write and record call scripts

**Sat PM: write the eval set (before any retrieval code)**
- Write the 40 questions by reading the actual manual pages and photos and listening to the calls
- Record gold source and gold answer for each, and review each one by hand

**Sun AM: ingestion**
- `pdf.py`: render pages at 150 dpi, extract text blocks, detect figure/table regions (PyMuPDF `get_images` + drawing-cluster bboxes), save crops
- `audio.py`: faster-whisper to segments, merge into ~30s windows with timestamps
- `photos.py`: load, resize, metadata

**Sun PM: caption baseline end to end**
- Caption all images and pages, embed as text, retrieve, answer
- `run_eval.py` produces the first numbers. You now have something to beat.

### Weekend 2: the real system, measurement, packaging

**Sat AM: native multimodal index**
- SigLIP for photos and figure crops, text encoder for text and transcripts, all in Qdrant with `modality` payload

**Sat PM: fusion, rerank, synthesis**
- Parallel async searches, merge, jina-reranker-m0 top-20 → top-5
- Claude vision prompt: answer **only** from the provided evidence, cite `[S1]`-style IDs, say "not found" when evidence is missing
- Map citations to doc/page/crop or call/timestamp

**Sun AM: measure and tune**
- Full eval for both systems; latency harness (p95 < 6s at k=20)
- One ablation: SigLIP vs ColQwen2 for manual pages, **or** rerank on vs off
- Write `docs/error_analysis.md`: the 5 questions that still fail, and why

**Sun PM: package**
- README: results table at the top, architecture diagram, the "trap" paragraph, tradeoffs, how to run
- Streamlit demo showing answer, cropped chart and page number; record `demo.gif`
- GitHub Actions: lint, unit tests, 5-question smoke eval

---

## 7. Ship gate checklist

- [ ] 40-question eval set, 15 visual-only, gold sources reviewed by hand
- [ ] Table: Fieldnote vs caption-only on the 15 visual-only questions, with a clear margin
- [ ] Every answer has a citation (doc + page + cropped image, or call + timestamp)
- [ ] p95 latency < 6s at k=20, with the measurement method documented
- [ ] Written recommendation: when late fusion is worth its cost, and when captions are good enough
- [ ] Error analysis of the 5 failing questions

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| p95 > 6s (rerank + VLM) | Rerank only top 20; send the VLM ≤ 4 crops, not full pages; stream the answer; cache encoders in memory |
| SigLIP weak on dense charts | That's why ColQwen2 is the ablation. Report whichever wins |
| Crop detection misses figures | Fallback: tile each page into a 2×2 grid and index the tiles |
| Eval set too easy or biased | Write the questions before building retrieval; include distractor pages with similar charts from other models |
| Copyright | Don't commit manuals; manifest + fetch script; CC-licensed photos with a license file |

---

## 9. After it ships: portfolio and GitHub

1. **GitHub:** public repo `fieldnote`, with topics (`rag`, `multimodal`, `qdrant`, `siglip`, `whisper`, `llm-eval`), a pinned repo, the demo GIF in the README, and a release tag `v1.0` with the results table in the notes.
2. **Portfolio entry:** problem (one line), the results table, the demo GIF, three tradeoffs you'd defend in an interview, and a link to the repo.
3. **Write-up / LinkedIn post:** "Captioning a chart throws away the axis values," with the before/after table.
4. **Interview talking points:** early vs late fusion; why a text cross-encoder can't rerank images; how you kept retrieval and generation metrics separate; what you'd change at 300 manuals and 4,000 photos (sharding, batch ingestion, a ColPali token budget).
