# Fieldnote

**Multimodal RAG for HVAC field technicians.** Fieldnote answers questions whose evidence exists
only inside a picture: a pressure curve on page 112 of a service manual, a wiring diagram, a photo
of a failed contactor. It also searches recorded manufacturer support calls.

> A tech on a roof asks why unit 7 keeps tripping on high pressure. The answer lives in a chart,
> not in a paragraph.

**Status:** pipeline, corpus (40 public manuals, 3,445 pages), 40-question eval set and
TTS support calls are built. The caption-only baseline run is next. Every number below comes
from a results file in `eval/results/`; none are typed by hand.

## Results so far

Fieldnote on all 40 questions, answer model `gemini-3.8-flash`:

| Iteration | Recall@5 | Answer acc. (all 40) | Answer acc. (15 visual-only) | p95 latency |
|---|---|---|---|---|
| Parallel search + RRF | 0.50 | 0.30 | 0.13 | 5.2 s |
| + multimodal reranker (jina-reranker-m0) | 0.80 | 0.65 | 0.33* | 9.7 s |
| + page-figure expansion, capped rerank pool | 0.80 | **0.75** | **0.53** | 6.2 s |

\* included one question later found to leak as text elsewhere in the corpus; it was replaced
(see `eval/README.md`).

What the numbers say:
- **When the gold chart reaches the model, it answers correctly (8 of 8).** Every remaining
  visual miss (7 of 7) is a retrieval miss. Image retrieval over dense diagrams is the
  bottleneck, not the answer model. Next: a ColQwen2 page-image index.
- **Page-figure expansion** fixed chart questions (0.25 to 0.75): text retrieval finds the right
  page through the chart's title and axis labels, but the curve only exists in the image, so
  the page's figures now ride along with the text.
- **The reranker** moved the right support-call segment into the top 5 every time
  (audio 0.14 to 0.86, cross-modal 0.00 to 1.00).
- p95 latency is 0.2 s over the 6 s target; reranking and answering are the two big costs.

### Head to head: caption-only baseline vs Fieldnote

Same corpus, questions, reranker (scoring caption text in baseline mode), page expansion and
answer model. The baseline captions every figure with `gemini-3.8-flash` (told to transcribe
labels and read curve values) and never sends pixels anywhere.

| System | Visual-only acc. (15) | Charts (8) | Diagrams / image tables (7) | All 40 | p95 latency |
|---|---|---|---|---|---|
| Caption-only baseline | 0.47 | 0.50 | **0.43** | 0.38 | **3.5 s** |
| Fieldnote | **0.53** | **0.75** | 0.29 | **0.75** | 6.2 s |

What it actually shows (the honest version):
- **Reading: pixels beat captions.** With the gold evidence in hand, Fieldnote answered 8/8 from
  the image; the baseline answered 7/10 from captions. Captions dropped or misread the value
  the question needed (e.g. a charging chart read as 116 °F instead of 106 °F).
- **Retrieval: captions beat SigLIP on text-dense diagrams.** Captions transcribe the words
  printed inside schematics and image-only tables, so text search finds them; a single SigLIP
  vector per image does not. That is the gap the ColQwen2.5 page lane targets.
- **Most of the all-40 gap is not about vision.** Long captions win ~38% of the top-5 evidence
  slots even on text and audio questions, crowding out support-call segments. Real, but a
  property of caption pipelines, not proof that native vision is better.
- On the 15 visual-only questions the margin is small (0.53 vs 0.47). Not yet the clear win
  the project is after.

## The trap this project avoids

The easy version of "multimodal RAG" captions every image with a vision model, embeds the captions,
and runs a text pipeline. That is a text pipeline wearing a costume. Turning a chart into a caption
throws away the axis values the question needed.

Fieldnote keeps each modality in its own embedding space and lets the answer model read the actual
pixels. The caption-only pipeline is built too, as the baseline, with a captioner told to transcribe
every value it can see. That makes it a strong baseline, not a straw man.

## How it works

```
 manual PDFs ─► PyMuPDF: text blocks ─────────────────► BGE text encoder ──┐
            └─► figure/table crops + page renders ────► SigLIP image enc. ─┤
 photos ──────────────────────────────────────────────► SigLIP image enc. ─┼─► Qdrant
 call audio ──► faster-whisper, 30s windows w/ timestamps ► BGE text enc. ─┘   (named vectors
                                                                                + modality tag)
 question ─┬─► text lane  (text vector,  modality=text)  ┐
           ├─► image lane (image vector, modality=image) ├─ parallel ─► RRF ─► multimodal
           └─► audio lane (text vector,  modality=audio) ┘                     rerank (top 20→5)
                                                                                    │
                  Claude (vision) reads crops + passages, cites [S1]..[S5] ◄────────┘
                                                                                    │
            answer + citations: doc, page, cropped chart image | call @ mm:ss ◄─────┘
```

- **Late fusion.** Each modality is searched on its own; lists are merged by rank (RRF) because
  cosine scores from different encoders aren't comparable, then reranked on one scale.
- **Multimodal reranker.** A text cross-encoder can't score a chart, so the reranker is
  `jina-reranker-m0`, which scores query–text and query–image pairs with one model. RRF order is
  the fallback and the "rerank off" ablation.
- **Figure-level chunks.** Charts, diagrams and tables are detected per page (raster images,
  clusters of vector strokes, table detection) and cropped with a margin that keeps axis labels.
  The citation crop comes free, and retrieval is more precise than whole pages.
- **Citations you can check.** Manual answers come back with page number and a cropped image;
  call answers come back with a timestamp.

| Piece | Choice |
|---|---|
| PDF parsing | PyMuPDF |
| Text encoder | `BAAI/bge-base-en-v1.5` |
| Image encoder | `google/siglip-so400m-patch14-384` |
| Speech | faster-whisper |
| Vector store | Qdrant (local mode, or server via `docker compose`) |
| Reranker | `jinaai/jina-reranker-m0` |
| Answer model | `gemini-3.8-flash` (current runs) or Claude `claude-opus-5`, vision input, switchable via `FIELDNOTE_LLM_PROVIDER` |

All model choices are environment variables; see `fieldnote/config.py`.

## Run it

```bash
make install-ml            # uv sync with torch, transformers, faster-whisper, streamlit
export ANTHROPIC_API_KEY=...
make fetch                 # download manuals listed in data/manifest.yaml
make calls                 # render support-call scripts to audio (macOS say)
make ingest                # manuals, photos, calls -> artifacts/chunks.jsonl
make index                 # native multimodal index
make caption               # caption-only baseline index
make eval-native eval-caption compare
make demo                  # Streamlit UI
```

Ask one question from the CLI:

```bash
uv run --extra ml fieldnote ask "What discharge pressure should the unit show at 95F outdoor?"
```

## Tests

```bash
make test
```

The suite builds a synthetic service manual (prose, a vector pressure chart, a raster photo) and
runs ingestion, indexing, retrieval, citation mapping, the caption baseline and the eval runner end
to end. It uses hashing encoders and a scripted LLM, so CI needs no model downloads and no API key.

## Corpus

The corpus is a scaled-down version of the real use case, built from public data: manufacturer
service manuals (not committed, fetched by URL and checksum), CC-licensed and self-shot job-site
photos (licenses in `data/photos/LICENSES.csv`), and scripted support calls recorded for this
project. See [docs/PLAN.md](docs/PLAN.md) for the build plan and [eval/README.md](eval/README.md)
for the eval design.

## License

MIT for the code. Manuals belong to their manufacturers. `jina-reranker-m0` is CC-BY-NC-4.0,
which is fine for a portfolio project; swap it out for commercial use.
