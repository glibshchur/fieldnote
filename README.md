# Fieldnote

**Multimodal RAG for HVAC field technicians.** Fieldnote answers questions whose evidence exists
only inside a picture: a pressure curve on page 112 of a service manual, a wiring diagram, a photo
of a failed contactor. It also searches recorded manufacturer support calls.

> A tech on a roof asks why unit 7 keeps tripping on high pressure. The answer lives in a chart,
> not in a paragraph.

**Status:** pipeline, baseline and eval harness are built and tested. The corpus and the
40-question eval set come next. The results table below gets filled from real runs, never by hand.

## Results

| System | Slice | Recall@5 | Answer acc. | Citation acc. | p95 latency |
|---|---|---|---|---|---|
| Caption-only baseline | 15 visual-only questions | pending | pending | pending | pending |
| **Fieldnote** | 15 visual-only questions | pending | pending | pending | pending |
| Caption-only baseline | all 40 | pending | pending | pending | pending |
| **Fieldnote** | all 40 | pending | pending | pending | pending |

Generated with `make compare` from files in `eval/results/`.

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
| Answer model | Claude (`claude-opus-5`, low effort for latency), vision input |

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
