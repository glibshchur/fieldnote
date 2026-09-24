"""Command line: fieldnote fetch | ingest | index | caption | ask | eval | compare."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from fieldnote.config import get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False)

CHUNKS_FILE = "chunks.jsonl"


@app.command()
def fetch(manifest: Path = Path("data/manifest.yaml")) -> None:
    """Download manuals listed in the manifest and verify checksums."""
    from fieldnote.fetch import fetch_manifest

    for line in fetch_manifest(manifest, get_settings().data_dir / "manuals"):
        typer.echo(line)


@app.command()
def ingest(no_pages: bool = False) -> None:
    """Parse manuals, photos and calls into chunks (artifacts/chunks.jsonl)."""
    from collections import Counter

    from fieldnote.ingest import PdfIngestOptions, ingest_corpus, save_chunks

    s = get_settings()
    chunks = ingest_corpus(s, pdf_opts=PdfIngestOptions(index_pages=not no_pages))
    save_chunks(chunks, s.artifacts_dir / CHUNKS_FILE)
    typer.echo(f"{len(chunks)} chunks: {dict(Counter(c.source_type for c in chunks))}")


@app.command()
def index() -> None:
    """Embed chunks with their modality's encoder and build the native index."""
    from fieldnote.encoders.image import SiglipEncoder
    from fieldnote.encoders.text import SentenceTransformerEncoder
    from fieldnote.index import ChunkStore, build_index
    from fieldnote.index.store import make_client
    from fieldnote.ingest import load_chunks

    s = get_settings()
    chunks = load_chunks(s.artifacts_dir / CHUNKS_FILE)
    store = ChunkStore(make_client(s.qdrant_path, s.qdrant_url), s.collection)
    counts = build_index(
        store,
        chunks,
        SentenceTransformerEncoder(s.text_model, s.device),
        SiglipEncoder(s.image_model, s.device),
    )
    typer.echo(f"indexed {counts} into '{s.collection}'")


@app.command()
def caption(workers: int = 8) -> None:
    """Build the caption-only baseline index (captions every image with a VLM)."""
    from fieldnote.baseline.caption import CaptionCache, build_caption_index, caption_chunks
    from fieldnote.encoders.text import SentenceTransformerEncoder
    from fieldnote.index import ChunkStore
    from fieldnote.index.store import make_client
    from fieldnote.ingest import load_chunks
    from fieldnote.llm import make_llm
    from fieldnote.pipeline import collection_for

    s = get_settings()
    chunks = load_chunks(s.artifacts_dir / CHUNKS_FILE)
    cache = CaptionCache(s.artifacts_dir / "captions.jsonl")
    captioned = caption_chunks(
        chunks, make_llm(s.llm_provider, s.caption_model), s.caption_model, cache, workers
    )
    store = ChunkStore(make_client(s.qdrant_path, s.qdrant_url), collection_for(s, "caption"))
    n = build_caption_index(store, captioned, SentenceTransformerEncoder(s.text_model, s.device))
    typer.echo(f"indexed {n} chunks into '{store.collection}'")


@app.command()
def ask(
    question: str,
    mode: str = typer.Option("native", help="native | caption"),
    reranker: str | None = typer.Option(None, help="jina | rrf"),
) -> None:
    """Answer one question and print citations and timings."""
    from fieldnote.pipeline import load_pipeline

    ans = load_pipeline(get_settings(), mode, reranker).ask(question)  # type: ignore[arg-type]
    typer.echo(ans.text + "\n")
    for c in ans.citations:
        typer.echo(f"  [{c.sid}] {c.label}  {c.image_path or ''}")
    typer.echo("\n" + json.dumps({k: round(v, 2) for k, v in ans.timings.items()}))


@app.command("eval")
def eval_cmd(
    questions: Path = Path("eval/questions.jsonl"),
    mode: str = typer.Option("native", help="native | caption"),
    reranker: str | None = typer.Option(None, help="jina | rrf"),
    name: str | None = typer.Option(None, help="label for the results file"),
    repeats: int = typer.Option(1, help="extra passes for latency percentiles"),
    no_judge: bool = False,
) -> None:
    """Run the eval set and write eval/results/<stamp>_<name>.json."""
    from fieldnote.evaluation.runner import load_questions, run_eval
    from fieldnote.llm import make_llm
    from fieldnote.pipeline import load_pipeline

    s = get_settings()
    pipe = load_pipeline(s, mode, reranker)  # type: ignore[arg-type]
    judge = None if no_judge else make_llm(s.llm_provider, s.judge_model)
    label = name or f"{mode}-{reranker or s.reranker}"
    summary, out = run_eval(
        pipe, load_questions(questions), judge, Path("eval/results"), label, repeats
    )
    typer.echo(json.dumps(summary["visual_only"] | {"latency": summary["latency"]}, indent=2))
    typer.echo(f"wrote {out}")


@app.command()
def rescore(results: Path, questions: Path = Path("eval/questions.jsonl")) -> None:
    """Re-apply deterministic scorers to a saved results file (no LLM calls)."""
    from fieldnote.evaluation.runner import load_questions
    from fieldnote.evaluation.runner import rescore as do_rescore

    summary, out = do_rescore(results, load_questions(questions))
    data = json.loads(out.read_text())
    for line in data["rescored"]["changed"]:
        typer.echo(line)
    typer.echo(json.dumps({k: round(v, 3) for k, v in summary["all"].items()}))
    typer.echo(f"wrote {out}")


@app.command()
def compare(results: list[Path]) -> None:
    """Print the headline markdown table from two or more results files."""
    from fieldnote.evaluation.runner import results_table

    loaded = {}
    for p in results:
        data = json.loads(p.read_text())
        loaded[data["system"]] = data
    k_total = int(next(iter(loaded.values()))["config"]["k_total"])
    typer.echo(results_table(loaded, k_total))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("fieldnote.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
