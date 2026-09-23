from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from fieldnote.baseline.caption import CaptionCache, build_caption_index, caption_chunks
from fieldnote.encoders.fake import HashingImageEncoder, HashingTextEncoder
from fieldnote.evaluation.runner import load_questions, results_table, run_eval
from fieldnote.index import ChunkStore, build_index
from fieldnote.index.store import make_client
from fieldnote.llm import ScriptedLLM
from fieldnote.pipeline import Fieldnote
from fieldnote.retrieve.fuse import rrf_fuse
from fieldnote.retrieve.rerank import RRFOrder
from fieldnote.retrieve.search import ParallelRetriever, caption_lanes, native_lanes
from fieldnote.types import Candidate, Chunk


def _corpus(tmp_path: Path) -> list[Chunk]:
    img = tmp_path / "discharge_pressure_ambient_chart.png"
    Image.new("RGB", (64, 64), "white").save(img)
    photo = tmp_path / "burnt_contactor_photo.png"
    Image.new("RGB", (64, 64), "black").save(photo)
    return [
        Chunk("text", "manual_text", "acme.pdf", text="Clean the condenser coil.", page=10),
        Chunk("image", "manual_figure", "acme.pdf", page=112, image_path=str(img)),
        Chunk("image", "photo", "job_0142.jpg", image_path=str(photo)),
        Chunk(
            "audio",
            "call",
            "call_03.wav",
            text="support said check superheat first",
            start_s=180.0,
            end_s=210.0,
        ),
    ]


def _pipeline(settings, chunks, llm, mode="native") -> Fieldnote:
    store = ChunkStore(make_client(":memory:"), "t")
    text_enc, image_enc = HashingTextEncoder(), HashingImageEncoder()
    if mode == "native":
        build_index(store, chunks, text_enc, image_enc)
        lanes = native_lanes(text_enc, image_enc, 2, 2, 1)
    else:
        build_caption_index(store, chunks, text_enc)
        lanes = caption_lanes(text_enc, 2, 2, 1)
    return Fieldnote(settings, ParallelRetriever(store, lanes), RRFOrder(), llm, mode)


def test_rrf_merges_duplicates_across_lanes():
    c = Chunk("text", "manual_text", "a.pdf", text="x", page=1)
    d = Chunk("image", "manual_page", "a.pdf", page=1)
    fused = rrf_fuse(
        {
            "text": [Candidate(c, 0.9, "text", rank=1)],
            "image": [Candidate(d, 0.2, "image", rank=1), Candidate(c, 0.1, "image", rank=2)],
        }
    )
    assert fused[0].chunk.id == c.id  # found by two lanes beats found by one
    assert len(fused) == 2


def test_each_lane_only_sees_its_modality(settings, tmp_path):
    p = _pipeline(settings, _corpus(tmp_path), ScriptedLLM([]))
    lanes = p.retriever.retrieve("discharge pressure chart")
    assert all(c.chunk.modality == "image" for c in lanes["image"])
    assert all(c.chunk.modality == "text" for c in lanes["text"])
    assert all(c.chunk.modality == "audio" for c in lanes["audio"])


def test_native_mode_sends_pixels_and_maps_citations(settings, tmp_path):
    llm = ScriptedLLM(["About 118 psig at 95F [S1]."])
    ans = _pipeline(settings, _corpus(tmp_path), llm).ask("discharge pressure ambient chart")
    _system, content = llm.calls[0]
    assert any(b["type"] == "image" for b in content)
    assert ans.citations and ans.citations[0].sid == "S1"
    assert ans.citations[0].candidate is ans.evidence[0]
    assert set(ans.timings) >= {"search_s", "rerank_s", "answer_s", "total_s"}


def test_caption_mode_never_sends_images(settings, tmp_path):
    chunks = _corpus(tmp_path)
    cap_llm = ScriptedLLM(["Chart of discharge pressure vs ambient.", "Burnt contactor."])
    captioned = caption_chunks(chunks, cap_llm, "m", CaptionCache(tmp_path / "c.jsonl"), workers=1)
    assert all(c.text for c in captioned if c.modality == "image")

    llm = ScriptedLLM(["Not found in the provided sources."])
    _pipeline(settings, captioned, llm, mode="caption").ask("discharge pressure")
    _system, content = llm.calls[0]
    assert not any(b["type"] == "image" for b in content)


def test_caption_cache_skips_repeat_calls(tmp_path):
    chunks = _corpus(tmp_path)
    cache_path = tmp_path / "c.jsonl"
    caption_chunks(chunks, ScriptedLLM(["a", "b"]), "m", CaptionCache(cache_path), workers=1)
    second = ScriptedLLM([])
    caption_chunks(chunks, second, "m", CaptionCache(cache_path), workers=1)
    assert second.calls == []


def test_eval_end_to_end(settings, tmp_path):
    questions = tmp_path / "q.jsonl"
    questions.write_text(
        "\n".join(
            json.dumps(q)
            for q in [
                {
                    "id": "q1",
                    "question": "discharge pressure ambient chart value at 95F?",
                    "category": "visual_chart",
                    "gold_sources": [{"doc": "acme.pdf", "page": 112}],
                    "gold_answer": "118 psig",
                    "answer_type": "numeric",
                    "gold_value": 118,
                    "tolerance": 5,
                },
                {
                    "id": "q2",
                    "question": "what did support say about superheat?",
                    "category": "audio",
                    "gold_sources": [{"call": "call_03.wav", "t": 190}],
                    "gold_answer": "not in corpus",
                    "answer_type": "abstain",
                },
            ]
        )
    )
    llm = ScriptedLLM(["About 120 psig [S1].", "Not found in the provided sources."])
    pipe = _pipeline(settings, _corpus(tmp_path), llm)
    summary, out = run_eval(pipe, load_questions(questions), None, tmp_path / "res", "native")

    assert summary["all"]["n"] == 2 and summary["visual_only"]["n"] == 1
    assert summary["visual_only"]["answer_acc"] == 1.0
    assert summary["all"]["answer_acc"] == 1.0
    assert summary["latency"]["p95_s"] >= 0
    table = results_table({"native": json.loads(out.read_text())}, k_pool=5)
    assert "| native | visual-only | 1 |" in table
