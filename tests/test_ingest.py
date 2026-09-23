from __future__ import annotations

from pathlib import Path

from PIL import Image

from fieldnote.ingest import load_chunks, save_chunks
from fieldnote.ingest.audio import ingest_calls, window_segments
from fieldnote.ingest.pdf import ingest_pdf
from fieldnote.ingest.photos import ingest_photos


def test_pdf_yields_text_figure_and_page_chunks(manual_pdf: Path, settings):
    chunks = ingest_pdf(manual_pdf, settings.crops_dir, settings.pages_dir)
    by_type: dict[str, list] = {}
    for c in chunks:
        by_type.setdefault(c.source_type, []).append(c)

    text = by_type["manual_text"]
    assert any("condenser airflow" in c.text for c in text)
    assert all(c.page and c.bbox for c in text)

    figures = by_type["manual_figure"]
    pages_with_figures = {c.page for c in figures}
    assert pages_with_figures == {1, 2}  # vector chart on p1, raster image on p2
    for f in figures:
        assert Path(f.image_path).exists()
        assert f.modality == "image"
    # The crop margin should pull the axis title into the figure crop.
    chart = next(f for f in figures if f.page == 1)
    assert "Discharge pressure" in chart.text or "Outdoor ambient" in chart.text

    assert {c.page for c in by_type["manual_page"]} == {1, 2}


def test_chunk_ids_are_stable(manual_pdf: Path, settings):
    a = ingest_pdf(manual_pdf, settings.crops_dir, settings.pages_dir)
    b = ingest_pdf(manual_pdf, settings.crops_dir, settings.pages_dir)
    assert [c.id for c in a] == [c.id for c in b]


def test_chunks_roundtrip(manual_pdf: Path, settings, tmp_path: Path):
    chunks = ingest_pdf(manual_pdf, settings.crops_dir, settings.pages_dir)
    save_chunks(chunks, tmp_path / "chunks.jsonl")
    loaded = load_chunks(tmp_path / "chunks.jsonl")
    assert [c.id for c in loaded] == [c.id for c in chunks]
    assert loaded[0].bbox == chunks[0].bbox


def test_photos_are_resized_and_keep_notes(tmp_path: Path):
    photos = tmp_path / "photos"
    photos.mkdir()
    Image.new("RGB", (4000, 3000), "gray").save(photos / "job_0142.jpg")
    (photos / "metadata.csv").write_text("file,notes\njob_0142.jpg,burnt contactor unit 7\n")
    chunks = ingest_photos(photos, tmp_path / "out")
    assert len(chunks) == 1
    assert chunks[0].text == "burnt contactor unit 7"
    with Image.open(chunks[0].image_path) as im:
        assert max(im.size) <= 1600


def test_window_segments_overlap_and_cover():
    segs = [(i * 5.0, i * 5.0 + 5.0, f"s{i}") for i in range(20)]  # 100 seconds
    windows = window_segments(segs, window_s=30, stride_s=20)
    assert windows[0][0] == 0.0 and windows[-1][1] == 100.0
    assert all(w[1] - w[0] <= 30 for w in windows)
    assert windows[1][0] == 20.0  # 10s overlap


def test_calls_use_transcriber_and_cache(tmp_path: Path):
    calls = tmp_path / "calls"
    calls.mkdir()
    (calls / "call_01.wav").write_bytes(b"")
    n_calls = []

    def fake(path):
        n_calls.append(path)
        return [(0.0, 12.0, "Check superheat before replacing the TXV."), (12.0, 40.0, "Yes.")]

    chunks = ingest_calls(calls, tmp_path / "cache", fake)
    chunks_again = ingest_calls(calls, tmp_path / "cache", fake)
    assert len(n_calls) == 1  # second run reads the cached transcript
    assert chunks[0].modality == "audio" and chunks[0].start_s == 0.0
    assert "superheat" in chunks[0].text
    assert [c.id for c in chunks] == [c.id for c in chunks_again]
