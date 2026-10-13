from fieldnote.answer.synthesize import extract_citations
from fieldnote.evaluation.scorers import (
    abstained,
    first_relevant_rank,
    matches_gold,
    numeric_correct,
    percentile,
    reciprocal_rank,
)
from fieldnote.types import Chunk


def test_gold_matching_by_page_photo_and_call_time():
    page = Chunk("image", "manual_figure", "m.pdf", page=112)
    photo = Chunk("image", "photo", "job_0142.jpg")
    call = Chunk("audio", "call", "c.wav", text="x", start_s=180, end_s=210)
    assert matches_gold(page, {"doc": "m.pdf", "page": 112})
    assert not matches_gold(page, {"doc": "m.pdf", "page": 111})
    assert matches_gold(photo, {"photo": "job_0142.jpg"})
    assert matches_gold(call, {"call": "c.wav", "t": 194})
    assert not matches_gold(call, {"call": "c.wav", "t": 250})


def test_rank_and_mrr():
    chunks = [Chunk("text", "manual_text", "m.pdf", text="a", page=p) for p in (1, 2, 3)]
    rank = first_relevant_rank(chunks, [{"doc": "m.pdf", "page": 3}])
    assert rank == 3 and abs(reciprocal_rank(rank) - 1 / 3) < 1e-9
    assert first_relevant_rank(chunks, [{"doc": "m.pdf", "page": 9}]) is None


def test_numeric_and_abstain():
    assert numeric_correct("Head pressure should be about 140 psig", 118, 5) is False
    assert numeric_correct("Head pressure should be about 121 psig [S1]", 118, 5)
    assert abstained("Not found in the provided sources. Check the unit's data plate.")


def test_grouped_citations_are_all_kept():
    text = "About 105°F [S4, S6]. The ZP*KC envelope is Figure 1 [S4]."
    assert extract_citations(text, n_evidence=7) == ["S4", "S6"]
    assert extract_citations("See [S2] and [S9].", n_evidence=5) == ["S2"]
    assert extract_citations("No source named.", n_evidence=5) == []


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([], 95) == 0.0


def test_numeric_ignores_references_and_abstentions():
    # Regression: "Not found ... page 7" was scored correct against gold 7.5 ± 1.5.
    assert not numeric_correct(
        "Not found in the provided sources. See Figure 6 on page 7 [S1].", 7.5, 1.5
    )
    assert not numeric_correct("Per Fig. 22 [S1, S2] on p.106 the value is unclear", 106, 3)
    assert numeric_correct("About 7.5°F superheat at 30°F evaporator [S1] (Figure 6).", 7.5, 1.5)
