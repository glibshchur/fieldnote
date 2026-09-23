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


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([], 95) == 0.0
