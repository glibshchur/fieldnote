# Error analysis

Five visual-only questions that still fail on the published k=20 system
(`eval/results/20260925-203047_native-k20.json`, reranker at 640 px) and on the
caption-only baseline (`eval/results/20260924-231557_caption-jina-expand.json`).
The ColQwen page-lane ablation is `eval/results/20260925-204221_native-pages-crops.json`.

A miss is a retrieval miss when the gold page never reaches the answer model, and a
reading miss when an image of that page is in the evidence and the number is still wrong.

## q09 — Trane 4TWR7 fan-motor leads (retrieval)

The question asks where the brown, purple, and black leads land. Gold is page 2 of
`trane_4twr7_service_facts.pdf`. Neither Fieldnote nor the caption baseline retrieves
that page. The answer abstains. The diagram is a small service-facts schematic with
almost no prose around the terminals, so the text lane has nothing to match and one
SigLIP vector does not land on the right crop.

## q11 — Carrier 48TC control-transformer breaker (retrieval)

Gold is 3.2 A on page 113 of `carrier_48tc_04-16_service.pdf`. Not in the top 20 for
either system. The rating sits in a wiring-diagram callout, not in a sentence, and
nearby Carrier pages about other sizes outrank it.

## q15 — York ZF low-pressure switch (retrieval)

Gold is open at 50 psig / close at 71 psig on page 41 of
`york_zf_7.5-12.5_techguide.pdf`. Not retrieved. The model cites a component diagram
on page 2 of the same manual and abstains. The cut-in and cut-out values are drawn on
the switch detail, and the text layer of that page does not repeat them.

## q04 — Trane Voyager 30-ton discharge pressure (retrieval, then a misread)

Gold is about 417 psig on page 102 of `trane_voyager_27-50_iom.pdf`. The k=20 system
does not retrieve the page. The page lane does (rank 1) and still answers wrong: the
chart is a full-page performance table, and the model reads a neighboring cell. Getting
the page into the prompt is not the same as reading the right intersection.

## q06 — Sporlan selective charges, static superheat (reading)

Gold is the liquid-charge curve at 30°F evaporator on page 7 of
`sporlan_bulletin_10-9.pdf`. Rank 1 for Fieldnote, the page lane, and the caption
baseline, and all three miss the number. The chart has several curves close together.
The model names the right figure and then reports a neighboring curve's superheat.
This is the case the project is about: the evidence is the picture, and both pixels
and a careful caption can still pick the wrong line.

## What would move these

q09, q11, and q15 need a retriever that matches a few terminal names or a pressure
value inside a drawing. The page lane raises visual recall@5 from 0.53 to 0.73 and
picks up q01 and q12, but visual answer accuracy stays 0.47 because other charts
(q08, q14) then get misread, and p95 goes from 6.6 s to 8.1 s. It is an ablation,
not the published config.
