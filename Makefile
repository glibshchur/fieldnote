.PHONY: install install-ml test lint fetch calls ingest index index-pages caption eval-native eval-caption compare serve demo qdrant

install:
	uv sync

install-ml:
	uv sync --extra ml --extra app

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

fetch:
	uv run fieldnote fetch

# macOS only: render the support-call scripts to audio with `say`
calls:
	uv run python scripts/tts_calls.py

ingest:
	uv run --extra ml fieldnote ingest

index:
	uv run --extra ml fieldnote index

index-pages:
	uv run --extra ml fieldnote index-pages

caption:
	uv run --extra ml fieldnote caption

eval-native:
	uv run --extra ml fieldnote eval --mode native --repeats 3

eval-caption:
	uv run --extra ml fieldnote eval --mode caption --repeats 3

compare:
	uv run fieldnote compare $$(ls -t eval/results/*_native-*.json | head -1) $$(ls -t eval/results/*_caption-*.json | head -1)

serve:
	uv run --extra ml fieldnote serve

demo:
	uv run --extra ml --extra app streamlit run app/streamlit_app.py

# Optional: Qdrant server instead of local mode (then export QDRANT_URL=http://localhost:6333)
qdrant:
	docker compose up -d qdrant
