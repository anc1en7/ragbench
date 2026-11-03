# Keep uv downloads on the project drive (C: is space-constrained).
export UV_CACHE_DIR := $(CURDIR)/.uv_cache

.PHONY: setup lint test test-integration up down ingest index goldset goldset-review sweep report reproduce dashboard update-baseline regression-check api

setup:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src

test:
	uv run pytest tests/unit -q

test-integration:
	uv run pytest tests/integration -m integration -q

up:
	docker compose up -d --wait

down:
	docker compose down

ingest:
	uv run ragbench corpus ingest

index:
	uv run ragbench index --config configs/pipeline.example.yaml

goldset:
	uv run ragbench goldset generate

goldset-review:
	uv run ragbench goldset review

sweep:
	uv run ragbench sweep --grid configs/grid.yaml

report:
	uv run ragbench report

reproduce:
	uv run ragbench reproduce

dashboard:
	uv run streamlit run dashboard/app.py

update-baseline:
	uv run ragbench eval --config configs/regression_mini.yaml --split test --results-dir .tmp/regression --update-baseline

regression-check:
	uv run ragbench regression-check

api:
	uv run uvicorn ragbench.api.app:app --port 8000
