.PHONY: setup lint test up down ingest goldset goldset-review sweep report reproduce dashboard update-baseline

setup:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src

test:
	uv run pytest tests/unit -q

up:
	docker compose up -d --wait

down:
	docker compose down

ingest:
	uv run ragbench corpus ingest

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
	uv run ragbench eval --update-baseline
