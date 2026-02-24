# Repository Guidelines

## Project Structure & Module Organization
Core runtime modules are in `jobs/`, `alerts/`, `ai/`, `notify/`, and `app/`.
- `jobs/`: entry points such as `ingest`, `review`, and `backtest`.
- `alerts/`: signal rules and risk scoring engine.
- `app/streamlit_app.py`: dashboard UI.
- `app/providers/store/`: SQLite persistence layer.
- `tests/`: pytest suite (currently smoke tests).
- `data/`: runtime artifacts (`alerts.jsonl`, cache files, backtest outputs). Treat as generated state, not source.

## Build, Test, and Development Commands
Use `uv` as the default workflow.
- `uv sync`: install dependencies and create/update `.venv`.
- `uv run python -m jobs.ingest`: fetch market data and evaluate alerts.
- `uv run streamlit run app/streamlit_app.py`: run dashboard locally.
- `uv run python -m jobs.review`: run weekly review/evolution job.
- `uv run python -m jobs.backtest COIN`: run backtest for a ticker.
- `uv run python -m pytest`: run all tests (`-q` configured in `pyproject.toml`).

## Coding Style & Naming Conventions
Follow Python conventions already used in the codebase:
- 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes/dataclasses.
- Keep module boundaries clear: ingest logic in `jobs/`, rule logic in `alerts/`, storage in `app/providers/store/`.
- Prefer type hints for public functions and dataclasses.
- Keep comments brief and only where logic is non-obvious.

## Testing Guidelines
Testing uses `pytest` with tests discovered from `tests/`.
- Name test files as `test_*.py` and test functions as `test_*`.
- Add tests for new data transformations, alert rules, and DB path/config behavior.
- Run targeted tests during development, e.g. `uv run python -m pytest tests/test_smoke.py`.

## Commit & Pull Request Guidelines
Recent commits use short prefix-based subjects (e.g., `Feat: ...`, `Cleanup: ...`, `Add ...`, `Tighten ...`), often in imperative style. Keep commits focused and descriptive.

For PRs:
- include a clear summary, scope, and risk/rollback notes;
- link related issues/tasks;
- attach screenshots for Streamlit UI changes;
- confirm tests pass (`uv run python -m pytest`) before requesting review.

## Security & Configuration Tips
Do not commit secrets. Use `.env` (from `.env.example`) for local overrides, especially `WATCHTOWER_DB_PATH`. Keep `config.yaml` defaults safe and environment-agnostic.
