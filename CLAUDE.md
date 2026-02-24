# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

当需要使用 evomap skill 时，请读取 .claude/skills/evomap-skill.md

## Commands

**Environment manager: `uv` (not pip/venv directly)**

```bash
uv sync                          # install all dependencies
uv run python -m jobs.ingest     # fetch data + evaluate alerts
uv run streamlit run app/streamlit_app.py  # launch dashboard
uv run python -m jobs.backtest   # backtest (default: 0700.HK)
uv run python -m jobs.walk_forward 0700.HK # walk-forward validation
uv run python -m jobs.param_sweep 0700.HK  # parameter sensitivity sweep
uv run python -m jobs.review     # weekly signal quality + weight evolution
uv run python -m pytest          # run all tests
uv run python -m pytest tests/test_alerts_p0.py  # run single test file
uv add <pkg>                     # add runtime dependency
uv add --dev <pkg>               # add dev dependency
```

## Architecture

### Data Flow

`jobs/ingest.py` is the main entry point: fetch → store → alert eval → notify.
1. Fetches `^VIX` first as global market context
2. For each ticker: fetches OHLCV via yfinance (`fetch_ticker`) with 10-min parquet cache in `data/yf_cache/`; falls back to stooq HTTP CSV if yfinance returns empty
3. Upserts rows into `prices` table (SQLite, unique on `ticker, ts`)
4. Runs `evaluate_alerts()` which emits alerts to `data/alerts.jsonl`
5. Sends triggered alerts to Telegram if env vars are set

### Alert Engine (`alerts/`)

- **`alerts/rules.py`** — `AlertConfig` frozen dataclass holding all tunable parameters (EMA periods, RSI, VIX thresholds, factor weights). `AlertConfig.load_evolution()` merges overrides from `data/evolution.json` at runtime.
- **`alerts/engine.py`** — three key functions:
  - `compute_features(df, cfg)`: adds EMA50/200, Wilder RSI (matches TradingView), ATR, price Z-score vs EMA200, volume Z-score
  - `calculate_risk_score(last, cfg, vix, valuation)`: weighted 0–100 composite of trend + overextension + momentum + volatility + fundamental valuation, modified by VIX
  - `evaluate_alerts(symbol, df, cfg, vix, valuation)`: fires `RISK_EXTREME/HIGH/LOW`, `OVERSOLD_OPP`, `EVENT_SHOCK` rules with cooldown dedup via `data/alert_state.json`; each alert gets an AI narrative via `AINarrativeAnalyst`

### Backtest Layer (`jobs/backtest.py`)

`BacktestEngine` is vectorized. Design invariant: signals generated at bar `t` take effect at `t+1` (no look-ahead bias). Position logic:
- `score >= score_sell_hard` → `pos_high_risk` (default 20%)
- `score >= score_sell_soft` → `pos_mid_risk` (default 50%)
- bear market signal (EMA50 < EMA200 AND price < EMA200) → caps position at `pos_bear_market` (default 30%)

`run_split(is_ratio=0.75)` returns IS and OOS metrics. `run_backtest()` is the CLI entry.

### Validation Jobs

- **`jobs/walk_forward.py`** — `WalkForwardRunner`: rolls a 252-bar train + 63-bar test window (step 21 bars) across full history; robustness verdict requires mean OOS Sharpe > 0.3 and ≥70% positive windows
- **`jobs/param_sweep.py`** — `ParamSweeper`: grid over `score_sell_soft × score_sell_hard`; stability requires >60% of combos with positive OOS Sharpe

### Self-Evolution (`jobs/review.py`)

Weekly job that reads `data/alerts.jsonl`, measures signal outcomes at 20 trading days forward (threshold: ±5%), adjusts `AlertConfig.weights` (stored in `data/evolution.json`): high sell false-positive rate raises `valuation` weight, lowers `momentum`; high sell win rate raises `trend` weight.

### AI Narratives (`ai/analyst.py`)

`AINarrativeAnalyst.analyze_risk_context()` tries in priority order: claude CLI (`claude -p`) → MiniMax API (`MINIMAX_API_KEY`) → Anthropic API (`ANTHROPIC_API_KEY`) → rule-based fallback. All paths produce a ≤200-char Chinese narrative attached to alert messages.

### Storage & Config

- **SQLite** (`watchtower.db` or `WATCHTOWER_DB_PATH`): single `prices` table via `app/providers/store/sqlite_store.py`
- **`config.yaml`**: `tickers`, `interval`, `days`, `db_path`
- **`app/providers/analytics/performance.py`**: `sharpe_ratio`, `max_drawdown`, `calmar_ratio`, `t_stat_alpha`, `summarize` — shared across backtest and walk-forward

### Environment Variables

| Variable | Purpose |
|---|---|
| `WATCHTOWER_DB_PATH` | Override SQLite path |
| `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` | Push alert notifications |
| `MINIMAX_API_KEY` | MiniMax LLM for narratives |
| `ANTHROPIC_API_KEY` | Anthropic Claude for narratives |
