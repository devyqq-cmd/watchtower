# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup（uv 管理环境）**
```bash
# 安装 uv（首次）
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # 或重启终端

# 安装依赖（自动创建 .venv + 生成 uv.lock）
uv sync
```

**运行**
```bash
uv run python -m jobs.ingest                    # 拉数据 + 评估信号
uv run streamlit run app/streamlit_app.py       # 启动看板
uv run python -m jobs.review                    # 周度复盘 / 自进化
uv run python -m jobs.backtest COIN             # 回测（默认 0700.HK）
```

**测试**
```bash
uv run python -m pytest                         # 全部测试
uv run python -m pytest tests/test_smoke.py     # 单文件
uv run python -m pytest -k test_db_init_tmp     # 单个用例
```

**包管理**
```bash
uv add requests          # 添加运行时依赖
uv add --dev ruff        # 添加开发依赖
uv remove requests       # 移除依赖
```

**环境变量**
```bash
export WATCHTOWER_DB_PATH=/path/to/custom.db   # 覆盖 config.yaml 的 db_path
```

## Architecture

Watchtower is a market monitoring system that fetches daily OHLCV data, stores it in SQLite, scores risk using technical indicators + fundamentals, emits alerts, and exposes a Streamlit dashboard with backtesting.

### Data flow

```
jobs/ingest.py
  ├── fetch_ticker()         # yfinance (primary), with 10-min parquet cache under data/yf_cache/
  ├── fetch_ticker_stooq_daily()  # Stooq fallback when yfinance returns empty (daily only)
  ├── fetch_fundamental_valuation()  # yfinance .info for P/E, P/B, dividend yield
  ├── SQLiteStore.upsert_prices()    # INSERT OR IGNORE into prices table (dedup key: ticker+ts)
  └── evaluate_alerts()      # builds alert list → appended to data/alerts.jsonl
```

### Signal types

| Severity | Rule ID | Direction | Trigger |
|----------|---------|-----------|---------|
| `high` | `RISK_EXTREME` | 卖出 | risk_score ≥ 80 |
| `med` | `RISK_HIGH` | 卖出 | risk_score ≥ 60 |
| `buy` | `RISK_LOW` | 买入 | risk_score ≤ 25 |
| `buy` | `OVERSOLD_OPP` | 买入 | RSI < 35 AND z_dist < −1.5（独立于评分，专为长线抄底设计） |
| `high` | `EVENT_SHOCK` | 卖出 | vol_z ≥ 3 AND rv spike |

### Alert / risk-scoring pipeline (`alerts/`)

- **`alerts/rules.py` → `AlertConfig`**: frozen dataclass holding all thresholds and factor weights. Weights can be overridden at runtime from `data/evolution.json` via `AlertConfig.load_evolution()`.
- **`alerts/engine.py`**:
  - `compute_features(df, cfg)` – adds EMA-fast/slow, RSI-14, ATR-14, price Z-score from EMA-slow, volume Z-score.
  - `calculate_risk_score(last, cfg, global_vix, valuation)` – weighted sum of 5 factors (trend, overextension, momentum, volatility, valuation) + VIX modifier → 0–100 score.
  - `evaluate_alerts(symbol, df, cfg, ...)` – requires `min_bars` rows (default 100); emits `RISK_EXTREME` (≥80), `RISK_HIGH` (≥60), or `EVENT_SHOCK` (vol+volatility spike); cooldown state stored in `data/alert_state.json`.
  - Each alert is enriched with an AI narrative from `ai/analyst.py`.

### Self-evolution (`jobs/review.py`)

Runs weekly. Reads `data/alerts.jsonl`, looks up 5-day price outcomes for each high-risk alert (5–30 days old), computes win/false-positive rate, and adjusts `AlertConfig` weights heuristically. Persists new weights to `data/evolution.json`.

### Storage

- **SQLite** (`watchtower.db` by default, overridable via `WATCHTOWER_DB_PATH`): single `prices` table, schema in `app/providers/store/sqlite_store.py:SQLiteStore`.
- **`data/alerts.jsonl`** – append-only alert log.
- **`data/alert_state.json`** – per-symbol/rule cooldown timestamps.
- **`data/evolution.json`** – evolved weights persisted across runs.
- **`data/yf_cache/*.parquet`** – 10-minute yfinance download cache (keyed `{ticker}_{interval}_{days}d.parquet`).

### Dashboard (`app/streamlit_app.py`)

Two tabs: live monitoring (price chart, OHLCV table, stats) and AI backtest simulation. Backtest results written to `data/backtest_results.csv` and visualised as equity curves.

### AI narrative (`ai/analyst.py → AINarrativeAnalyst`)

Falls back to a rule-based engine when `GEMINI_API_KEY` is not set. The LLM path is stubbed; the fallback covers greed-trap and panic-selling scenarios.

### Configuration (`config.yaml`)

Key fields: `tickers`, `interval` (currently `"1d"`), `days` (lookback for ingest), `db_path`. Ingest always fetches at least 365 days regardless of `days` when interval is daily (needed for EMA-200 and Z-score lookback).
