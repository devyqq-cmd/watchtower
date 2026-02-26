# Daily Review Feature Design
Date: 2026-02-26

## Overview

Add a daily market recap report delivered to Telegram after Hong Kong market close (16:30 HKT = 08:30 UTC). Report covers per-ticker metrics, VIX sentiment, today's triggered alerts, latest news, and an AI-generated market narrative.

---

## Requirements

- **Trigger**: Weekdays at 16:30 HKT (08:30 UTC) via cron
- **Delivery**: Telegram (existing bot infrastructure)
- **Content**:
  1. Per-ticker: close price, daily % change, risk score, RSI, EMA trend state
  2. VIX level with sentiment label
  3. Latest 3 news headlines per ticker (from `yf.Ticker.news`)
  4. Today's triggered alerts (from `data/alerts.jsonl`)
  5. AI-generated macro narrative (≤200 chars, Chinese)
- **News source**: `yf.Ticker(sym).news` — already available via yfinance, no new API key needed
- **Scheduling**: cron (user will add crontab entry manually)
- **AI narrative**: Reuses `AINarrativeAnalyst` with a new `generate_market_narrative()` method

---

## Architecture

### New Files

```
jobs/daily_review.py          # Main logic and CLI entry point
```

### Modified Files

```
notify/telegram.py            # +send_daily_report(text: str) -> bool
ai/analyst.py                 # +generate_market_narrative(ctx: dict) -> str
```

### Data Flow

```
jobs/daily_review.py
  ├── load_config()                     [jobs/ingest.py — reused]
  ├── fetch_ticker(sym, "1d", 365)      [jobs/ingest.py — reused; 10min parquet cache]
  ├── compute_features(df, cfg)         [alerts/engine.py — reused]
  ├── calculate_risk_score(last, cfg)   [alerts/engine.py — reused]
  ├── yf.Ticker(sym).news[:5]           [new: fetch news headlines]
  ├── read data/alerts.jsonl            [filter for today's UTC date]
  ├── AINarrativeAnalyst
  │     .generate_market_narrative(ctx) [new method: macro narrative]
  └── send_daily_report(text)           [notify/telegram.py — new function]
```

---

## Implementation Details

### `jobs/daily_review.py`

**Public entry point**: `run_daily_review()` — called by `python -m jobs.daily_review`

**Internal functions**:

| Function | Responsibility |
|---|---|
| `_fetch_vix() -> float` | Fetch latest VIX close (reuse `fetch_ticker("^VIX", "1d", 5)`) |
| `_fetch_news(sym: str) -> list[str]` | `yf.Ticker(sym).news`, return top 3 titles sorted by `providerPublishTime` desc |
| `_get_today_alerts(date: str) -> list[dict]` | Read `data/alerts.jsonl`, filter by today's UTC date |
| `_vix_label(vix: float) -> str` | Rule-based sentiment: <15 "极度乐观", <20 "偏乐观", <25 "中性", <30 "偏恐慌", ≥30 "极度恐慌" |
| `_ema_status(last: pd.Series) -> str` | Compare close/ema_fast/ema_slow → "多头排列"/"空头排列"/"震荡" |
| `_risk_emoji(score: float) -> str` | <35→🟢, <60→🟡, ≥60→🔴 |
| `_format_report(...) -> str` | Assemble final Telegram message string |
| `run_daily_review()` | Orchestrate all steps |

**Holiday handling**: If the fetched daily data has no bar for today (market closed), log and skip. Optionally send a brief "今日港股休市" message.

**Per-ticker data needed**:
- `today_close`: `last["close"]`
- `prev_close`: `feat.iloc[-2]["close"]` (second-to-last bar)
- `pct_change`: `(today_close - prev_close) / prev_close * 100`
- `risk_score`: from `calculate_risk_score()`
- `rsi`: `last["rsi"]`
- `ema_status`: from `_ema_status(last)`
- `news`: from `_fetch_news(sym)`

### `notify/telegram.py` — `send_daily_report(text: str) -> bool`

Sends a plain-text Telegram message using the same `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` env vars. No formatting changes needed—just reuse the existing HTTP POST pattern.

### `ai/analyst.py` — `generate_market_narrative(ctx: dict) -> str`

**Input `ctx` keys**:
```python
{
  "vix": 18.5,
  "tickers": [
    {"sym": "0700.HK", "name": "腾讯控股", "pct_change": 1.23, "risk_score": 45, "rsi": 52.3},
    ...
  ],
  "alerts_count": 2,
  "news_headlines": ["腾讯Q4净利润超预期...", "微信月活达13亿新高", ...],
}
```

**Priority chain** (same as `analyze_risk_context`):
1. `claude -p "<prompt>"` via subprocess
2. MiniMax API (`MINIMAX_API_KEY`)
3. Anthropic API (`ANTHROPIC_API_KEY`)
4. Rule-based fallback

**Prompt template**:
```
你是专业港股分析师。根据以下数据生成一段 ≤200字 的中文市场点评（宏观+个股动态）：
VIX: {vix} ({vix_label})
个股: {ticker_summary}
今日头条: {headlines}
要求：简洁、客观、点到为止，不要重复数据。
```

**Rule-based fallback**: Compare gainer/loser count; combine VIX label; mention highest-risk ticker if risk_score > 60.

---

## Report Format

```
📊 港股日报 · 2026-02-26 周四

🌡 恐慌指数 VIX: 18.5 — 市场偏乐观

━━━ 个股概况 ━━━

🏢 腾讯控股 0700.HK
收盘: 375.00 HKD  今日 +1.23% ▲
风险分: 45/100 🟡  RSI: 52.3
均线: EMA50 > EMA200（多头排列）
📰 今日动态:
  · 腾讯Q4净利润超预期，环比增长15%
  · 微信月活达13亿新高

🏢 富途控股 INFQ
收盘: 68.50 USD  今日 -0.87% ▼
风险分: 38/100 🟢  RSI: 44.1
均线: EMA50 ≈ EMA200（震荡）
📰 今日动态:
  · 富途发布2025年度财报预告

━━━ 今日告警 ━━━
⚠️ 无告警触发

━━━ 宏观解读 ━━━
🤖 今日港股在科技股带动下小幅上涨，VIX
处于低位显示市场情绪偏乐观。腾讯受财
报利好支撑，短期动能良好，注意高位追涨
风险。

─────────────────
Watchtower · HK收盘复盘 16:30 HKT
```

---

## Scheduling (cron)

Add to crontab with `crontab -e`:

```cron
# Watchtower daily review — HK close (16:30 HKT = 08:30 UTC), Mon-Fri
30 8 * * 1-5 cd /Users/mac/Desktop/watchtower && /Users/mac/.local/bin/uv run python -m jobs.daily_review >> /tmp/watchtower_daily_review.log 2>&1
```

---

## Testing

New test file: `tests/test_daily_review.py`

| Test | What it covers |
|---|---|
| `test_vix_label` | All VIX threshold labels |
| `test_ema_status` | Bull/bear/neutral EMA states |
| `test_risk_emoji` | Score→emoji mapping |
| `test_get_today_alerts_empty` | Empty/missing alerts.jsonl |
| `test_get_today_alerts_filters_today` | Only today's alerts returned |
| `test_format_report_no_alerts` | Report string contains expected sections |
| `test_send_daily_report_no_creds` | Returns False without env vars |

---

## Module Boundary Compliance

- Fetch/ingest logic stays in `jobs/` (`fetch_ticker` reused)
- Signal/scoring stays in `alerts/` (`compute_features`, `calculate_risk_score` reused)
- Notification delivery stays in `notify/` (new `send_daily_report`)
- AI narrative stays in `ai/` (new `generate_market_narrative`)
- No cross-layer mixing
