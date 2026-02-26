# Daily Review Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `jobs/daily_review.py` that generates and sends a Telegram market recap after Hong Kong market close, covering per-ticker metrics, VIX sentiment, today's alerts, latest news, and an AI-generated macro narrative.

**Architecture:** New standalone job (`jobs/daily_review.py`) reusing existing `fetch_ticker`, `compute_features`, `calculate_risk_score` from the ingest/alerts layers. Extends `notify/telegram.py` with `send_daily_report()` and `ai/analyst.py` with `generate_market_narrative()`. Triggered by cron at 08:30 UTC (= 16:30 HKT) Mon–Fri.

**Tech Stack:** Python 3.11, yfinance, httpx, pandas, SQLite (read-only for alerts), Telegram Bot API

---

## Task 1: `send_daily_report()` in `notify/telegram.py`

**Files:**
- Modify: `notify/telegram.py`
- Test: `tests/test_daily_review.py` (create)

### Step 1: Create test file with failing test

Create `tests/test_daily_review.py`:

```python
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── Task 1: send_daily_report ────────────────────────────────────────────────

def test_send_daily_report_returns_false_without_creds(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    from notify.telegram import send_daily_report
    assert send_daily_report("hello") is False


def test_send_daily_report_posts_text_to_telegram(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat456")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True}

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        from notify.telegram import send_daily_report
        result = send_daily_report("📊 港股日报测试")

    assert result is True
    call_kwargs = mock_client.post.call_args
    payload = call_kwargs[1]["json"]
    assert payload["chat_id"] == "chat456"
    assert "📊 港股日报测试" in payload["text"]
```

### Step 2: Run test to verify it fails

```bash
uv run python -m pytest tests/test_daily_review.py::test_send_daily_report_returns_false_without_creds tests/test_daily_review.py::test_send_daily_report_posts_text_to_telegram -v
```

Expected: `ImportError: cannot import name 'send_daily_report'`

### Step 3: Implement `send_daily_report` in `notify/telegram.py`

Append to end of `notify/telegram.py`:

```python

def send_daily_report(text: str) -> bool:
    """
    发送每日复盘报告到 Telegram（纯文本消息）。
    读取环境变量 TELEGRAM_BOT_TOKEN 和 TELEGRAM_CHAT_ID。
    任意一个缺失则静默跳过，返回 False。
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        return False

    url = _API_BASE.format(token=token)
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(url, json={"chat_id": chat_id, "text": text})
            if resp.status_code == 200 and resp.json().get("ok"):
                return True
            print(f"[telegram] 日报发送失败 HTTP {resp.status_code}: {resp.text[:300]}")
            return False
    except Exception as e:
        print(f"[telegram] 日报发送异常: {e}")
        return False
```

### Step 4: Run tests to verify they pass

```bash
uv run python -m pytest tests/test_daily_review.py::test_send_daily_report_returns_false_without_creds tests/test_daily_review.py::test_send_daily_report_posts_text_to_telegram -v
```

Expected: 2 PASSED

### Step 5: Commit

```bash
git add notify/telegram.py tests/test_daily_review.py
git commit -m "feat: add send_daily_report() to notify/telegram"
```

---

## Task 2: `generate_market_narrative()` in `ai/analyst.py`

**Files:**
- Modify: `ai/analyst.py`
- Test: `tests/test_daily_review.py`

### Step 1: Add failing tests

Append to `tests/test_daily_review.py`:

```python
# ── Task 2: generate_market_narrative ────────────────────────────────────────

def test_generate_market_narrative_fallback_no_ai(monkeypatch):
    """With no AI available, rule-based fallback returns a non-empty Chinese string."""
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from ai.analyst import AINarrativeAnalyst
    analyst = AINarrativeAnalyst()
    analyst._claude_cli = None  # disable CLI

    ctx = {
        "vix": 18.5,
        "vix_label": "市场偏乐观",
        "tickers": [
            {"sym": "0700.HK", "pct_change": 1.2, "risk_score": 45, "rsi": 52.0},
            {"sym": "INFQ", "pct_change": -0.5, "risk_score": 35, "rsi": 44.0},
        ],
        "alerts_count": 0,
        "news_headlines": ["腾讯Q4净利润超预期", "港股科技板块走强"],
    }
    result = analyst.generate_market_narrative(ctx)
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_market_narrative_high_risk_ticker_mentioned(monkeypatch):
    """Fallback narrative mentions the ticker with risk_score > 60."""
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from ai.analyst import AINarrativeAnalyst
    analyst = AINarrativeAnalyst()
    analyst._claude_cli = None

    ctx = {
        "vix": 28.0,
        "vix_label": "偏恐慌",
        "tickers": [
            {"sym": "0700.HK", "pct_change": -3.5, "risk_score": 75, "rsi": 28.0},
        ],
        "alerts_count": 1,
        "news_headlines": [],
    }
    result = analyst.generate_market_narrative(ctx)
    assert "0700.HK" in result
```

### Step 2: Run tests to verify they fail

```bash
uv run python -m pytest tests/test_daily_review.py::test_generate_market_narrative_fallback_no_ai tests/test_daily_review.py::test_generate_market_narrative_high_risk_ticker_mentioned -v
```

Expected: `AttributeError: 'AINarrativeAnalyst' object has no attribute 'generate_market_narrative'`

### Step 3: Add `generate_market_narrative` to `ai/analyst.py`

Add the prompt constant near the top of `ai/analyst.py` (after `PROMPT_TEMPLATE`):

```python
DAILY_NARRATIVE_PROMPT = """你是专业港股分析师。根据以下数据生成一段 ≤200字 的中文市场点评（宏观+个股动态）：

VIX恐慌指数: {vix:.1f}（{vix_label}）
个股表现: {ticker_summary}
今日头条: {headlines}

要求：简洁、客观、点到为止，不要重复数字数据，突出最值得关注的风险或机会。"""
```

Add this method to `AINarrativeAnalyst` class (before the closing of the class, after `_fallback_rule_engine`):

```python
    def generate_market_narrative(self, ctx: dict) -> str:
        """
        生成每日市场宏观叙事。
        ctx keys: vix, vix_label, tickers (list of dicts), alerts_count, news_headlines
        Priority: claude CLI → MiniMax → Anthropic → rule-based fallback.
        """
        vix = ctx.get("vix", 20.0)
        vix_label = ctx.get("vix_label", "中性")
        tickers = ctx.get("tickers", [])
        headlines = ctx.get("news_headlines", [])

        ticker_summary = "; ".join(
            f"{t['sym']} {t['pct_change']:+.1f}% 风险分{t['risk_score']:.0f}"
            for t in tickers
        )
        headlines_str = "、".join(headlines[:5]) if headlines else "暂无"

        prompt = DAILY_NARRATIVE_PROMPT.format(
            vix=vix,
            vix_label=vix_label,
            ticker_summary=ticker_summary,
            headlines=headlines_str,
        )

        if self._claude_cli:
            result = self._try_claude_cli(prompt)
            if result:
                return result

        if self.minimax_key:
            return self._call_minimax(prompt, "MARKET", vix, 0, 0)

        if self.anthropic_key:
            return self._call_anthropic(prompt, "MARKET", vix, 0, 0)

        return self._fallback_daily_narrative(ctx)

    def _fallback_daily_narrative(self, ctx: dict) -> str:
        """规则备选：无 AI 时生成简单中文市场点评。"""
        vix = ctx.get("vix", 20.0)
        vix_label = ctx.get("vix_label", "中性")
        tickers = ctx.get("tickers", [])
        alerts_count = ctx.get("alerts_count", 0)

        gainers = [t for t in tickers if t.get("pct_change", 0) > 0]
        losers = [t for t in tickers if t.get("pct_change", 0) < 0]

        if len(gainers) > len(losers):
            trend = "整体偏多，多数标的收涨"
        elif len(losers) > len(gainers):
            trend = "整体承压，多数标的收跌"
        else:
            trend = "涨跌互现，市场分歧明显"

        high_risk = [t for t in tickers if t.get("risk_score", 0) > 60]
        risk_note = ""
        if high_risk:
            syms = "、".join(t["sym"] for t in high_risk)
            risk_note = f"注意 {syms} 风险分偏高，请控制仓位。"

        alert_note = f"今日触发 {alerts_count} 条告警。" if alerts_count > 0 else ""

        return f"今日港股{trend}，VIX {vix:.1f}（{vix_label}）。{risk_note}{alert_note}".strip()
```

### Step 4: Run tests to verify they pass

```bash
uv run python -m pytest tests/test_daily_review.py::test_generate_market_narrative_fallback_no_ai tests/test_daily_review.py::test_generate_market_narrative_high_risk_ticker_mentioned -v
```

Expected: 2 PASSED

### Step 5: Commit

```bash
git add ai/analyst.py tests/test_daily_review.py
git commit -m "feat: add generate_market_narrative() to AINarrativeAnalyst"
```

---

## Task 3: Helper functions in `jobs/daily_review.py`

**Files:**
- Create: `jobs/daily_review.py`
- Test: `tests/test_daily_review.py`

### Step 1: Add failing tests for helper functions

Append to `tests/test_daily_review.py`:

```python
# ── Task 3: jobs/daily_review helpers ────────────────────────────────────────

def test_vix_label_thresholds():
    from jobs.daily_review import _vix_label
    assert _vix_label(12.0) == "极度乐观"
    assert _vix_label(17.0) == "偏乐观"
    assert _vix_label(22.0) == "中性"
    assert _vix_label(27.0) == "偏恐慌"
    assert _vix_label(35.0) == "极度恐慌"


def test_risk_emoji_thresholds():
    from jobs.daily_review import _risk_emoji
    assert _risk_emoji(20.0) == "🟢"
    assert _risk_emoji(34.9) == "🟢"
    assert _risk_emoji(35.0) == "🟡"
    assert _risk_emoji(59.9) == "🟡"
    assert _risk_emoji(60.0) == "🔴"
    assert _risk_emoji(90.0) == "🔴"


def test_ema_status_bull():
    from jobs.daily_review import _ema_status
    last = pd.Series({"close": 120.0, "ema_fast": 110.0, "ema_slow": 100.0})
    assert _ema_status(last) == "多头排列"


def test_ema_status_bear():
    from jobs.daily_review import _ema_status
    last = pd.Series({"close": 80.0, "ema_fast": 90.0, "ema_slow": 100.0})
    assert _ema_status(last) == "空头排列"


def test_ema_status_neutral():
    from jobs.daily_review import _ema_status
    # price above slow but fast below slow
    last = pd.Series({"close": 105.0, "ema_fast": 95.0, "ema_slow": 100.0})
    assert _ema_status(last) == "震荡"


def test_get_today_alerts_empty_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    from jobs.daily_review import _get_today_alerts
    result = _get_today_alerts("2026-02-26")
    assert result == []


def test_get_today_alerts_filters_by_date(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    alerts_file = data_dir / "alerts.jsonl"
    # Write two alerts: one today, one yesterday
    alerts_file.write_text(
        json.dumps({"ts": "2026-02-26T09:00:00+00:00", "rule_id": "RISK_HIGH", "symbol": "0700.HK"}) + "\n" +
        json.dumps({"ts": "2026-02-25T09:00:00+00:00", "rule_id": "RISK_LOW", "symbol": "INFQ"}) + "\n",
        encoding="utf-8",
    )
    from jobs.daily_review import _get_today_alerts
    result = _get_today_alerts("2026-02-26")
    assert len(result) == 1
    assert result[0]["rule_id"] == "RISK_HIGH"


def test_format_report_contains_required_sections():
    from jobs.daily_review import _format_report
    ticker_data = [
        {
            "sym": "0700.HK",
            "name": "0700.HK",
            "today_close": 375.0,
            "pct_change": 1.23,
            "risk_score": 45.0,
            "rsi": 52.3,
            "ema_status": "多头排列",
            "news": ["腾讯Q4净利润超预期"],
        }
    ]
    report = _format_report(
        date_str="2026-02-26",
        weekday_str="周四",
        vix=18.5,
        vix_label="偏乐观",
        ticker_data=ticker_data,
        today_alerts=[],
        narrative="今日市场偏乐观。",
    )
    assert "港股日报" in report
    assert "VIX" in report
    assert "0700.HK" in report
    assert "375.00" in report
    assert "+1.23%" in report
    assert "45/100" in report
    assert "今日告警" in report
    assert "宏观解读" in report
    assert "今日市场偏乐观" in report
```

### Step 2: Run tests to verify they fail

```bash
uv run python -m pytest tests/test_daily_review.py -k "vix_label or risk_emoji or ema_status or today_alerts or format_report" -v
```

Expected: `ImportError: No module named 'jobs.daily_review'`

### Step 3: Create `jobs/daily_review.py` with helpers

Create `jobs/daily_review.py`:

```python
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf

from alerts.engine import compute_features, calculate_risk_score
from alerts.rules import AlertConfig
from jobs.ingest import fetch_ticker, load_config

ALERTS_PATH = "data/alerts.jsonl"
WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


# ── Pure helper functions (easily tested) ─────────────────────────────────────

def _vix_label(vix: float) -> str:
    if vix < 15:
        return "极度乐观"
    if vix < 20:
        return "偏乐观"
    if vix < 25:
        return "中性"
    if vix < 30:
        return "偏恐慌"
    return "极度恐慌"


def _risk_emoji(score: float) -> str:
    if score < 35:
        return "🟢"
    if score < 60:
        return "🟡"
    return "🔴"


def _ema_status(last: pd.Series) -> str:
    close = last["close"]
    fast = last["ema_fast"]
    slow = last["ema_slow"]
    if close > slow and fast > slow:
        return "多头排列"
    if close < slow and fast < slow:
        return "空头排列"
    return "震荡"


def _get_today_alerts(date_str: str) -> list[dict]:
    """Return alerts from data/alerts.jsonl whose ts falls on date_str (YYYY-MM-DD, UTC)."""
    if not os.path.exists(ALERTS_PATH):
        return []
    results = []
    try:
        with open(ALERTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    alert = json.loads(line)
                    ts = alert.get("ts", "")
                    if ts.startswith(date_str):
                        results.append(alert)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[daily_review] Error reading alerts: {e}")
    return results


def _fetch_news(sym: str) -> list[str]:
    """Return up to 3 most recent news titles for a ticker via yfinance."""
    try:
        ticker = yf.Ticker(sym)
        raw_news = ticker.news or []
        # Sort by providerPublishTime descending (most recent first)
        raw_news = sorted(raw_news, key=lambda x: x.get("providerPublishTime", 0), reverse=True)
        titles = []
        for item in raw_news[:3]:
            # yfinance news dict may vary across versions; handle both structures
            title = (
                item.get("title")
                or item.get("content", {}).get("title")
                or ""
            )
            if title:
                titles.append(title)
        return titles
    except Exception as e:
        print(f"[daily_review] News fetch failed for {sym}: {e}")
        return []


def _format_report(
    date_str: str,
    weekday_str: str,
    vix: float,
    vix_label: str,
    ticker_data: list[dict[str, Any]],
    today_alerts: list[dict],
    narrative: str,
) -> str:
    lines = [
        f"📊 港股日报 · {date_str} {weekday_str}",
        "",
        f"🌡 恐慌指数 VIX: {vix:.1f} — {vix_label}",
        "",
        "━━━ 个股概况 ━━━",
    ]

    for td in ticker_data:
        sym = td["sym"]
        name = td["name"]
        close = td["today_close"]
        pct = td["pct_change"]
        score = td["risk_score"]
        rsi = td["rsi"]
        ema = td["ema_status"]
        emoji = _risk_emoji(score)
        arrow = "▲" if pct >= 0 else "▼"
        sign = "+" if pct >= 0 else ""

        header = sym if name == sym else f"{name} {sym}"
        lines += [
            "",
            f"🏢 {header}",
            f"收盘: {close:.2f}  今日 {sign}{pct:.2f}% {arrow}",
            f"风险分: {score:.0f}/100 {emoji}  RSI: {rsi:.1f}",
            f"均线: {ema}",
        ]
        news = td.get("news", [])
        if news:
            lines.append("📰 今日动态:")
            for headline in news:
                lines.append(f"  · {headline}")

    lines += ["", "━━━ 今日告警 ━━━"]
    if today_alerts:
        for a in today_alerts:
            sev_emoji = {"high": "🔴", "med": "🟡", "buy": "🟢"}.get(a.get("severity", ""), "⚪")
            ts_short = a.get("ts", "")[:16].replace("T", " ")
            lines.append(f"{sev_emoji} {a.get('symbol','')} {a.get('rule_id','')} @ {ts_short}")
    else:
        lines.append("✅ 无告警触发")

    lines += [
        "",
        "━━━ 宏观解读 ━━━",
        f"🤖 {narrative}",
        "",
        "─────────────────",
        "Watchtower · HK收盘复盘 16:30 HKT",
    ]
    return "\n".join(lines)
```

### Step 4: Run tests to verify helpers pass

```bash
uv run python -m pytest tests/test_daily_review.py -k "vix_label or risk_emoji or ema_status or today_alerts or format_report" -v
```

Expected: 8 PASSED

### Step 5: Commit

```bash
git add jobs/daily_review.py tests/test_daily_review.py
git commit -m "feat: add daily_review helper functions with tests"
```

---

## Task 4: Main orchestrator `run_daily_review()`

**Files:**
- Modify: `jobs/daily_review.py` (append `run_daily_review` + `__main__`)

### Step 1: Add failing smoke test

Append to `tests/test_daily_review.py`:

```python
# ── Task 4: run_daily_review orchestrator ────────────────────────────────────

def test_run_daily_review_sends_report(tmp_path, monkeypatch):
    """End-to-end smoke: run_daily_review calls send_daily_report with a non-empty string."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    # Minimal config
    config_file = tmp_path / "config.yaml"
    config_file.write_text("tickers:\n  - AAPL\ninterval: 1d\ndays: 30\ndb_path: watchtower.db\n")

    # Stub fetch_ticker to return a minimal DataFrame
    import numpy as np
    dates = pd.date_range("2026-01-01", periods=250, freq="B", tz="UTC")
    prices = np.linspace(100, 150, 250)
    fake_df = pd.DataFrame({
        "Open": prices,
        "High": prices * 1.01,
        "Low": prices * 0.99,
        "Close": prices,
        "Volume": [1_000_000.0] * 250,
    }, index=dates)

    monkeypatch.setattr("jobs.daily_review.fetch_ticker", lambda *a, **kw: fake_df)
    monkeypatch.setattr("jobs.daily_review._fetch_news", lambda sym: ["News headline 1"])
    monkeypatch.setattr("jobs.daily_review.load_config", lambda: __import__("jobs.ingest", fromlist=["WatchtowerConfig"]).WatchtowerConfig(
        tickers=["AAPL"], interval="1d", days=30, db_path="watchtower.db"
    ))

    sent_reports = []

    def fake_send(text: str) -> bool:
        sent_reports.append(text)
        return True

    monkeypatch.setattr("jobs.daily_review.send_daily_report", fake_send)
    monkeypatch.setattr("jobs.daily_review._fetch_vix", lambda: 18.5)

    from jobs.daily_review import run_daily_review
    run_daily_review()

    assert len(sent_reports) == 1
    report = sent_reports[0]
    assert "港股日报" in report
    assert "VIX" in report
    assert "AAPL" in report
```

### Step 2: Run test to verify it fails

```bash
uv run python -m pytest tests/test_daily_review.py::test_run_daily_review_sends_report -v
```

Expected: `ImportError` or `AttributeError` — `run_daily_review` and `_fetch_vix` not yet defined

### Step 3: Append orchestrator to `jobs/daily_review.py`

Add at the end of `jobs/daily_review.py`:

```python

# ── VIX fetcher ───────────────────────────────────────────────────────────────

def _fetch_vix() -> float:
    """Return the latest VIX close. Defaults to 20.0 on failure."""
    try:
        df = fetch_ticker("^VIX", "1d", 5)
        if not df.empty:
            return float(df.iloc[-1]["Close"])
    except Exception as e:
        print(f"[daily_review] VIX fetch failed: {e}")
    return 20.0


def _get_ticker_name(sym: str) -> str:
    """Try to get a human-readable display name; fall back to symbol on any error."""
    try:
        info = yf.Ticker(sym).info
        return info.get("longName") or info.get("shortName") or sym
    except Exception:
        return sym


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_daily_review() -> None:
    """
    Generate and send the daily HK market recap via Telegram.
    Called by cron at 08:30 UTC (= 16:30 HKT) Mon–Fri.
    """
    from notify.telegram import send_daily_report
    from ai.analyst import AINarrativeAnalyst

    cfg = load_config()
    alert_cfg = AlertConfig.load_evolution()
    analyst = AINarrativeAnalyst()

    now_utc = datetime.now(timezone.utc)
    date_str = now_utc.strftime("%Y-%m-%d")
    weekday_str = WEEKDAY_CN[now_utc.weekday()]

    print(f"[daily_review] 开始生成 {date_str} 港股复盘报告...")

    # 1. VIX
    vix = _fetch_vix()
    vix_label = _vix_label(vix)
    print(f"[daily_review] VIX: {vix:.1f} ({vix_label})")

    # 2. Per-ticker metrics
    tickers = [t for t in cfg.tickers if t != "^VIX"]
    ticker_data: list[dict[str, Any]] = []
    all_headlines: list[str] = []

    for sym in tickers:
        print(f"[daily_review] Processing {sym}...")
        try:
            raw_df = fetch_ticker(sym, "1d", 365)
            if raw_df is None or raw_df.empty or len(raw_df) < 2:
                print(f"[daily_review] Skipping {sym}: insufficient data")
                continue

            # Normalize to alerts engine format
            tmp = raw_df.reset_index()
            ts_col = next(
                (c for c in ("Datetime", "Date") if c in tmp.columns),
                tmp.columns[0],
            )
            df_alert = pd.DataFrame({
                "ts": pd.to_datetime(tmp[ts_col], utc=True),
                "open": tmp["Open"].astype(float),
                "high": tmp["High"].astype(float),
                "low": tmp["Low"].astype(float),
                "close": tmp["Close"].astype(float),
                "volume": tmp.get("Volume", pd.Series([0.0] * len(tmp))).astype(float),
            }).sort_values("ts").reset_index(drop=True)

            feat = compute_features(df_alert, alert_cfg)
            if len(feat) < 2:
                continue

            last = feat.iloc[-1]
            prev = feat.iloc[-2]
            today_close = float(last["close"])
            prev_close = float(prev["close"])
            pct_change = (today_close - prev_close) / prev_close * 100 if prev_close else 0.0
            risk_score = calculate_risk_score(last, alert_cfg, global_vix=vix)
            rsi = float(last["rsi"]) if not pd.isna(last["rsi"]) else 50.0
            ema = _ema_status(last)
            news = _fetch_news(sym)
            all_headlines.extend(news)
            name = _get_ticker_name(sym)

            ticker_data.append({
                "sym": sym,
                "name": name,
                "today_close": today_close,
                "pct_change": pct_change,
                "risk_score": risk_score,
                "rsi": rsi,
                "ema_status": ema,
                "news": news,
            })
        except Exception as e:
            print(f"[daily_review] Error processing {sym}: {e}")
            continue

    if not ticker_data:
        print("[daily_review] No ticker data available — possibly a market holiday. Skipping report.")
        return

    # 3. Today's alerts
    today_alerts = _get_today_alerts(date_str)
    print(f"[daily_review] Today's alerts: {len(today_alerts)}")

    # 4. AI narrative
    narrative_ctx = {
        "vix": vix,
        "vix_label": vix_label,
        "tickers": [
            {"sym": t["sym"], "pct_change": t["pct_change"],
             "risk_score": t["risk_score"], "rsi": t["rsi"]}
            for t in ticker_data
        ],
        "alerts_count": len(today_alerts),
        "news_headlines": all_headlines[:6],
    }
    narrative = analyst.generate_market_narrative(narrative_ctx)
    print(f"[daily_review] Narrative generated ({len(narrative)} chars)")

    # 5. Format and send
    report = _format_report(
        date_str=date_str,
        weekday_str=weekday_str,
        vix=vix,
        vix_label=vix_label,
        ticker_data=ticker_data,
        today_alerts=today_alerts,
        narrative=narrative,
    )
    success = send_daily_report(report)
    if success:
        print(f"[daily_review] 报告已发送 ({len(report)} chars)")
    else:
        print("[daily_review] 报告发送失败（未配置 Telegram 凭证？）")


if __name__ == "__main__":
    run_daily_review()
```

### Step 4: Run all tests

```bash
uv run python -m pytest tests/test_daily_review.py -v
```

Expected: all tests PASSED

### Step 5: Run full test suite to check for regressions

```bash
uv run python -m pytest -v
```

Expected: all existing tests still PASS

### Step 6: Commit

```bash
git add jobs/daily_review.py tests/test_daily_review.py
git commit -m "feat: add run_daily_review() orchestrator"
```

---

## Task 5: Cron Setup

**Files:** None — this is a manual user action.

### Step 1: Verify the script runs manually

```bash
uv run python -m jobs.daily_review
```

Expected: logs output, report printed. If Telegram creds are set, message is delivered.

### Step 2: Add crontab entry

Open crontab editor:
```bash
crontab -e
```

Add this line (replace path if your project is elsewhere):
```cron
# Watchtower daily review — HK close (16:30 HKT = 08:30 UTC), Mon-Fri
30 8 * * 1-5 cd /Users/mac/Desktop/watchtower && /Users/mac/.local/bin/uv run python -m jobs.daily_review >> /tmp/watchtower_daily_review.log 2>&1
```

### Step 3: Verify crontab saved

```bash
crontab -l | grep daily_review
```

Expected: line shown above

### Step 4: Final commit with CLAUDE.md update

Add the new command to CLAUDE.md `Commands` section:
```bash
uv run python -m jobs.daily_review   # 手动触发每日复盘报告
```

```bash
git add CLAUDE.md
git commit -m "docs: document daily_review command in CLAUDE.md"
```

---

## Summary of Changes

| File | Change |
|---|---|
| `notify/telegram.py` | +`send_daily_report(text)` |
| `ai/analyst.py` | +`generate_market_narrative(ctx)`, +`_fallback_daily_narrative(ctx)`, +`DAILY_NARRATIVE_PROMPT` |
| `jobs/daily_review.py` | New file — helpers + orchestrator |
| `tests/test_daily_review.py` | New file — 11 tests |
| `CLAUDE.md` | +command doc |

## Cron Reference

```
30 8 * * 1-5   → 每周一至周五 08:30 UTC = 16:30 HKT（港股收盘后30分钟）
```

Log file: `/tmp/watchtower_daily_review.log`
