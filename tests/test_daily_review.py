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


def test_send_daily_report_returns_false_with_only_token(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    from notify.telegram import send_daily_report
    assert send_daily_report("hello") is False


def test_send_daily_report_returns_false_on_http_error(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat456")

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_resp.json.return_value = {"ok": False}

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        from notify.telegram import send_daily_report
        result = send_daily_report("report text")

    assert result is False


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
