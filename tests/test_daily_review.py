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
