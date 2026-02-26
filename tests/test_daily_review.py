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
