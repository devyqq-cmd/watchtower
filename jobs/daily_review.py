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
