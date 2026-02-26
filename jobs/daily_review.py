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
from notify.telegram import send_daily_report

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
