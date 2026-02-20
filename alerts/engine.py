from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from .rules import AlertConfig


STATE_PATH = "data/alert_state.json"


def _ensure_data_dir() -> None:
    os.makedirs("data", exist_ok=True)


def _load_state() -> dict:
    _ensure_data_dir()
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _save_state(st: dict) -> None:
    _ensure_data_dir()
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)


def _should_emit(symbol: str, rule_id: str, ts_iso: str, cooldown_hours: int) -> bool:
    st = _load_state()
    key = f"{symbol}:{rule_id}"
    now = pd.to_datetime(ts_iso, utc=True)

    if key in st:
        last = pd.to_datetime(st[key], utc=True)
        if (now - last).total_seconds() < cooldown_hours * 3600:
            return False

    st[key] = ts_iso
    _save_state(st)
    return True


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df columns required: ts, open, high, low, close, volume
    """
    out = df.copy()
    out["ret"] = out["close"].pct_change()
    # realized vol proxy: rolling std of returns
    out["rv"] = out["ret"].rolling(20).std()
    # volume zscore
    mu = out["volume"].rolling(50).mean()
    sd = out["volume"].rolling(50).std(ddof=0).replace(0, np.nan)
    out["vol_z"] = (out["volume"] - mu) / sd
    return out


def evaluate_alerts(symbol: str, df: pd.DataFrame, cfg: AlertConfig) -> List[Dict[str, Any]]:
    """
    Returns a list of alert dicts:
      {symbol, severity, rule_id, ts, msg, context}
    """
    if df is None or df.empty or len(df) < cfg.min_bars:
        return []

    df = df.sort_values("ts").reset_index(drop=True)
    feat = compute_features(df)
    last = feat.iloc[-1]

    ts_iso = pd.to_datetime(last["ts"], utc=True).to_pydatetime().isoformat()
    ret = float(last.get("ret", np.nan)) if pd.notna(last.get("ret")) else 0.0
    vol_z = float(last.get("vol_z", np.nan)) if pd.notna(last.get("vol_z")) else np.nan

    # vol percentile based on rv history (needs enough)
    rv = feat["rv"].dropna()
    vol_pct = float(rv.rank(pct=True).iloc[-1]) if len(rv) >= 30 else 0.0

    # rv "warming up" proxy: compare last rv vs 10-bar mean
    rv_last = float(rv.iloc[-1]) if len(rv) else np.nan
    rv_mean10 = float(rv.tail(10).mean()) if len(rv) >= 10 else np.nan
    rv_warming = (np.isfinite(rv_last) and np.isfinite(rv_mean10) and rv_last > rv_mean10)

    alerts: List[Dict[str, Any]] = []

    def emit(rule_id: str, severity: str, msg: str) -> None:
        if _should_emit(symbol, rule_id, ts_iso, cfg.cooldown_hours):
            alerts.append({
                "symbol": symbol,
                "severity": severity,
                "rule_id": rule_id,
                "ts": ts_iso,
                "msg": msg,
                "context": {
                    "ret": ret,
                    "vol_z": None if not np.isfinite(vol_z) else vol_z,
                    "vol_pct": vol_pct,
                    "rv_last": None if not np.isfinite(rv_last) else rv_last,
                }
            })

    # 1) Event Shock (少但关键)：高波分位 + 放量 + 波动升温
    if (vol_pct >= cfg.vol_percentile_hi
        and np.isfinite(vol_z) and vol_z >= cfg.volume_z_hi
        and rv_warming):
        emit(
            "EVENT_SHOCK",
            "high",
            f"Event shock: vol_pct={vol_pct:.2f}, vol_z={vol_z:.2f}, ret={ret:+.2%}"
        )

    # 2) Structural Break：突破/破位 + 放量确认
    lb = cfg.breakout_lookback
    if len(feat) >= lb:
        window = feat.tail(lb)
        hi = float(window["high"].max())
        lo = float(window["low"].min())
        close = float(last["close"])

        if close >= hi and np.isfinite(vol_z) and vol_z >= cfg.breakout_volume_z:
            emit(
                "BREAKOUT_UP",
                "med",
                f"Breakout up: close>=hi({lb}) vol_z={vol_z:.2f}, ret={ret:+.2%}"
            )
        if close <= lo and np.isfinite(vol_z) and vol_z >= cfg.breakout_volume_z:
            emit(
                "BREAKDOWN_DOWN",
                "med",
                f"Breakdown down: close<=lo({lb}) vol_z={vol_z:.2f}, ret={ret:+.2%}"
            )

    # 3) Tail Risk：大涨跌 + 波动升温（过滤噪声）
    if (vol_pct >= cfg.vol_percentile_hi) and (ret >= cfg.tail_ret_hi or ret <= cfg.tail_ret_lo):
        emit(
            "TAIL_RISK",
            "high",
            f"Tail risk: ret={ret:+.2%}, vol_pct={vol_pct:.2f}, vol_z={vol_z:.2f}"
        )

    return alerts


def append_alerts_jsonl(alerts: List[Dict[str, Any]], path: str = "data/alerts.jsonl") -> None:
    if not alerts:
        return
    _ensure_data_dir()
    with open(path, "a", encoding="utf-8") as f:
        for a in alerts:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")