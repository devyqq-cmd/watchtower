from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import List, Dict, Any

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .rules import AlertConfig

# Ensure root is in path for ai.analyst
root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
from ai.analyst import AINarrativeAnalyst

STATE_PATH = "data/alert_state.json"
ai_analyst = AINarrativeAnalyst()


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


def compute_features(df: pd.DataFrame, cfg: AlertConfig) -> pd.DataFrame:
    """
    Enhanced feature engineering for long-term risk assessment.
    """
    out = df.copy()
    out["ret"] = out["close"].pct_change()
    
    # 1. Trend: EMAs
    out["ema_fast"] = out["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    
    # 2. Momentum: RSI (Manual Implementation to avoid pandas-ta dependency)
    delta = out["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=cfg.rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=cfg.rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))
    
    # 3. Overextension: Distance from EMA_slow
    out["dist_ema"] = (out["close"] - out["ema_slow"]) / out["ema_slow"]
    mu = out["dist_ema"].rolling(cfg.z_score_lookback).mean()
    sd = out["dist_ema"].rolling(cfg.z_score_lookback).std()
    out["z_dist"] = (out["dist_ema"] - mu) / sd
    
    # 4. Volatility: ATR
    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift()).abs()
    low_close = (out["low"] - out["close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    out["atr"] = true_range.rolling(14).mean()
    out["rv"] = out["ret"].rolling(20).std()
    
    # 5. Volume Z-Score
    mu_vol = out["volume"].rolling(50).mean()
    sd_vol = out["volume"].rolling(50).std(ddof=0).replace(0, np.nan)
    out["vol_z"] = (out["volume"] - mu_vol) / sd_vol
    
    return out


def calculate_risk_score(last: pd.Series, cfg: AlertConfig, global_vix: float = 20.0, valuation: dict = None) -> float:
    """
    Aggregates factors into a 0-100 Risk Score, influenced by VIX and Valuation.
    """
    valuation = valuation or {}
    pe = float(valuation.get("trailingPE", 25.0)) # Default mid-range
    div_yield = float(valuation.get("dividendYield", 0.0))
    
    # 1. Trend Factor (0 to 100)
    if last["close"] > last["ema_slow"] and last["ema_fast"] > last["ema_slow"]:
        s_trend = 0
    elif last["close"] < last["ema_slow"] and last["ema_fast"] < last["ema_slow"]:
        s_trend = 100
    else:
        s_trend = 50
        
    # 2. Overextension (Z-score based)
    z = last["z_dist"]
    if z > 2.0:
        s_ext = 100
    elif z > 1.0:
        s_ext = 70
    elif z < -2.0:
        s_ext = 80 
    else:
        s_ext = 20
        
    # 3. Momentum (RSI)
    rsi = last["rsi"]
    if rsi > 80: s_mom = 100
    elif rsi > 70: s_mom = 80
    elif rsi < 30: s_mom = 70 
    else: s_mom = 20
    
    # 4. Volatility (ATR Percentile)
    s_vol = 100 if last["vol_z"] > 2.0 else 20

    # 5. --- FUNDAMENTAL VALUATION FACTOR (Long-term Anchor) ---
    # High PE -> High Risk (Greed); Low PE or High Div -> Low Risk (Value)
    if pe > cfg.pe_high_threshold:
        s_val = 100 # Very expensive
    elif pe < cfg.pe_low_threshold:
        s_val = 10 # Strong value
    elif div_yield > cfg.div_yield_min:
        s_val = 20 # Safe yield
    else:
        s_val = 50 # Fairly valued
    
    # --- VIX INFLUENCE ---
    vix_modifier = 0
    if global_vix < cfg.vix_complacency_threshold and (s_ext > 50 or s_mom > 50):
        vix_modifier += 15
    if global_vix > cfg.vix_panic_threshold:
        vix_modifier -= 10

    # Weighted Sum
    score = (
        s_trend * cfg.weights["trend"] +
        s_ext * cfg.weights["overextension"] +
        s_mom * cfg.weights["momentum"] +
        s_vol * cfg.weights["volatility"] +
        s_val * cfg.weights["valuation"]
    ) + vix_modifier
    return float(np.clip(score, 0, 100))


def evaluate_alerts(symbol: str, df: pd.DataFrame, cfg: AlertConfig, global_vix: float = 20.0, valuation: dict = None) -> List[Dict[str, Any]]:
    if df is None or df.empty or len(df) < cfg.min_bars:
        return []

    df = df.sort_values("ts").reset_index(drop=True)
    feat = compute_features(df, cfg)
    last = feat.iloc[-1]

    ts_iso = pd.to_datetime(last["ts"], utc=True).to_pydatetime().isoformat()
    risk_score = calculate_risk_score(last, cfg, global_vix=global_vix, valuation=valuation)
    
    pe_str = f"P/E: {valuation.get('trailingPE', 'N/A')}" if valuation else ""
    print(f"[DEBUG] Symbol: {symbol}, Risk Score: {risk_score:.1f}/100, RSI: {last['rsi']:.1f}, Z-Dist: {last['z_dist']:.2f}, {pe_str}, Global VIX: {global_vix:.1f}")
    
    alerts: List[Dict[str, Any]] = []

    def emit(rule_id: str, severity: str, msg: str, context: dict = None) -> None:
        if _should_emit(symbol, rule_id, ts_iso, cfg.cooldown_hours):
            ctx = context or {}
            ctx.update({
                "risk_score": risk_score,
                "rsi": last["rsi"],
                "z_dist": last["z_dist"],
                "price": last["close"]
            })
            # Generate AI Narrative based on risk_data
            ai_advice = ai_analyst.analyze_risk_context(symbol, ctx)
            full_msg = f"{msg}\nAI ADVICE: {ai_advice}"
            
            alerts.append({
                "symbol": symbol,
                "severity": severity,
                "rule_id": rule_id,
                "ts": ts_iso,
                "msg": full_msg,
                "context": ctx
            })

    # 1. CORE: Risk Scoring Alerts (Discipline)
    if risk_score >= 80:
        emit("RISK_EXTREME", "high", f"EXTREME RISK ({risk_score:.0f}/100): Extreme Greed or Structural Breakdown. ACTION: STRICT PROFIT TAKING / DE-LEVERAGE.")
    elif risk_score >= 60:
        emit("RISK_HIGH", "med", f"High Risk ({risk_score:.0f}/100): Overextended Upwards or Extreme Panic.")
    elif risk_score <= 20:
        # emit("RISK_LOW", "info", f"Low Risk ({risk_score:.0f}/100): Bullish Trend & Healthy Consolidation.")
        pass

    # 2. Event Shock (Keep original logic but integrate)
    vol_z = last["vol_z"]
    rv_last = last["rv"]
    rv_mean10 = feat["rv"].tail(10).mean()
    if vol_z >= cfg.volume_z_hi and rv_last > rv_mean10:
        emit("EVENT_SHOCK", "high", f"Volume & Volatility Shock detected (vol_z={vol_z:.2f}). Market regime shift likely.")

    return alerts


def append_alerts_jsonl(alerts: List[Dict[str, Any]], path: str = "data/alerts.jsonl") -> None:
    if not alerts:
        return
    _ensure_data_dir()
    with open(path, "a", encoding="utf-8") as f:
        for a in alerts:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")


def append_alerts_jsonl(alerts: List[Dict[str, Any]], path: str = "data/alerts.jsonl") -> None:
    if not alerts:
        return
    _ensure_data_dir()
    with open(path, "a", encoding="utf-8") as f:
        for a in alerts:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")