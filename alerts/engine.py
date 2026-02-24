from __future__ import annotations

import json
import logging
import os
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
logger = logging.getLogger(__name__)


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


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Wilder 平滑 RSI，与 TradingView ta.rsi() 算法一致。
    种子：前 period 根K线的简单均值；之后每根K线做指数平滑。
    avg = (prev_avg * (period-1) + current) / period
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = pd.Series(np.nan, index=close.index)
    avg_loss = pd.Series(np.nan, index=close.index)

    # 种子位置：iloc[period]（iloc[0] 的 diff 是 NaN，所以用 iloc[1..period] 共 period 个值）
    seed = period
    if len(close) <= seed:
        return avg_gain  # 数据不足，返回全 NaN

    avg_gain.iloc[seed] = gain.iloc[1 : seed + 1].mean()
    avg_loss.iloc[seed] = loss.iloc[1 : seed + 1].mean()

    for i in range(seed + 1, len(close)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_features(df: pd.DataFrame, cfg: AlertConfig) -> pd.DataFrame:
    """
    Enhanced feature engineering for long-term risk assessment.
    """
    out = df.copy()
    out["ret"] = out["close"].pct_change()

    # 1. Trend: EMAs
    out["ema_fast"] = out["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=cfg.ema_slow, adjust=False).mean()

    # 2. Momentum: Wilder RSI（与 TradingView 对齐）
    out["rsi"] = _wilder_rsi(out["close"], cfg.rsi_period)
    
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
    # Risk score semantics: higher = bigger downside/trim risk.
    # Deeply negative z-score should reduce sell-risk, not increase it.
    z = last["z_dist"]
    if z > 2.0:
        s_ext = 100
    elif z > 1.0:
        s_ext = 70
    elif z < -2.0:
        s_ext = 10
    elif z < -1.0:
        s_ext = 15
    else:
        s_ext = 20
        
    # 3. Momentum (RSI)
    rsi = last["rsi"]
    if rsi > 80: s_mom = 100
    elif rsi > 70: s_mom = 80
    elif rsi < 30: s_mom = 15
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
    logger.debug(
        "Symbol=%s RiskScore=%.1f RSI=%.1f ZDist=%.2f %s GlobalVIX=%.1f",
        symbol,
        risk_score,
        last["rsi"],
        last["z_dist"],
        pe_str,
        global_vix,
    )
    
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
            ctx["rule_id"] = rule_id
            ctx["severity"] = severity
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
        emit("RISK_EXTREME", "high", f"极端风险 ({risk_score:.0f}/100)：市场严重过热或结构性崩溃。操作建议：严格止盈 / 大幅降低仓位。")
    elif risk_score >= 60:
        emit("RISK_HIGH", "med", f"高风险 ({risk_score:.0f}/100)：价格过度偏离均值或极度恐慌。操作建议：考虑减仓，控制风险敞口。")
    elif risk_score <= 25:
        emit("RISK_LOW", "buy", f"低风险区间 ({risk_score:.0f}/100)：趋势健康，未过热。长线视角下的逢低布局机会。")

    # 2. 超卖机会信号（独立于风险评分，专为长线买点设计）
    rsi_val = float(last["rsi"])
    z_val = float(last["z_dist"])
    if rsi_val < 35 and z_val < -1.5:
        emit(
            "OVERSOLD_OPP",
            "buy",
            f"超卖机会：RSI={rsi_val:.1f}（超卖区间），价格偏离均线={z_val:.2f}σ（严重低估）。"
            f"长线买入机会区间，建议分批布局。",
        )

    # 3. Event Shock (Keep original logic but integrate)
    vol_z = last["vol_z"]
    rv_last = last["rv"]
    rv_mean10 = feat["rv"].tail(10).mean()
    if vol_z >= cfg.volume_z_hi and rv_last > rv_mean10:
        emit("EVENT_SHOCK", "high", f"成交量与波动率异常放大 (vol_z={vol_z:.2f})，市场可能发生结构性变化，注意风险。")

    return alerts


def append_alerts_jsonl(alerts: List[Dict[str, Any]], path: str = "data/alerts.jsonl") -> None:
    if not alerts:
        return
    _ensure_data_dir()
    with open(path, "a", encoding="utf-8") as f:
        for a in alerts:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
