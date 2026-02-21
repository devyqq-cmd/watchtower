from __future__ import annotations
import json
import os
from datetime import datetime, timedelta, timezone
from dataclasses import asdict

import pandas as pd
from alerts import AlertConfig
from app.providers.store.sqlite_store import SQLiteStore
from ai.analyst import AINarrativeAnalyst

DB_PATH = "watchtower.db"
ALERTS_PATH = "data/alerts.jsonl"
EVOLUTION_PATH = "data/evolution.json"

def get_price_at(store: SQLiteStore, ticker: str, target_ts: datetime, window_days: int = 2) -> float | None:
    """
    Finds the closing price closest to the target timestamp.
    """
    start_str = (target_ts - timedelta(days=window_days)).isoformat()
    end_str = (target_ts + timedelta(days=window_days)).isoformat()
    
    with store._connect() as conn:
        cursor = conn.execute(
            "SELECT ts, close FROM prices WHERE ticker = ? AND ts BETWEEN ? AND ? ORDER BY ts ASC",
            (ticker, start_str, end_str)
        )
        rows = cursor.fetchall()
        
    if not rows:
        return None
        
    # Find closest row by time
    best_p = None
    min_diff = float("inf")
    
    for r_ts_str, r_close in rows:
        r_ts = datetime.fromisoformat(r_ts_str).replace(tzinfo=timezone.utc)
        diff = abs((r_ts - target_ts).total_seconds())
        if diff < min_diff:
            min_diff = diff
            best_p = r_close
            
    return best_p

def run_weekly_review():
    """
    Analyzes past alerts, calculates 'Win Rate', and evolves system weights.
    """
    print("--- [Weekly Review] Starting Self-Correction Protocol ---")
    store = SQLiteStore(db_path=DB_PATH)
    
    if not os.path.exists(ALERTS_PATH):
        print("No alerts history found. Cannot review.")
        return

    alerts = []
    with open(ALERTS_PATH, "r") as f:
        for line in f:
            if line.strip():
                alerts.append(json.loads(line))
                
    now = datetime.now(timezone.utc)
    review_window = timedelta(days=7) # Look back at alerts from >7 days ago
    
    analyzed_count = 0
    success_count = 0
    false_positives = 0
    
    results = []

    for alert in alerts:
        ts_str = alert.get("ts")
        if not ts_str: continue
        
        alert_ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        
        # Only review alerts that are 'mature' (older than 5 days) but not ancient (< 30 days)
        age = (now - alert_ts).days
        if age < 5 or age > 30:
            continue
            
        symbol = alert.get("symbol")
        risk_score = alert.get("context", {}).get("risk_score", 0)
        
        # Only care about High Risk alerts for now (Capital Preservation)
        if risk_score < 60:
            continue
            
        # Get Price at Alert
        p_start = get_price_at(store, symbol, alert_ts)
        # Get Price 5 days later
        p_end = get_price_at(store, symbol, alert_ts + timedelta(days=5))
        
        if p_start and p_end:
            ret_5d = (p_end - p_start) / p_start
            outcome = "UNKNOWN"
            
            # Logic: If Risk was High, we want Price to DROP (Success) or Stay Flat.
            # If Price went UP significantly (>2%), it was a False Alarm (Opportunity Cost).
            if ret_5d < -0.02:
                outcome = "SUCCESS (Avoided Drop)"
                success_count += 1
            elif ret_5d > 0.02:
                outcome = "FAILURE (False Alarm)"
                false_positives += 1
            else:
                outcome = "NEUTRAL"
                
            results.append({
                "symbol": symbol,
                "ts": ts_str,
                "score": risk_score,
                "return_5d": ret_5d,
                "outcome": outcome
            })
            analyzed_count += 1

    if analyzed_count == 0:
        print("[Review] No mature alerts found to analyze yet. (System needs to run for >5 days)")
        return

    # --- Self-Evolution Logic ---
    win_rate = success_count / analyzed_count if analyzed_count > 0 else 0
    print(f"[Review] Analyzed {analyzed_count} high-risk alerts.")
    print(f"[Review] Success Rate (Avoided Loss): {win_rate:.1%}")
    print(f"[Review] False Alarm Rate: {false_positives/analyzed_count:.1%}")
    
    # Load current evolution state
    current_cfg = AlertConfig.load_evolution()
    new_weights = current_cfg.weights.copy()
    
    evolution_log = []
    
    # Heuristic Evolution Rule:
    # If False Alarms are high (> 40%), it means we are too sensitive.
    # We should increase the weight of 'Valuation' (Value Investing) and reduce 'Momentum' (Noise).
    if (false_positives / analyzed_count) > 0.4:
        print("[Evolution] 🧬 Detected High False Alarm Rate. Becoming more conservative...")
        new_weights["valuation"] = min(new_weights.get("valuation", 0.2) + 0.1, 0.5)
        new_weights["momentum"] = max(new_weights.get("momentum", 0.2) - 0.05, 0.05)
        evolution_log.append("Increased Valuation weight, Decreased Momentum weight.")
        
    # If Win Rate is high (> 70%), we can afford to be slightly more aggressive to catch tops earlier.
    elif win_rate > 0.7:
        print("[Evolution] 🧬 System is performing well. Optimizing for earlier detection...")
        new_weights["trend"] = min(new_weights.get("trend", 0.25) + 0.05, 0.4)
        evolution_log.append("Increased Trend weight for earlier detection.")
        
    # Save Evolution
    evo_data = {
        "last_review": now.isoformat(),
        "stats": {
            "win_rate": win_rate,
            "false_positive_rate": false_positives/analyzed_count
        },
        "weights": new_weights,
        "log": evolution_log
    }
    
    with open(EVOLUTION_PATH, "w") as f:
        json.dump(evo_data, f, indent=2)
        
    print(f"[System] 🧬 Evolution Complete. Updated weights saved to {EVOLUTION_PATH}")
    print(f"[System] New Weights: {new_weights}")

if __name__ == "__main__":
    run_weekly_review()
