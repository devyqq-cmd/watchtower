from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import json
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

from alerts import AlertConfig, evaluate_alerts, compute_features, calculate_risk_score
from app.providers.store.sqlite_store import SQLiteStore

def run_backtest(symbol: str, days_back: int = 30):
    print(f"--- [Backtest Engine] Starting Simulation for {symbol} ---")
    store = SQLiteStore(db_path="watchtower.db")
    
    # 1. Fetch full history for indicators
    with store._connect() as conn:
        df = pd.read_sql(
            "SELECT ts, open, high, low, close, volume FROM prices WHERE ticker = ? ORDER BY ts ASC",
            conn, params=(symbol,)
        )
    
    if df.empty or len(df) < 150:
        print(f"Not enough historical data for {symbol} (Found {len(df)} bars).")
        return

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    
    # Start simulation from N days ago
    now = datetime.now(timezone.utc)
    sim_start_date = now - timedelta(days=days_back)
    
    results = []
    equity_hold = 1.0  # Buy and Hold Benchmark
    equity_strat = 1.0 # Strategy Equity
    position = 1.0     # 1.0 = Full Long, 0.5 = Half, etc.
    
    current_cfg = AlertConfig.load_evolution()
    
    # Simulation Loop: Day by Day
    sim_dates = df[df["ts"] >= sim_start_date]["ts"].unique()
    
    print(f"Simulating {len(sim_dates)} days...")
    
    for i, sim_date in enumerate(sim_dates):
        # Only look at data available up to sim_date
        df_slice = df[df["ts"] <= sim_date].copy()
        if len(df_slice) < 100: continue
        
        # Calculate features and risk score at THAT MOMENT
        feat = compute_features(df_slice, current_cfg)
        last = feat.iloc[-1]
        
        # (Simplified: assume constant VIX/Valuation for backtest demo)
        risk_score = calculate_risk_score(last, current_cfg, global_vix=20.0, valuation={"trailingPE": 25.0})
        
        # Strategy Logic: Discipline Enforcement
        # Reduce position on high risk, increase on low risk
        prev_pos = position
        if risk_score >= 80:
            position = 0.2  # Heavy Sell-off
        elif risk_score >= 65:
            position = 0.5  # Partial Profit taking
        elif risk_score <= 35:
            position = 1.0  # Buy back / Re-entry
            
        # Update Equity Curves based on the NEXT day's return
        if i < len(sim_dates) - 1:
            next_day_ret = (df[df["ts"] == sim_dates[i+1]]["close"].values[0] / last["close"]) - 1
            equity_hold *= (1 + next_day_ret)
            equity_strat *= (1 + (next_day_ret * prev_pos))
            
        results.append({
            "ts": sim_date.isoformat(),
            "price": last["close"],
            "score": risk_score,
            "position": prev_pos,
            "equity_hold": equity_hold,
            "equity_strat": equity_strat
        })

    # Save to CSV for visualization
    res_df = pd.DataFrame(results)
    os.makedirs("data", exist_ok=True)
    res_df.to_csv("data/backtest_results.csv", index=False)
    
    def calc_mdd(series):
        rollup = series.cummax()
        drawdown = (series - rollup) / rollup
        return float(drawdown.min())

    mdd_hold = calc_mdd(res_df["equity_hold"])
    mdd_strat = calc_mdd(res_df["equity_strat"])
    final_ret_hold = (res_df["equity_hold"].iloc[-1] - 1)
    final_ret_strat = (res_df["equity_strat"].iloc[-1] - 1)

    print(f"\n--- [Backtest Report: {symbol}] ---")
    print(f"Period: Last {days_back} days")
    print(f"Final Return (Buy & Hold): {final_ret_hold:+.2%}")
    print(f"Final Return (AI Strategy): {final_ret_strat:+.2%}")
    print(f"Max Drawdown (Buy & Hold): {mdd_hold:.2%}")
    print(f"Max Drawdown (AI Strategy): {mdd_strat:.2%}")
    
    # Return metrics for integration
    return {
        "mdd_hold": mdd_hold,
        "mdd_strat": mdd_strat,
        "ret_hold": final_ret_hold,
        "ret_strat": final_ret_strat
    }

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "0700.HK"
    run_backtest(ticker, 60) # Test 60 days
