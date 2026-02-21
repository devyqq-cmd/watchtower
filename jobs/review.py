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

OUTCOME_DAYS = 20      # 长线持仓：用20个交易日后的价格验证信号质量
OUTCOME_THRESHOLD = 0.05  # 5% 涨跌幅才算有效信号（长线标准）
MIN_AGE_DAYS = OUTCOME_DAYS + 2   # 信号至少要"成熟"才能评估
MAX_AGE_DAYS = 180                # 超过180天的信号不再评估


def run_weekly_review():
    """
    分析历史信号质量，区分买入/卖出信号分别计算胜率，并进化系统权重。
    长线标准：用20天后价格验证，5%涨跌幅为有效信号门槛。
    """
    print("--- [周度复盘] 开始信号质量校准 ---")
    store = SQLiteStore(db_path=DB_PATH)

    if not os.path.exists(ALERTS_PATH):
        print("未找到历史信号记录，无法复盘。")
        return

    alerts = []
    with open(ALERTS_PATH, "r") as f:
        for line in f:
            if line.strip():
                alerts.append(json.loads(line))

    now = datetime.now(timezone.utc)

    sell_analyzed = sell_success = sell_false_pos = 0
    buy_analyzed = buy_success = buy_false_pos = 0

    results = []

    for alert in alerts:
        ts_str = alert.get("ts")
        if not ts_str:
            continue

        alert_ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        age = (now - alert_ts).days

        # 只评估"已成熟"但不太古老的信号
        if age < MIN_AGE_DAYS or age > MAX_AGE_DAYS:
            continue

        symbol = alert.get("symbol")
        rule_id = alert.get("rule_id", "")
        risk_score = alert.get("context", {}).get("risk_score", 0)
        severity = alert.get("severity", "")

        is_buy_signal = severity == "buy" or rule_id in ("RISK_LOW", "OVERSOLD_OPP")
        is_sell_signal = severity in ("high", "med") or rule_id in ("RISK_EXTREME", "RISK_HIGH", "EVENT_SHOCK")

        if not is_buy_signal and not is_sell_signal:
            continue

        # 获取信号时价格 和 20天后价格
        p_start = get_price_at(store, symbol, alert_ts)
        p_end = get_price_at(store, symbol, alert_ts + timedelta(days=OUTCOME_DAYS))

        if p_start and p_end:
            ret = (p_end - p_start) / p_start
            outcome = "NEUTRAL"

            if is_sell_signal:
                # 卖出信号成功 = 20天后下跌 > 5%（成功规避风险）
                if ret < -OUTCOME_THRESHOLD:
                    outcome = "SUCCESS（规避了下跌）"
                    sell_success += 1
                elif ret > OUTCOME_THRESHOLD:
                    outcome = "FAILURE（误报，错失上涨）"
                    sell_false_pos += 1
                else:
                    outcome = "NEUTRAL"
                sell_analyzed += 1
            elif is_buy_signal:
                # 买入信号成功 = 20天后上涨 > 5%（成功捕捉机会）
                if ret > OUTCOME_THRESHOLD:
                    outcome = "SUCCESS（捕捉到上涨）"
                    buy_success += 1
                elif ret < -OUTCOME_THRESHOLD:
                    outcome = "FAILURE（误报，买在了高点）"
                    buy_false_pos += 1
                else:
                    outcome = "NEUTRAL"
                buy_analyzed += 1

            results.append({
                "symbol": symbol,
                "rule_id": rule_id,
                "ts": ts_str,
                "score": risk_score,
                "is_buy": is_buy_signal,
                f"return_{OUTCOME_DAYS}d": ret,
                "outcome": outcome,
            })

    total_analyzed = sell_analyzed + buy_analyzed
    if total_analyzed == 0:
        print(f"[复盘] 暂无可评估信号（需要信号发出 {MIN_AGE_DAYS} 天后才能评估）。")
        return

    sell_wr = sell_success / sell_analyzed if sell_analyzed > 0 else 0
    buy_wr = buy_success / buy_analyzed if buy_analyzed > 0 else 0
    sell_fpr = sell_false_pos / sell_analyzed if sell_analyzed > 0 else 0

    print(f"[复盘] 共评估 {total_analyzed} 条信号（卖出{sell_analyzed}条 / 买入{buy_analyzed}条）")
    print(f"[复盘] 卖出信号胜率：{sell_wr:.1%}  误报率：{sell_fpr:.1%}")
    print(f"[复盘] 买入信号胜率：{buy_wr:.1%}")
    
    # --- 自进化逻辑（以卖出信号为主要校准依据）---
    current_cfg = AlertConfig.load_evolution()
    new_weights = current_cfg.weights.copy()

    evolution_log = []

    # 卖出信号误报率过高（>40%）：系统过于敏感，加大估值权重、降低动量权重
    if sell_analyzed > 0 and sell_fpr > 0.4:
        print("[进化] 卖出信号误报率偏高，系统调整为更保守的估值锚定策略...")
        new_weights["valuation"] = min(new_weights.get("valuation", 0.2) + 0.1, 0.5)
        new_weights["momentum"] = max(new_weights.get("momentum", 0.15) - 0.05, 0.05)
        evolution_log.append("卖出误报率高：上调估值权重，下调动量权重。")

    # 卖出信号胜率高（>70%）：系统表现良好，适度提前捕捉顶部
    elif sell_analyzed > 0 and sell_wr > 0.7:
        print("[进化] 卖出信号质量优秀，优化提前预警能力...")
        new_weights["trend"] = min(new_weights.get("trend", 0.25) + 0.05, 0.4)
        evolution_log.append("卖出胜率高：上调趋势权重以提前预警。")

    # Save Evolution
    evo_data = {
        "last_review": now.isoformat(),
        "stats": {
            "sell_win_rate": sell_wr,
            "sell_false_positive_rate": sell_fpr,
            "buy_win_rate": buy_wr,
            "outcome_days": OUTCOME_DAYS,
        },
        "weights": new_weights,
        "log": evolution_log,
        "signal_results": results,
    }

    with open(EVOLUTION_PATH, "w") as f:
        json.dump(evo_data, f, indent=2, ensure_ascii=False)

    print(f"[进化] 权重更新完成，已保存至 {EVOLUTION_PATH}")
    print(f"[进化] 最新权重: {new_weights}")

if __name__ == "__main__":
    run_weekly_review()
