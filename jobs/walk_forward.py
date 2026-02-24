# jobs/walk_forward.py
from __future__ import annotations

import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from alerts.rules import AlertConfig
from app.providers.analytics.performance import sharpe_ratio, t_stat_alpha, max_drawdown
from jobs.backtest import BacktestConfig, BacktestEngine


@dataclass
class WalkForwardConfig:
    train_bars: int = 252    # ~1 年交易日（够 EMA200 + 指标热身）
    test_bars: int = 63      # ~1 季度（每次 OOS 检验期）
    step_bars: int = 21      # ~1 月滚动步长（最大化窗口数）
    cost_bps: float = 10.0


class WalkForwardRunner:
    def __init__(self, df: pd.DataFrame, cfg: WalkForwardConfig):
        self.df = df.copy().sort_values("ts").reset_index(drop=True)
        self.cfg = cfg

    def run(self) -> List[dict]:
        results = []
        n = len(self.df)
        start = 0
        window_idx = 0

        while start + self.cfg.train_bars + self.cfg.test_bars <= n:
            train_end = start + self.cfg.train_bars
            test_end = train_end + self.cfg.test_bars

            train_df = self.df.iloc[start:train_end].copy()
            test_df = self.df.iloc[train_end:test_end].copy()

            # Use default parameters (can be extended to param optimization)
            alert_cfg = AlertConfig.load_evolution()
            bt_cfg = BacktestConfig(cost_bps=self.cfg.cost_bps, alert_cfg=alert_cfg)

            # Combine train+test so indicators have warm-up history from train
            combined = pd.concat([train_df, test_df], ignore_index=True)
            engine = BacktestEngine(combined, bt_cfg)
            full_result = engine.run()

            # Extract only the test (OOS) portion of the result
            n_train_result = len(full_result) - self.cfg.test_bars
            if n_train_result < 0:
                start += self.cfg.step_bars
                continue

            oos_result = full_result.iloc[max(0, n_train_result):].copy()

            if len(oos_result) < 10:
                start += self.cfg.step_bars
                continue

            oos_sr = sharpe_ratio(oos_result["strat_return"])
            oos_t = t_stat_alpha(oos_result["strat_return"] - oos_result["bench_return"])
            oos_mdd = max_drawdown((1 + oos_result["strat_return"]).cumprod())

            results.append({
                "window": window_idx,
                "train_start": str(train_df["ts"].iloc[0].date()),
                "train_end": str(train_df["ts"].iloc[-1].date()),
                "test_start": str(test_df["ts"].iloc[0].date()),
                "test_end": str(test_df["ts"].iloc[-1].date()),
                "oos_sharpe": round(oos_sr, 3),
                "oos_t_stat": round(oos_t, 3),
                "oos_mdd": round(oos_mdd, 4),
                "n_bars": len(oos_result),
            })

            window_idx += 1
            start += self.cfg.step_bars

        return results

    def summarize(self, results: List[dict]) -> dict:
        if not results:
            return {}
        sharpes = [r["oos_sharpe"] for r in results]
        t_stats = [r["oos_t_stat"] for r in results]
        valid_t = [t for t in t_stats if not np.isnan(t)]
        return {
            "n_windows": len(results),
            "mean_oos_sharpe": round(float(np.mean(sharpes)), 3),
            "median_oos_sharpe": round(float(np.median(sharpes)), 3),
            "pct_positive_sharpe": round(sum(s > 0 for s in sharpes) / len(sharpes), 3),
            "mean_oos_t_stat": round(float(np.mean(valid_t)), 3) if valid_t else float("nan"),
            "pct_significant": round(sum(abs(t) >= 2 for t in valid_t) / len(valid_t), 3) if valid_t else float("nan"),
        }

    def print_report(self, results: List[dict]) -> None:
        summary = self.summarize(results)
        print(f"\n{'='*60}")
        print(f"  Walk-Forward 报告（{summary.get('n_windows', 0)} 个 OOS 窗口）")
        print(f"{'='*60}")
        print(f"{'窗口':<4} {'测试区间':<24} {'Sharpe':>8} {'t-stat':>8} {'MDD':>8}")
        print(f"{'-'*60}")
        for r in results:
            period = f"{r['test_start']} ~ {r['test_end']}"
            print(f"{r['window']:<4} {period:<24} {r['oos_sharpe']:>8.2f} {r['oos_t_stat']:>8.2f} {r['oos_mdd']:>8.1%}")
        print(f"{'-'*60}")
        print(f"{'平均':<28} {summary['mean_oos_sharpe']:>8.2f} {summary['mean_oos_t_stat']:>8.2f}")
        pct_sig = summary.get("pct_significant", float("nan"))
        pct_sig_str = f"{pct_sig:.0%}" if not np.isnan(pct_sig) else "N/A（全仓期无超额收益）"
        print(f"\n显著窗口比例: {pct_sig_str}（t≥2，仅含有超额收益的窗口）")
        print(f"正 Sharpe 比例: {summary['pct_positive_sharpe']:.0%}")
        # 主要判断依据：正 Sharpe 比例和均值；t-stat 在全仓期为 NaN 属正常
        robust = summary['mean_oos_sharpe'] > 0.3 and summary['pct_positive_sharpe'] >= 0.7
        verdict = "✓ 策略具有跨期稳健性" if robust else "✗ 策略可能过拟合（OOS 表现不一致）"
        print(f"结论: {verdict}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys as _sys
    from app.providers.store.sqlite_store import SQLiteStore

    symbol = _sys.argv[1] if len(_sys.argv) > 1 else "0700.HK"
    db_path = _sys.argv[2] if len(_sys.argv) > 2 else "watchtower.db"

    store = SQLiteStore(db_path=db_path)
    with store._connect() as conn:
        df = pd.read_sql(
            "SELECT ts, open, high, low, close, volume FROM prices WHERE ticker = ? ORDER BY ts ASC",
            conn, params=(symbol,)
        )

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    cfg = WalkForwardConfig()
    min_required = cfg.train_bars + cfg.test_bars + cfg.step_bars
    if df.empty or len(df) < min_required:
        print(f"数据不足（{len(df)} 根 bar），Walk-Forward 需要 {min_required}+ 根。")
        _sys.exit(1)
    runner = WalkForwardRunner(df, cfg)
    results = runner.run()
    runner.print_report(results)
