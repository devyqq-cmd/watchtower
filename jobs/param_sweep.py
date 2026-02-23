# jobs/param_sweep.py
from __future__ import annotations

import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import os
from dataclasses import dataclass, field
from typing import List
from itertools import product

import pandas as pd

from alerts.rules import AlertConfig
from jobs.backtest import BacktestConfig, BacktestEngine


@dataclass
class SweepConfig:
    # 扫描真正影响仓位决策的评分阈值
    score_sell_soft_range: List[float] = field(default_factory=lambda: [40.0, 45.0, 50.0, 55.0, 60.0, 65.0])
    score_sell_hard_range: List[float] = field(default_factory=lambda: [55.0, 60.0, 65.0, 70.0, 75.0, 80.0])
    cost_bps: float = 10.0
    is_ratio: float = 0.75  # OOS 占 25%


class ParamSweeper:
    def __init__(self, df: pd.DataFrame, cfg: SweepConfig):
        self.df = df.copy().sort_values("ts").reset_index(drop=True)
        self.cfg = cfg

    def run(self) -> pd.DataFrame:
        rows = []
        # 只取 soft < hard 的合法组合，避免无意义参数
        combos = [
            (soft, hard)
            for soft, hard in product(self.cfg.score_sell_soft_range, self.cfg.score_sell_hard_range)
            if soft < hard
        ]
        total = len(combos)
        for i, (soft, hard) in enumerate(combos, 1):
            print(f"  [{i}/{total}] soft={soft:.0f}, hard={hard:.0f}", end="\r")
            bt_cfg = BacktestConfig(
                cost_bps=self.cfg.cost_bps,
                score_sell_soft=soft,
                score_sell_hard=hard,
            )
            try:
                engine = BacktestEngine(self.df, bt_cfg)
                report = engine.run_split(is_ratio=self.cfg.is_ratio)
                oos = report["oos_metrics"]
                rows.append({
                    "score_sell_soft": soft,
                    "score_sell_hard": hard,
                    "oos_sharpe": round(oos.get("sharpe", float("nan")), 3),
                    "oos_t_stat": round(oos.get("t_stat", float("nan")), 3),
                    "oos_mdd": round(oos.get("max_drawdown", float("nan")), 4),
                    "oos_total_return": round(oos.get("total_return", float("nan")), 4),
                })
            except Exception:
                rows.append({
                    "score_sell_soft": soft,
                    "score_sell_hard": hard,
                    "oos_sharpe": float("nan"),
                    "oos_t_stat": float("nan"),
                    "oos_mdd": float("nan"),
                    "oos_total_return": float("nan"),
                })

        print()  # newline after \r
        return pd.DataFrame(rows)

    def save(self, grid: pd.DataFrame, path: str = "data/param_sweep.csv") -> None:
        os.makedirs(Path(path).parent, exist_ok=True)
        grid.to_csv(path, index=False)
        print(f"[param_sweep] 结果已保存: {path}")

    def print_heatmap(self, grid: pd.DataFrame) -> None:
        pivot = grid.pivot(
            index="score_sell_soft",
            columns="score_sell_hard",
            values="oos_sharpe"
        )
        print("\nOOS Sharpe 热力图 (行=软阈值, 列=硬阈值):")
        print(pivot.to_string(float_format="{:.2f}".format))
        n_positive = (grid["oos_sharpe"] > 0).sum()
        n_total = len(grid)
        print(f"\n正 Sharpe 参数组合: {n_positive}/{n_total} ({n_positive/n_total:.0%})")
        if n_positive / n_total > 0.6:
            print("✓ 策略对参数变化稳健（>60% 参数组合有正 Sharpe）")
        else:
            print("✗ 策略对参数敏感（<60% 参数组合有正 Sharpe），可能过拟合")


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

    if df.empty or len(df) < 300:
        print(f"数据不足（{len(df)} 根 bar）")
        _sys.exit(1)

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    cfg = SweepConfig()
    sweeper = ParamSweeper(df, cfg)
    n_combos = sum(1 for s in cfg.score_sell_soft_range for h in cfg.score_sell_hard_range if s < h)
    print(f"开始参数扫描: {symbol}，共 {n_combos} 组有效参数组合（soft < hard）...")
    grid = sweeper.run()
    sweeper.print_heatmap(grid)
    sweeper.save(grid)
