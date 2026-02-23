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
    rsi_overbought_range: List[int] = field(default_factory=lambda: list(range(65, 82, 3)))
    z_threshold_greed_range: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])
    cost_bps: float = 10.0
    is_ratio: float = 0.75  # OOS is 25%


class ParamSweeper:
    def __init__(self, df: pd.DataFrame, cfg: SweepConfig):
        self.df = df.copy().sort_values("ts").reset_index(drop=True)
        self.cfg = cfg

    def run(self) -> pd.DataFrame:
        rows = []
        combos = list(product(
            self.cfg.rsi_overbought_range,
            self.cfg.z_threshold_greed_range,
        ))
        total = len(combos)
        for i, (rsi_ob, z_greed) in enumerate(combos, 1):
            print(f"  [{i}/{total}] rsi_ob={rsi_ob}, z_greed={z_greed:.1f}", end="\r")
            alert_cfg = AlertConfig(
                rsi_overbought=rsi_ob,
                z_threshold_greed=z_greed,
            )
            bt_cfg = BacktestConfig(
                cost_bps=self.cfg.cost_bps,
                alert_cfg=alert_cfg,
            )
            try:
                engine = BacktestEngine(self.df, bt_cfg)
                report = engine.run_split(is_ratio=self.cfg.is_ratio)
                oos = report["oos_metrics"]
                rows.append({
                    "rsi_overbought": rsi_ob,
                    "z_threshold_greed": z_greed,
                    "oos_sharpe": round(oos.get("sharpe", float("nan")), 3),
                    "oos_t_stat": round(oos.get("t_stat", float("nan")), 3),
                    "oos_mdd": round(oos.get("max_drawdown", float("nan")), 4),
                    "oos_total_return": round(oos.get("total_return", float("nan")), 4),
                })
            except Exception:
                rows.append({
                    "rsi_overbought": rsi_ob,
                    "z_threshold_greed": z_greed,
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
            index="rsi_overbought",
            columns="z_threshold_greed",
            values="oos_sharpe"
        )
        print("\nOOS Sharpe 热力图 (行=RSI超买阈值, 列=Z过热阈值):")
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
    print(f"开始参数扫描: {symbol}，共 {len(cfg.rsi_overbought_range) * len(cfg.z_threshold_greed_range)} 组参数...")
    grid = sweeper.run()
    sweeper.print_heatmap(grid)
    sweeper.save(grid)
