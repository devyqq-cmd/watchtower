# jobs/backtest.py
from __future__ import annotations

import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from alerts.engine import compute_features, calculate_risk_score
from alerts.rules import AlertConfig
from app.providers.analytics.performance import summarize
from app.providers.store.sqlite_store import SQLiteStore


@dataclass
class BacktestConfig:
    """回测参数配置。"""
    cost_bps: float = 10.0          # 单边交易成本，基点（10bps = 0.1%）
    vix: float = 20.0               # 固定 VIX（基础回测用，Walk-Forward 中可替换）
    valuation: dict = field(default_factory=lambda: {"trailingPE": 25.0})
    # 仓位映射：risk_score -> position_fraction
    pos_high_risk: float = 0.2      # score >= 80
    pos_mid_risk: float = 0.5       # score >= 65
    pos_low_risk: float = 1.0       # score <= 35
    alert_cfg: Optional[AlertConfig] = None

    def __post_init__(self):
        if self.alert_cfg is None:
            self.alert_cfg = AlertConfig()


class BacktestEngine:
    """
    向量化回测引擎。

    设计原则：
    1. 特征一次性计算（rolling window 是 causal 的，无前视偏差）
    2. 信号在 t 时刻生成，仓位在 t+1 时刻才生效（position_next）
    3. 交易成本按仓位变化量计算
    """

    def __init__(self, df: pd.DataFrame, cfg: BacktestConfig):
        """
        Args:
            df: OHLCV DataFrame，必须包含 [ts, open, high, low, close, volume]
                ts 必须是 UTC-aware datetime
            cfg: 回测配置
        """
        self.df = df.copy().sort_values("ts").reset_index(drop=True)
        self.cfg = cfg
        self._min_bars = cfg.alert_cfg.min_bars  # type: ignore

    def _compute_signals(self) -> pd.DataFrame:
        """计算全量特征和风险评分，返回带 risk_score 列的 DataFrame。"""
        feat = compute_features(self.df, self.cfg.alert_cfg)  # type: ignore
        feat["risk_score"] = feat.apply(
            lambda row: calculate_risk_score(
                row, self.cfg.alert_cfg,  # type: ignore
                global_vix=self.cfg.vix,
                valuation=self.cfg.valuation,
            )
            if not pd.isna(row["rsi"])
            else np.nan,
            axis=1,
        )
        return feat

    def _signals_to_positions(self, risk_scores: pd.Series) -> pd.Series:
        """将风险评分序列转换为目标仓位序列。"""
        pos = pd.Series(self.cfg.pos_low_risk, index=risk_scores.index)
        pos[risk_scores >= 80] = self.cfg.pos_high_risk
        pos[(risk_scores >= 65) & (risk_scores < 80)] = self.cfg.pos_mid_risk
        pos[risk_scores <= 35] = self.cfg.pos_low_risk
        # 信号不足时（NaN）持默认仓位
        pos[risk_scores.isna()] = self.cfg.pos_low_risk
        return pos

    def run(self) -> pd.DataFrame:
        """
        运行回测，返回逐 bar 结果。

        Returns DataFrame with columns:
            ts, price, risk_score, position, strat_return, bench_return
        """
        feat = self._compute_signals()

        # 目标仓位（t 时刻信号）
        target_pos = self._signals_to_positions(feat["risk_score"])

        # 实际仓位滞后一期（t+1 才能按 t 的信号建仓）
        actual_pos = target_pos.shift(1).fillna(self.cfg.pos_low_risk)

        # 每日收益
        bench_ret = feat["close"].pct_change().fillna(0.0)

        # 交易成本：仓位变化量 * cost_bps / 10000
        pos_change = actual_pos.diff().abs().fillna(0.0)
        cost = pos_change * (self.cfg.cost_bps / 10_000)

        # 策略收益
        strat_ret = actual_pos * bench_ret - cost

        result = pd.DataFrame({
            "ts": feat["ts"],
            "price": feat["close"],
            "risk_score": feat["risk_score"],
            "position": actual_pos,
            "strat_return": strat_ret,
            "bench_return": bench_ret,
        })

        # 只保留有足够历史的 bar（前 min_bars 根不可靠）
        result = result.iloc[self._min_bars:].reset_index(drop=True)
        result.iloc[0, result.columns.get_loc("strat_return")] = 0.0
        result.iloc[0, result.columns.get_loc("bench_return")] = 0.0

        return result

    def run_split(self, is_ratio: float = 0.75) -> dict:
        """
        IS/OOS 分割回测。

        Args:
            is_ratio: 样本内占比（默认 75%，剩余 25% 为样本外）

        Returns:
            {
                "is_metrics": {...},   # 样本内绩效
                "oos_metrics": {...},  # 样本外绩效（真实表现）
                "full_result": DataFrame,
            }
        """
        result = self.run()
        split_idx = int(len(result) * is_ratio)

        is_df = result.iloc[:split_idx]
        oos_df = result.iloc[split_idx:]

        is_metrics = summarize(
            is_df["strat_return"], is_df["bench_return"], label="IS"
        )
        oos_metrics = summarize(
            oos_df["strat_return"], oos_df["bench_return"], label="OOS"
        )

        return {
            "is_metrics": is_metrics,
            "oos_metrics": oos_metrics,
            "full_result": result,
        }

    def save_results(self, result: pd.DataFrame, path: str = "data/backtest_results.csv") -> None:
        from pathlib import Path
        os.makedirs(Path(path).parent, exist_ok=True)
        result.to_csv(path, index=False)
        print(f"[backtest] Results saved to {path}")


def run_backtest(symbol: str, db_path: str = "watchtower.db", is_ratio: float = 0.75) -> dict:
    """
    CLI 入口：从 SQLite 读取完整历史，运行 IS/OOS 回测并打印报告。
    """
    store = SQLiteStore(db_path=db_path)
    with store._connect() as conn:
        df = pd.read_sql(
            "SELECT ts, open, high, low, close, volume FROM prices WHERE ticker = ? ORDER BY ts ASC",
            conn, params=(symbol,)
        )

    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    cfg = BacktestConfig(alert_cfg=AlertConfig.load_evolution())
    min_required = cfg.alert_cfg.ema_slow + cfg.alert_cfg.min_bars
    if df.empty or len(df) < min_required:
        print(f"[backtest] 数据不足（{len(df)} 根 bar），需要至少 {min_required} 根。请先运行 ingest。")
        return {}
    engine = BacktestEngine(df, cfg)
    report = engine.run_split(is_ratio=is_ratio)

    _print_report(symbol, report)
    engine.save_results(report["full_result"])
    return report


def _print_report(symbol: str, report: dict) -> None:
    is_m = report["is_metrics"]
    oos_m = report["oos_metrics"]

    print(f"\n{'='*55}")
    print(f"  回测报告: {symbol}")
    print(f"{'='*55}")
    print(f"{'指标':<20} {'样本内(IS)':>12} {'样本外(OOS)':>12}")
    print(f"{'-'*55}")

    def fmt(m: dict, key: str, pct: bool = False) -> str:
        v = m.get(key, float("nan"))
        if pct:
            return f"{v:>+11.1%}"
        return f"{v:>12.2f}"

    rows = [
        ("总收益", "total_return", True),
        ("基准收益", "benchmark_return", True),
        ("年化收益", "annual_return", True),
        ("Sharpe", "sharpe", False),
        ("最大回撤", "max_drawdown", True),
        ("Calmar", "calmar", False),
        ("t-stat (超额)", "t_stat", False),
        ("样本数(Bar)", "n_bars", False),
    ]
    for label, key, pct in rows:
        print(f"{label:<20} {fmt(is_m, key, pct)} {fmt(oos_m, key, pct)}")

    print(f"{'='*55}")
    t = oos_m.get("t_stat", 0)
    verdict = "✓ 统计显著 (t>2)" if abs(t) >= 2.0 else "✗ 统计噪音 (t<2)，策略无效"
    print(f"  OOS 结论: {verdict}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    import sys as _sys
    ticker = _sys.argv[1] if len(_sys.argv) > 1 else "0700.HK"
    db = _sys.argv[2] if len(_sys.argv) > 2 else "watchtower.db"
    run_backtest(ticker, db_path=db)
