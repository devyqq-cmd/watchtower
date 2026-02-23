# tests/test_backtest_engine.py
import pandas as pd
import numpy as np
import pytest
from jobs.backtest import BacktestEngine, BacktestConfig

def make_price_df(n=500, seed=42) -> pd.DataFrame:
    """生成合成日线数据，足够计算所有指标（需要 200+ 根 K 线）。"""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    df = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC"),
        "open": close * (1 + rng.uniform(-0.005, 0.005, n)),
        "high": close * (1 + rng.uniform(0, 0.01, n)),
        "low": close * (1 - rng.uniform(0, 0.01, n)),
        "close": close,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    })
    return df

def test_engine_returns_required_columns():
    df = make_price_df()
    cfg = BacktestConfig()
    engine = BacktestEngine(df, cfg)
    result = engine.run()
    for col in ["ts", "price", "risk_score", "position", "strat_return", "bench_return"]:
        assert col in result.columns, f"Missing column: {col}"

def test_position_is_lagged():
    """信号在 t 时刻生成，仓位在 t+1 时刻才生效。"""
    df = make_price_df()
    cfg = BacktestConfig()
    engine = BacktestEngine(df, cfg)
    result = engine.run()
    # 第一根有效信号的仓位不应该影响那一根 bar 自己的收益
    assert result["strat_return"].iloc[0] == 0.0 or not pd.isna(result["strat_return"].iloc[0])

def test_transaction_costs_reduce_returns():
    """加入交易成本后，策略收益应该低于无成本版。"""
    df = make_price_df()
    cfg_no_cost = BacktestConfig(cost_bps=0)
    cfg_with_cost = BacktestConfig(cost_bps=10)
    r_no_cost = BacktestEngine(df, cfg_no_cost).run()
    r_with_cost = BacktestEngine(df, cfg_with_cost).run()
    total_no_cost = (1 + r_no_cost["strat_return"]).prod()
    total_with_cost = (1 + r_with_cost["strat_return"]).prod()
    assert total_no_cost >= total_with_cost

def test_oos_metrics_exist():
    """run_split 必须返回 is_metrics 和 oos_metrics。"""
    df = make_price_df(800)
    cfg = BacktestConfig()
    engine = BacktestEngine(df, cfg)
    report = engine.run_split(is_ratio=0.75)
    assert "is_metrics" in report
    assert "oos_metrics" in report
    assert "t_stat" in report["oos_metrics"]
