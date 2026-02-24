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
    for col in ["ts", "price", "risk_score", "bear_market", "position", "strat_return", "bench_return"]:
        assert col in result.columns, f"Missing column: {col}"

def test_position_is_lagged():
    """仓位必须滞后一期：t 时刻信号在 t+1 时刻才生效。
    验证方法：直接对比 position 列与 risk_score 信号列。
    在第 i 行，position[i] 应该等于由 risk_score[i-1] 决定的目标仓位。
    """
    df = make_price_df(600)
    cfg = BacktestConfig()
    engine = BacktestEngine(df, cfg)
    result = engine.run()

    # 重建信号->仓位映射（含熊市保护逻辑）
    def score_to_pos(score, bear, cfg):
        if pd.isna(score):
            base = cfg.pos_low_risk
        elif score >= cfg.score_sell_hard:
            base = cfg.pos_high_risk
        elif score >= cfg.score_sell_soft:
            base = cfg.pos_mid_risk
        else:
            base = cfg.pos_low_risk
        if cfg.use_bear_protection and bear:
            base = min(base, cfg.pos_bear_market)
        return base

    # 验证几个非第一行：position[i] 应来自 risk_score[i-1] + bear_market[i-1]
    for i in range(1, min(10, len(result))):
        prev_score = result["risk_score"].iloc[i - 1]
        prev_bear = bool(result["bear_market"].iloc[i - 1])
        expected_pos = score_to_pos(prev_score, prev_bear, cfg)
        actual_pos = result["position"].iloc[i]
        assert abs(actual_pos - expected_pos) < 1e-9, (
            f"Row {i}: position={actual_pos} but score[{i-1}]={prev_score:.1f}, "
            f"bear[{i-1}]={prev_bear} should yield pos={expected_pos}"
        )

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

def test_bear_market_protection_caps_position():
    """熊市保护：死叉期间仓位不超过 pos_bear_market。"""
    # 构造明确的熊市数据：价格从高位持续下跌，最终跌破 EMA200
    rng = np.random.default_rng(99)
    n = 400
    # 前200根上涨建立牛市，后200根持续下跌形成死叉
    up = 100 * np.cumprod(1 + rng.normal(0.002, 0.01, 200))
    down = up[-1] * np.cumprod(1 + rng.normal(-0.003, 0.01, 200))
    close = np.concatenate([up, down])
    df = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC"),
        "open": close, "high": close * 1.005, "low": close * 0.995,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    })

    cfg = BacktestConfig(use_bear_protection=True, pos_bear_market=0.3)
    engine = BacktestEngine(df, cfg)
    result = engine.run()

    # 仓位滞后一期：bear_market[i-1]=True 时，position[i] 不得超过 0.3
    # （第一根 bear_market 变 True 的 bar，position 仍是前一期的值，这是正确行为）
    lagged_bear = result["bear_market"].shift(1, fill_value=False)
    after_bear_rows = result[lagged_bear == True]
    if not after_bear_rows.empty:
        assert after_bear_rows["position"].max() <= 0.3 + 1e-9, (
            f"熊市信号后下一期仓位超过上限: {after_bear_rows['position'].max()}"
        )

    # 关闭熊市保护时仓位可以回到满仓
    cfg_no_bear = BacktestConfig(use_bear_protection=False)
    result_no_bear = BacktestEngine(df, cfg_no_bear).run()
    lagged_bear_no = result_no_bear["bear_market"].shift(1, fill_value=False)
    after_bear_no = result_no_bear[lagged_bear_no == True]
    if not after_bear_no.empty:
        assert after_bear_no["position"].max() > 0.3, "关闭熊市保护后满仓应该可以出现"
