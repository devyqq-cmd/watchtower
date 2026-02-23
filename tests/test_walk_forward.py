# tests/test_walk_forward.py
import pandas as pd
import numpy as np
from jobs.walk_forward import WalkForwardRunner, WalkForwardConfig

def make_price_df(n=1500, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    return pd.DataFrame({
        "ts": pd.date_range("2019-01-01", periods=n, freq="B", tz="UTC"),
        "open": close,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    })

def test_walk_forward_produces_windows():
    df = make_price_df(1500)
    cfg = WalkForwardConfig(train_bars=500, test_bars=125, step_bars=63)
    runner = WalkForwardRunner(df, cfg)
    results = runner.run()
    assert len(results) >= 2, "需要至少 2 个 Walk-Forward 窗口"

def test_each_window_has_metrics():
    df = make_price_df(1500)
    cfg = WalkForwardConfig(train_bars=500, test_bars=125, step_bars=63)
    runner = WalkForwardRunner(df, cfg)
    results = runner.run()
    for w in results:
        assert "oos_sharpe" in w
        assert "oos_t_stat" in w
        assert "window" in w

def test_summary_aggregates_oos():
    df = make_price_df(1500)
    cfg = WalkForwardConfig(train_bars=500, test_bars=125, step_bars=63)
    runner = WalkForwardRunner(df, cfg)
    results = runner.run()
    summary = runner.summarize(results)
    assert "mean_oos_sharpe" in summary
    assert "mean_oos_t_stat" in summary
    assert "n_windows" in summary
