# tests/test_param_sweep.py
import pandas as pd
import numpy as np
from jobs.param_sweep import ParamSweeper, SweepConfig

def make_price_df(n=600, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    return pd.DataFrame({
        "ts": pd.date_range("2021-01-01", periods=n, freq="B", tz="UTC"),
        "open": close, "high": close * 1.005, "low": close * 0.995,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    })

def test_sweep_returns_grid():
    df = make_price_df()
    cfg = SweepConfig(
        rsi_overbought_range=[65, 70, 75],
        z_threshold_greed_range=[1.5, 2.0, 2.5],
    )
    sweeper = ParamSweeper(df, cfg)
    grid = sweeper.run()
    assert len(grid) == 9  # 3 x 3
    assert "oos_sharpe" in grid.columns

def test_sweep_saves_csv(tmp_path):
    df = make_price_df()
    cfg = SweepConfig(
        rsi_overbought_range=[70, 75],
        z_threshold_greed_range=[2.0, 2.5],
    )
    sweeper = ParamSweeper(df, cfg)
    grid = sweeper.run()
    out = str(tmp_path / "sweep.csv")
    sweeper.save(grid, out)
    loaded = pd.read_csv(out)
    assert len(loaded) == 4
