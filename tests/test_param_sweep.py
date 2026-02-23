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
    # soft=[40,50], hard=[55,65,75] → 合法组合(soft<hard): 40<55,40<65,40<75,50<55,50<65,50<75 = 6组
    cfg = SweepConfig(
        score_sell_soft_range=[40.0, 50.0],
        score_sell_hard_range=[55.0, 65.0, 75.0],
    )
    sweeper = ParamSweeper(df, cfg)
    grid = sweeper.run()
    assert len(grid) == 6  # 只有 soft < hard 的组合
    assert "oos_sharpe" in grid.columns
    assert "score_sell_soft" in grid.columns
    assert "score_sell_hard" in grid.columns

def test_sweep_saves_csv(tmp_path):
    df = make_price_df()
    # soft=[45], hard=[60,70] → 2 组合法组合
    cfg = SweepConfig(
        score_sell_soft_range=[45.0],
        score_sell_hard_range=[60.0, 70.0],
    )
    sweeper = ParamSweeper(df, cfg)
    grid = sweeper.run()
    out = str(tmp_path / "sweep.csv")
    sweeper.save(grid, out)
    loaded = pd.read_csv(out)
    assert len(loaded) == 2
