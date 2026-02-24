# tests/test_performance.py
import numpy as np
import pandas as pd
from app.providers.analytics.performance import (
    sharpe_ratio, max_drawdown, calmar_ratio, t_stat_alpha
)

def test_sharpe_ratio_perfect_returns():
    # 每天 0.1% 稳定收益
    returns = pd.Series([0.001] * 252)
    sr = sharpe_ratio(returns, periods=252)
    assert sr > 5.0  # 完美收益 Sharpe 应该很高

def test_sharpe_ratio_zero_returns():
    returns = pd.Series([0.0] * 252)
    sr = sharpe_ratio(returns, periods=252)
    assert sr == 0.0

def test_max_drawdown_known_case():
    # 从 100 涨到 120 再跌到 90，MDD = (90-120)/120 = -25%
    equity = pd.Series([100, 110, 120, 105, 90])
    mdd = max_drawdown(equity)
    assert abs(mdd - (-0.25)) < 0.001

def test_calmar_ratio_positive():
    returns = pd.Series([0.001] * 252)
    equity = (1 + returns).cumprod()
    cr = calmar_ratio(returns, equity, periods=252)
    assert cr > 0

def test_t_stat_significant():
    # 持续正收益应该有高 t-stat
    # 用微小方差（std=0.0001 << mean=0.002）避免零方差引发 scipy 精度警告
    rng = np.random.default_rng(0)
    returns = pd.Series(0.002 + rng.normal(0, 0.0001, 100))
    t = t_stat_alpha(returns)
    assert t > 2.0

def test_t_stat_noise():
    # 随机噪音应该接近 0
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0, 0.01, 252))
    t = t_stat_alpha(returns)
    assert abs(t) < 3.0  # 不应该虚假显著
