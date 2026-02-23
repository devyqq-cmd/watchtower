# app/providers/analytics/performance.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def sharpe_ratio(returns: pd.Series, periods: int = 252, risk_free: float = 0.0) -> float:
    """年化 Sharpe 比率。periods=252 为日线，52 为周线。risk_free 为年化无风险利率（如 0.03 = 3%）。"""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods
    std = excess.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float((excess.mean() / std) * np.sqrt(periods))


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤，返回负数（如 -0.25 表示 -25%）。输入为净值序列（wealth index），不应含零或负值。"""
    if len(equity) < 2:
        return 0.0
    roll_max = equity.cummax()
    # Guard against zero/negative equity (shouldn't happen in wealth index but be safe)
    roll_max = roll_max.replace(0, np.nan)
    drawdown = (equity - roll_max) / roll_max
    mdd = drawdown.min()
    return float(mdd) if not pd.isna(mdd) else 0.0


def calmar_ratio(returns: pd.Series, equity: pd.Series, periods: int = 252) -> float:
    """Calmar = 年化收益(CAGR) / |最大回撤|。"""
    if len(returns) < 2 or len(equity) < 2:
        return 0.0
    # Use geometric CAGR: (end/start)^(periods/n) - 1
    n = len(returns)
    start_val = equity.iloc[0]
    end_val = equity.iloc[-1]
    if start_val <= 0:
        return 0.0
    cagr = (end_val / start_val) ** (periods / n) - 1
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return float("inf")
    return float(cagr / mdd)


def t_stat_alpha(returns: pd.Series) -> float:
    """单样本 t-test：检验收益均值是否显著异于 0。t > 2 代表 p < 0.05。"""
    if len(returns) < 2:
        return 0.0
    t, _ = stats.ttest_1samp(returns.dropna(), popmean=0)
    return float(t)


def summarize(
    strat_returns: pd.Series,
    bench_returns: pd.Series,
    periods: int = 252,
    label: str = "Strategy",
) -> dict:
    """生成完整的绩效摘要字典，方便打印或存储。"""
    strat_equity = (1 + strat_returns).cumprod()
    bench_equity = (1 + bench_returns).cumprod()

    excess_returns = strat_returns - bench_returns

    return {
        "label": label,
        "total_return": float(strat_equity.iloc[-1] - 1),
        "benchmark_return": float(bench_equity.iloc[-1] - 1),
        "sharpe": sharpe_ratio(strat_returns, periods),
        "max_drawdown": max_drawdown(strat_equity),
        "calmar": calmar_ratio(strat_returns, strat_equity, periods),
        "t_stat": t_stat_alpha(excess_returns),
        "n_bars": len(strat_returns),
        "annual_return": float((1 + strat_returns.mean()) ** periods - 1),
    }
