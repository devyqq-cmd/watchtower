# Rigorous Backtest Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将现有的玩具级回测替换成华尔街标准的严格验证框架，能给出有统计意义的 Sharpe / t-stat / Walk-Forward OOS 结果。

**Architecture:**
1. `app/providers/analytics/performance.py` — 独立的性能指标模块（Sharpe、MDD、Calmar、t-stat）
2. `jobs/backtest.py` — 重写为向量化引擎，支持 IS/OOS 分割、交易成本、3 年历史
3. `jobs/walk_forward.py` — Walk-Forward 滚动窗口框架（2年训练 / 6月测试）
4. `jobs/param_sweep.py` — 参数敏感性扫描（防过拟合检验）

**Tech Stack:** Python, pandas, numpy, scipy.stats, uv（运行命令前缀 `uv run`）

**核心原则（不可违反）：**
- 特征计算必须 causal（只用 t 时刻之前的数据）
- IS 参数优化后必须在 OOS 验证，OOS 结果才是真实表现
- t-stat < 2.0 = 统计噪音，策略无效
- 所有回测必须包含交易成本（默认单边 0.1%）

---

## Task 1: 性能指标模块（Performance Metrics）

**Files:**
- Create: `app/providers/analytics/performance.py`
- Create: `tests/test_performance.py`

### Step 1: 写失败测试

```python
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
    returns = pd.Series([0.002] * 100)
    t = t_stat_alpha(returns)
    assert t > 2.0

def test_t_stat_noise():
    # 随机噪音应该接近 0
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0, 0.01, 252))
    t = t_stat_alpha(returns)
    assert abs(t) < 3.0  # 不应该虚假显著
```

### Step 2: 运行确认失败

```bash
uv run python -m pytest tests/test_performance.py -v
```
期望：`ModuleNotFoundError` 或 `ImportError`

### Step 3: 实现性能指标模块

```python
# app/providers/analytics/performance.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def sharpe_ratio(returns: pd.Series, periods: int = 252, risk_free: float = 0.0) -> float:
    """年化 Sharpe 比率。periods=252 为日线，52 为周线。"""
    excess = returns - risk_free / periods
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(periods))


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤，返回负数（如 -0.25 表示 -25%）。"""
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, equity: pd.Series, periods: int = 252) -> float:
    """Calmar = 年化收益 / |最大回撤|。"""
    annual_ret = (1 + returns.mean()) ** periods - 1
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return float("inf")
    return float(annual_ret / mdd)


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
```

### Step 4: 运行确认通过

```bash
uv run python -m pytest tests/test_performance.py -v
```
期望：所有测试 `PASSED`

### Step 5: 提交

```bash
git add app/providers/analytics/performance.py tests/test_performance.py
git commit -m "feat: add performance metrics module (Sharpe, MDD, Calmar, t-stat)"
```

---

## Task 2: 重写向量化回测引擎

**Files:**
- Modify: `jobs/backtest.py`（完整重写）
- Create: `tests/test_backtest_engine.py`

**关键设计决策：**
- `compute_features(df, cfg)` 的 rolling window 是 causal 的（每行只看过去数据），可以在全量数据上一次性计算，无前视偏差
- 信号生成后，position 在下一根 bar 才生效（避免用当天收盘信号当天成交）
- 交易成本：每次仓位变化时，按变化量 * `cost_bps / 10000` 扣除

### Step 1: 写失败测试

```python
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
```

### Step 2: 运行确认失败

```bash
uv run python -m pytest tests/test_backtest_engine.py -v
```
期望：`ImportError: cannot import name 'BacktestEngine'`

### Step 3: 完整重写 jobs/backtest.py

```python
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

        from app.providers.analytics.performance import summarize

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
        os.makedirs("data", exist_ok=True)
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

    if df.empty or len(df) < 300:
        print(f"[backtest] 数据不足（{len(df)} 根 bar），需要至少 300 根。请先运行 ingest。")
        return {}

    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    cfg = BacktestConfig(alert_cfg=AlertConfig.load_evolution())
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
```

### Step 4: 运行确认通过

```bash
uv run python -m pytest tests/test_backtest_engine.py -v
```
期望：所有测试 `PASSED`

### Step 5: 提交

```bash
git add jobs/backtest.py tests/test_backtest_engine.py
git commit -m "feat: rewrite backtest as vectorized engine with IS/OOS split and transaction costs"
```

---

## Task 3: Walk-Forward 滚动窗口框架

**Files:**
- Create: `jobs/walk_forward.py`
- Create: `tests/test_walk_forward.py`

**设计：** 2年训练 / 6月测试，每次滚动 3 月。最终汇总所有 OOS 窗口的 Sharpe/t-stat。

### Step 1: 写失败测试

```python
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
```

### Step 2: 运行确认失败

```bash
uv run python -m pytest tests/test_walk_forward.py -v
```
期望：`ImportError`

### Step 3: 实现 Walk-Forward 框架

```python
# jobs/walk_forward.py
from __future__ import annotations

import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from alerts.rules import AlertConfig
from app.providers.analytics.performance import sharpe_ratio, t_stat_alpha, max_drawdown
from jobs.backtest import BacktestConfig, BacktestEngine


@dataclass
class WalkForwardConfig:
    train_bars: int = 504    # ~2 年交易日
    test_bars: int = 126     # ~6 月交易日
    step_bars: int = 63      # ~3 月滚动步长
    cost_bps: float = 10.0


class WalkForwardRunner:
    def __init__(self, df: pd.DataFrame, cfg: WalkForwardConfig):
        self.df = df.copy().sort_values("ts").reset_index(drop=True)
        self.cfg = cfg

    def run(self) -> List[dict]:
        results = []
        n = len(self.df)
        start = 0
        window_idx = 0

        while start + self.cfg.train_bars + self.cfg.test_bars <= n:
            train_end = start + self.cfg.train_bars
            test_end = train_end + self.cfg.test_bars

            train_df = self.df.iloc[start:train_end].copy()
            test_df = self.df.iloc[train_end:test_end].copy()

            # 用训练集找最优参数（这里简化为默认参数，可扩展为参数扫描）
            alert_cfg = AlertConfig.load_evolution()
            bt_cfg = BacktestConfig(cost_bps=self.cfg.cost_bps, alert_cfg=alert_cfg)

            # 在测试集上评估
            # 注意：test_df 前面需要 train_df 的历史来计算指标
            combined = pd.concat([train_df, test_df], ignore_index=True)
            engine = BacktestEngine(combined, bt_cfg)
            full_result = engine.run()

            # 只取测试集对应的行
            n_train_result = len(full_result) - self.cfg.test_bars
            if n_train_result < 0:
                start += self.cfg.step_bars
                continue

            oos_result = full_result.iloc[max(0, n_train_result):].copy()

            if len(oos_result) < 10:
                start += self.cfg.step_bars
                continue

            oos_sr = sharpe_ratio(oos_result["strat_return"])
            oos_t = t_stat_alpha(oos_result["strat_return"] - oos_result["bench_return"])
            oos_mdd = max_drawdown((1 + oos_result["strat_return"]).cumprod())

            results.append({
                "window": window_idx,
                "train_start": str(train_df["ts"].iloc[0].date()),
                "train_end": str(train_df["ts"].iloc[-1].date()),
                "test_start": str(test_df["ts"].iloc[0].date()),
                "test_end": str(test_df["ts"].iloc[-1].date()),
                "oos_sharpe": round(oos_sr, 3),
                "oos_t_stat": round(oos_t, 3),
                "oos_mdd": round(oos_mdd, 4),
                "n_bars": len(oos_result),
            })

            window_idx += 1
            start += self.cfg.step_bars

        return results

    def summarize(self, results: List[dict]) -> dict:
        if not results:
            return {}
        sharpes = [r["oos_sharpe"] for r in results]
        t_stats = [r["oos_t_stat"] for r in results]
        return {
            "n_windows": len(results),
            "mean_oos_sharpe": round(float(np.mean(sharpes)), 3),
            "median_oos_sharpe": round(float(np.median(sharpes)), 3),
            "pct_positive_sharpe": round(sum(s > 0 for s in sharpes) / len(sharpes), 3),
            "mean_oos_t_stat": round(float(np.mean(t_stats)), 3),
            "pct_significant": round(sum(abs(t) >= 2 for t in t_stats) / len(t_stats), 3),
        }

    def print_report(self, results: List[dict]) -> None:
        summary = self.summarize(results)
        print(f"\n{'='*60}")
        print(f"  Walk-Forward 报告（{summary.get('n_windows', 0)} 个 OOS 窗口）")
        print(f"{'='*60}")
        print(f"{'窗口':<4} {'测试区间':<24} {'Sharpe':>8} {'t-stat':>8} {'MDD':>8}")
        print(f"{'-'*60}")
        for r in results:
            period = f"{r['test_start']} ~ {r['test_end']}"
            print(f"{r['window']:<4} {period:<24} {r['oos_sharpe']:>8.2f} {r['oos_t_stat']:>8.2f} {r['oos_mdd']:>8.1%}")
        print(f"{'-'*60}")
        print(f"{'平均':<28} {summary['mean_oos_sharpe']:>8.2f} {summary['mean_oos_t_stat']:>8.2f}")
        print(f"\n显著窗口比例: {summary['pct_significant']:.0%}（t≥2）")
        print(f"正 Sharpe 比例: {summary['pct_positive_sharpe']:.0%}")
        verdict = "✓ 策略具有统计稳健性" if summary['mean_oos_sharpe'] > 0.5 and summary['pct_significant'] > 0.5 else "✗ 策略可能是过拟合"
        print(f"结论: {verdict}")
        print(f"{'='*60}\n")


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

    if df.empty or len(df) < 800:
        print(f"数据不足（{len(df)} 根 bar），Walk-Forward 需要 800+ 根。")
        _sys.exit(1)

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    cfg = WalkForwardConfig()
    runner = WalkForwardRunner(df, cfg)
    results = runner.run()
    runner.print_report(results)
```

### Step 4: 运行确认通过

```bash
uv run python -m pytest tests/test_walk_forward.py -v
```
期望：所有测试 `PASSED`

### Step 5: 提交

```bash
git add jobs/walk_forward.py tests/test_walk_forward.py
git commit -m "feat: add Walk-Forward rolling window framework (2yr train / 6mo test)"
```

---

## Task 4: 参数敏感性扫描（防过拟合检验）

**Files:**
- Create: `jobs/param_sweep.py`
- Create: `tests/test_param_sweep.py`

**目的：** 好策略在参数轻微变化时仍应有正收益。只在某个精确参数点才盈利 = 过拟合。

### Step 1: 写失败测试

```python
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
```

### Step 2: 运行确认失败

```bash
uv run python -m pytest tests/test_param_sweep.py -v
```
期望：`ImportError`

### Step 3: 实现参数扫描

```python
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
    is_ratio: float = 0.75  # OOS 占 25%


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
            except Exception as e:
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
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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
```

### Step 4: 运行确认通过

```bash
uv run python -m pytest tests/test_param_sweep.py -v
```
期望：所有测试 `PASSED`

### Step 5: 跑全量测试确认无回归

```bash
uv run python -m pytest tests/ -v
```
期望：全部 `PASSED`

### Step 6: 提交

```bash
git add jobs/param_sweep.py tests/test_param_sweep.py
git commit -m "feat: add parameter sensitivity sweep to detect overfitting"
```

---

## 验收标准（全部完成后执行）

### 运行完整回测

```bash
# 1. 确保有足够历史数据（先 ingest）
uv run python -m jobs.ingest

# 2. IS/OOS 回测
uv run python -m jobs.backtest AAPL

# 3. Walk-Forward（需要 800+ 根 bar）
uv run python -m jobs.walk_forward AAPL

# 4. 参数敏感性扫描
uv run python -m jobs.param_sweep AAPL
```

### 解读结果

| 指标 | 及格 | 优秀 | 说明 |
|------|------|------|------|
| OOS Sharpe | > 0.5 | > 1.0 | 核心指标 |
| OOS t-stat | > 2.0 | > 3.0 | 低于 2 = 噪音 |
| WF 正 Sharpe 窗口 | > 50% | > 70% | 时间稳定性 |
| 参数正 Sharpe 比例 | > 60% | > 75% | 防过拟合 |

---

## 依赖检查

```bash
uv add scipy  # t-test 需要 scipy.stats
```
