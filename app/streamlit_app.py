import sys
from pathlib import Path

# Add project root to sys.path for robust imports
root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import os
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import json
from pathlib import Path
from app.providers.store.sqlite_store import SQLiteStore
from jobs.backtest import run_backtest


def get_db_path() -> str:
    env_path = os.getenv("WATCHTOWER_DB_PATH")
    if env_path:
        return env_path
    return "watchtower.db"


def get_connection(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def load_tickers(conn: sqlite3.Connection) -> list[str]:
    query = "SELECT DISTINCT ticker FROM prices ORDER BY ticker"
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        return []
    return df["ticker"].tolist()


def load_price_data(
    conn: sqlite3.Connection,
    ticker: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    base_query = "SELECT ts, open, high, low, close, volume FROM prices WHERE ticker = ?"
    params: list = [ticker]

    if start is not None:
        base_query += " AND ts >= ?"
        params.append(start.isoformat())
    if end is not None:
        base_query += " AND ts <= ?"
        params.append(end.isoformat())

    base_query += " ORDER BY ts"
    df = pd.read_sql_query(base_query, conn, params=params, parse_dates=["ts"])
    return df


def main() -> None:
    st.set_page_config(page_title="Watchtower Dashboard", layout="wide")
    st.title("📈 Watchtower 市场监控")

    db_path = get_db_path()
    store = SQLiteStore(db_path=db_path)
    store.init_db()

    conn = get_connection(db_path)

    tickers = load_tickers(conn)
    if not tickers:
        st.warning("数据库中暂无数据，请先运行 `python -m jobs.ingest`。")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        ticker = st.selectbox("标的", options=tickers)

    default_end = datetime.utcnow()
    default_start = default_end - timedelta(days=7)

    with col2:
        start_date = st.date_input("开始日期", value=default_start.date())
    with col3:
        end_date = st.date_input("结束日期", value=default_end.date())

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    df = load_price_data(conn, ticker, start=start_dt, end=end_dt)

    if df.empty:
        st.info("所选时间范围内暂无数据。")
        return

    tab1, tab2 = st.tabs(["实时监控", "AI 回测分析"])

    with tab1:
        st.subheader(f"{ticker} 收盘价走势")
        st.line_chart(df.set_index("ts")["close"])

        st.subheader("K 线（简化视图）")
        st.dataframe(df.tail(100).set_index("ts"))

        st.subheader("统计信息")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("最新价", f"{df['close'].iloc[-1]:.2f}")
        with col_b:
            st.metric("最高价", f"{df['high'].max():.2f}")
        with col_c:
            st.metric("最低价", f"{df['low'].min():.2f}")
        with col_d:
            st.metric("成交量合计", f"{df['volume'].sum():.0f}")

    with tab2:
        st.subheader("仿真回测: AI 策略 vs. 基准")
        days_back = st.slider("回测天数", 30, 180, 60)
        
        if st.button("开始回测仿真"):
            with st.spinner("AI 正在模拟决策并跑历史数据..."):
                metrics = run_backtest(ticker, days_back)
                st.success(f"回测完成！标的: {ticker}")
                
                # Load backtest results for plotting
                res_p = Path("data/backtest_results.csv")
                if res_p.exists():
                    res_df = pd.read_csv(res_p)
                    res_df["ts"] = pd.to_datetime(res_df["ts"])
                    
                    # 1. Equity Curve Plot
                    st.write("### 权益曲线 (Equity Curve)")
                    st.line_chart(res_df.set_index("ts")[["equity_hold", "equity_strat"]])
                    
                    # 2. Metrics Metrics
                    st.write("### 绩效指标对比")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("最终收益 (AI)", f"{metrics['ret_strat']:+.2%}")
                    c2.metric("最终收益 (基准)", f"{metrics['ret_hold']:+.2%}")
                    c3.metric("最大回撤 (AI)", f"{metrics['mdd_strat']:.2%}", delta=f"{(metrics['mdd_strat']-metrics['mdd_hold']):.2%}")
                    c4.metric("最大回撤 (基准)", f"{metrics['mdd_hold']:.2%}")
                    
                    # 3. Risk Score vs Price
                    st.write("### 仿真期间风险分走势")
                    st.line_chart(res_df.set_index("ts")["score"])
                else:
                    st.error("未能生成回测数据。")


if __name__ == "__main__":
    main()

st.subheader("Key Alerts (jsonl)")
p = Path("data/alerts.jsonl")
if p.exists():
    lines = p.read_text(encoding="utf-8").splitlines()[-50:]
    st.code("\n".join(lines), language="json")
else:
    st.caption("No alerts yet.")