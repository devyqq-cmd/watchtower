import os
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import json
from pathlib import Path
from app.providers.store.sqlite_store import SQLiteStore


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


if __name__ == "__main__":
    main()

st.subheader("Key Alerts (jsonl)")
p = Path("data/alerts.jsonl")
if p.exists():
    lines = p.read_text(encoding="utf-8").splitlines()[-50:]
    st.code("\n".join(lines), language="json")
else:
    st.caption("No alerts yet.")