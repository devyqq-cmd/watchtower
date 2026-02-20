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
# =========================
# Alerts UI (watchtower)
# =========================
import json
from pathlib import Path

def _load_alerts_df():
    """
    Load alerts from jsonl files.
    Prefer data/alerts.jsonl; if missing, fall back to data/alerts.jsonl.bak
    """
    paths = [Path("data/alerts.jsonl"), Path("data/alerts.jsonl.bak")]
    p = next((x for x in paths if x.exists()), None)
    if not p:
        return None, None

    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

    if not rows:
        return p, None

    df = pd.DataFrame(rows)

    # normalize columns
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "context" in df.columns:
        # expand some context fields if present
        def ctx_get(x, k):
            try:
                return (x or {}).get(k, None)
            except Exception:
                return None
        df["ret"] = df["context"].apply(lambda x: ctx_get(x, "ret"))
        df["vol_pct"] = df["context"].apply(lambda x: ctx_get(x, "vol_pct"))
        df["vol_z"] = df["context"].apply(lambda x: ctx_get(x, "vol_z"))
        df["rv_last"] = df["context"].apply(lambda x: ctx_get(x, "rv_last"))

    # prettify
    if "ret" in df.columns:
        df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    for c in ["vol_pct", "vol_z", "rv_last"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return p, df


st.divider()
st.subheader("Key Alerts")

p, adf = _load_alerts_df()
if p is None:
    st.caption("No alerts file yet. (data/alerts.jsonl)")
elif adf is None or adf.empty:
    st.caption(f"No alerts in {p}.")
else:
    # filters
    sev_order = ["high", "med", "low"]
    sevs = [s for s in sev_order if s in set(adf.get("severity", []))] or sorted(adf.get("severity", []).dropna().unique())
    symbols = sorted(adf.get("symbol", pd.Series(dtype=str)).dropna().unique())

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        sev_sel = st.multiselect("Severity", options=sevs, default=sevs)
    with c2:
        sym_sel = st.multiselect("Symbol", options=symbols, default=symbols[: min(5, len(symbols))] if symbols else [])
    with c3:
        n = st.slider("Show latest N", min_value=10, max_value=500, value=100, step=10)

    df = adf.copy()
    if sev_sel and "severity" in df.columns:
        df = df[df["severity"].isin(sev_sel)]
    if sym_sel and "symbol" in df.columns:
        df = df[df["symbol"].isin(sym_sel)]

    if "ts" in df.columns:
        df = df.sort_values("ts", ascending=False)

    df = df.head(n)

    show_cols = [c for c in ["ts", "symbol", "severity", "rule_id", "msg", "ret", "vol_pct", "vol_z"] if c in df.columns]
    df_show = df[show_cols].copy()

    # formatting
    if "ret" in df_show.columns:
        df_show["ret"] = df_show["ret"].map(lambda x: None if pd.isna(x) else f"{x:+.2%}")
    if "vol_pct" in df_show.columns:
        df_show["vol_pct"] = df_show["vol_pct"].map(lambda x: None if pd.isna(x) else f"{x:.2f}")
    if "vol_z" in df_show.columns:
        df_show["vol_z"] = df_show["vol_z"].map(lambda x: None if pd.isna(x) else f"{x:.2f}")

    def _hl(row):
        sev = str(row.get("severity", "")).lower()
        if sev == "high":
            return ["font-weight:700"] * len(row)
        if sev == "med":
            return ["font-weight:600"] * len(row)
        return [""] * len(row)

    st.caption(f"Source: {p}")
    st.dataframe(df_show.style.apply(_hl, axis=1), use_container_width=True)
