from __future__ import annotations

import os
import time
from pathlib import Path

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List

import pandas as pd
import yaml
import yfinance as yf
import httpx

from alerts import AlertConfig, evaluate_alerts, append_alerts_jsonl
from app.providers.store.sqlite_store import PriceRow, SQLiteStore


@dataclass
class WatchtowerConfig:
    tickers: List[str]
    interval: str
    days: int
    db_path: str


def load_config(path: str = "config.yaml") -> WatchtowerConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    tickers = data.get(
        "tickers",
        ["0700.HK", "COIN", "INFQ"],
    )
    interval = data.get("interval", "60m")
    days = int(data.get("days", 30))

    db_path = os.getenv("WATCHTOWER_DB_PATH") or data.get("db_path") or "watchtower.db"

    return WatchtowerConfig(
        tickers=tickers,
        interval=interval,
        days=days,
        db_path=db_path,
    )


def df_to_rows(ticker: str, df: pd.DataFrame) -> Iterable[PriceRow]:
    for ts, row in df.iterrows():
        # 跳过周末（yfinance 对港股偶尔会返回周六时间戳）
        if ts.weekday() >= 5:
            continue
        yield PriceRow(
            ticker=ticker,
            ts=ts.to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row.get("Volume", 0.0)),
        )

CACHE_DIR = Path("data/yf_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_path(ticker: str, interval: str, days: int) -> Path:
    safe = ticker.replace("/", "_")
    return CACHE_DIR / f"{safe}_{interval}_{days}d.parquet"

def _read_cache(p: Path, ttl_seconds: int = 600) -> pd.DataFrame | None:
    if not p.exists():
        return None
    if (time.time() - p.stat().st_mtime) > ttl_seconds:
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None

def _write_cache(p: Path, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(p)
    except Exception:
        pass


def fetch_ticker(ticker: str, interval: str, days: int) -> pd.DataFrame:
    # Ensure we have enough bars for indicators (EMA200, Z-Score 100)
    # If daily, we need ~300 days. If hourly, we need ~300 hours (~13 days).
    min_required_days = 365 if interval.lower() == "1d" else 20
    eff_days = max(days, min_required_days)

    cache_p = _cache_path(ticker, interval, eff_days)
    cached = _read_cache(cache_p, ttl_seconds=600)  # 10 min cache
    if cached is not None and not cached.empty:
        return cached

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=eff_days)

    last_err: Exception | None = None
    for i in range(5):
        try:
            df = yf.download(
                tickers=ticker,
                interval=interval,
                start=start,
                end=end,
                progress=False,
                threads=False,     # crucial: be gentle
                auto_adjust=False,
            )

            if df is None or df.empty:
                return df

            # flatten multi-index if needed (e.g. from newer yfinance versions)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

            # normalize columns
            col_map = {str(c): str(c).capitalize() for c in df.columns}
            df = df.rename(columns=col_map)

            needed = ["Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                raise RuntimeError(f"Missing columns {missing} in data for {ticker}")

            _write_cache(cache_p, df)
            return df

        except Exception as e:
            last_err = e
            sleep_s = 2 ** i
            print(f"[ingest] Retry {i+1}/5 for {ticker} after {sleep_s}s due to: {e}")
            time.sleep(sleep_s)

    # retries exhausted
    raise last_err  # type: ignore

def fetch_ticker_stooq_daily(ticker: str, days: int) -> pd.DataFrame:
    """
    Stooq free daily CSV:
      https://stooq.com/q/d/l/?s=coin.us&i=d
    Columns: Date,Open,High,Low,Close,Volume
    """

    def candidates(t: str) -> list[str]:
        t0 = t.strip()
        tl = t0.lower()

        cands = []

        # If already has suffix like coin.us / 0700.hk
        if "." in tl:
            cands.append(tl)

            # yfinance HK uses .HK; stooq often uses .hk
            if tl.endswith(".hk"):
                cands.append(tl)  # keep

        else:
            # US equities on stooq usually need .us
            cands.append(f"{tl}.us")
            cands.append(tl)

        # yfinance uses 0700.HK; map to 0700.hk
        if tl.endswith(".hk") or tl.endswith(".hk"):
            pass
        if tl.endswith(".hk") is False and t0.upper().endswith(".HK"):
            cands.insert(0, t0[:-3].lower() + ".hk")

        # also handle 0700.HK explicitly
        if t0.upper().endswith(".HK"):
            cands = [t0[:-3].lower() + ".hk"] + cands

        # de-dup keep order
        out = []
        for x in cands:
            if x not in out:
                out.append(x)
        return out

    import pandas as pd
    from io import StringIO

    last_err = None
    for sym in candidates(ticker):
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        try:
            with httpx.Client(timeout=20, headers={"User-Agent": "watchtower/0.1"}) as client:
                r = client.get(url)
                r.raise_for_status()
                text = r.text.strip()

            if not text:
                continue

            head = text.splitlines()[0].strip()
            if head != "Date,Open,High,Low,Close,Volume":
                # stooq returns a tiny message/html when no data
                continue

            df = pd.read_csv(StringIO(text))
            if df.empty:
                continue

            df["Date"] = pd.to_datetime(df["Date"], utc=True)
            df = df.set_index("Date").sort_index()

            # keep roughly last N rows
            if days and len(df) > days:
                df = df.tail(days)

            # normalize casing
            col_map = {c: c.capitalize() for c in df.columns}
            df = df.rename(columns=col_map)
            return df

        except Exception as e:
            last_err = e
            continue

    # optional: print last error for debugging
    if last_err:
        print(f"[ingest] stooq failed for {ticker}: {last_err}")
    return pd.DataFrame()


def fetch_fundamental_valuation(ticker_str: str) -> dict:
    """
    Fetches core valuation metrics (P/E, P/B, Dividend Yield).
    """
    try:
        t = yf.Ticker(ticker_str)
        info = t.info
        return {
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "priceToBook": info.get("priceToBook"),
            "dividendYield": info.get("dividendYield", 0.0),
        }
    except Exception as e:
        print(f"[ingest] Failed to fetch valuation for {ticker_str}: {e}")
        return {}


def run_ingest() -> None:

    cfg = load_config()
    store = SQLiteStore(db_path=cfg.db_path)
    store.init_db()

    # Step 1: Pre-fetch Global VIX for Market 'Weather' Context
    global_vix = 20.0
    try:
        print("[ingest] Fetching ^VIX for market context...")
        vix_df = fetch_ticker("^VIX", "1d", 5)
        if not vix_df.empty:
            global_vix = float(vix_df.iloc[-1]["Close"])
            print(f"[ingest] Current Market VIX: {global_vix:.2f}")
    except Exception as e:
        print(f"[ingest] Warning: Failed to fetch VIX ({e}), using default 20.0")

    total_rows = 0
    process_tickers = [t for t in cfg.tickers if t != "^VIX"]
    
    for ticker in process_tickers:
        print(f"[ingest] Processing {ticker} (Daily/Hourly + Fundamentals)")

        # Fetch Valuation Context
        valuation = fetch_fundamental_valuation(ticker)
        
        df = fetch_ticker(ticker, cfg.interval, cfg.days)

        if df.empty and cfg.interval.lower() == "1d":
            print(f"[ingest] yfinance empty for {ticker}, trying stooq daily fallback ...")
            df = fetch_ticker_stooq_daily(ticker, 365) # request more for fallback

        if df.empty:
            print(f"[ingest] No data for {ticker}")
            continue

        rows = list(df_to_rows(ticker, df))
        inserted = store.upsert_prices(rows)
        total_rows += inserted
        print(f"[ingest] Stored {inserted} rows for {ticker}")
        # build a small dataframe for alerts using the same df we just fetched
        tmp = df.reset_index()
        # yfinance index might be "Date" or "Datetime"; stooq is "Date"
        ts_col = "Date" if "Date" in tmp.columns else ("Datetime" if "Datetime" in tmp.columns else tmp.columns[0])
        df_alert = pd.DataFrame({
            "ts": pd.to_datetime(tmp[ts_col], utc=True),
            "open": tmp["Open"].astype(float),
            "high": tmp["High"].astype(float),
            "low": tmp["Low"].astype(float),
            "close": tmp["Close"].astype(float),
            "volume": tmp.get("Volume", 0.0).astype(float),
            }).sort_values("ts")

        cfg_alert = AlertConfig.load_evolution()
        alerts = evaluate_alerts(ticker, df_alert, cfg_alert, global_vix=global_vix, valuation=valuation)
        append_alerts_jsonl(alerts)
        if alerts:
            from notify.telegram import send_alert
            for a in alerts:
                print(f"[ALERT] {a['severity']} {a['symbol']} {a['rule_id']} {a['ts']} - {a['msg']}")
                send_alert(a)
        
        time.sleep(2)

    print(f"[ingest] Done. Total inserted rows: {total_rows}")


if __name__ == "__main__":
    run_ingest()

