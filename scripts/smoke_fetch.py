import time
from pathlib import Path
import pandas as pd

CACHE_DIR = Path("data/yf_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def cache_key(ticker: str, interval: str, days: int) -> Path:
    safe = ticker.replace("/", "_")
    return CACHE_DIR / f"{safe}_{interval}_{days}d.parquet"

def fetch(ticker: str, interval: str, days: int) -> pd.DataFrame:
    # 1) cache 10 minutes
    p = cache_key(ticker, interval, days)
    if p.exists() and (time.time() - p.stat().st_mtime) < 600:
        return pd.read_parquet(p)

    # 2) reduce load
    days = min(days, 7)

    # 3) retry with backoff
    last_err = None
    for i in range(5):
        try:
            df = yf.download(
                ticker,
                period=f"{days}d",
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,   # threads=False 更“温和”
            )
            if df is None or df.empty:
                return df
            df.to_parquet(p)
            return df
        except Exception as e:
            last_err = e
            sleep = 2 ** i
            print(f"Retry {i+1}/5 after {sleep}s due to: {e}")
            time.sleep(sleep)
    raise last_err

