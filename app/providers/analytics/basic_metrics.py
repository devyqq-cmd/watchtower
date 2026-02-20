from __future__ import annotations

import pandas as pd


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    if "close" not in df.columns:
        return df
    df = df.copy()
    df["return"] = df["close"].pct_change()
    return df

