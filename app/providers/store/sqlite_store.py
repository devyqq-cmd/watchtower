import sqlite3
from dataclasses import dataclass
from typing import Iterable


@dataclass
class PriceRow:
    ticker: str
    ts: str  # ISO datetime string
    open: float
    high: float
    low: float
    close: float
    volume: float


class SQLiteStore:
    def __init__(self, db_path: str = "watchtower.db") -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def init_db(self) -> None:
        create_sql = """
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            ts TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            UNIQUE (ticker, ts)
        );
        """
        with self._connect() as conn:
            conn.execute(create_sql)
            conn.commit()

    def upsert_prices(self, rows: Iterable[PriceRow]) -> int:
        sql = """
        INSERT OR IGNORE INTO prices (ticker, ts, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            cur = conn.executemany(
                sql,
                [
                    (
                        r.ticker,
                        r.ts,
                        r.open,
                        r.high,
                        r.low,
                        r.close,
                        r.volume,
                    )
                    for r in rows
                ],
            )
            conn.commit()
            return cur.rowcount

