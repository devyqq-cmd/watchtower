import os

from app.streamlit_app import get_db_path
from app.providers.store.sqlite_store import SQLiteStore


def test_db_init_tmp(tmp_path):
    db_path = tmp_path / "test_watchtower.db"
    store = SQLiteStore(db_path=str(db_path))
    store.init_db()
    assert db_path.exists()


def test_get_db_path_env_override(monkeypatch):
    monkeypatch.setenv("WATCHTOWER_DB_PATH", "env.db")
    assert get_db_path().endswith("env.db")

