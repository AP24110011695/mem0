"""PGVector lifecycle must release owned connection pools and leave borrowed ones open."""

from unittest.mock import MagicMock

import os
import tempfile
import time

import pytest

from mem0.memory.main import Memory
from mem0.vector_stores import pgvector as pgvector_module
from mem0.vector_stores.base import VectorStoreBase
from mem0.vector_stores.pgvector import PGVector


@pytest.fixture
def pools(monkeypatch):
    """Replace the driver pool class so constructing a store touches no database."""
    created = []

    def make_pool(*args, **kwargs):
        pool = MagicMock()
        created.append(pool)
        return pool

    monkeypatch.setattr(pgvector_module, "ConnectionPool", make_pool)
    return created


def make_store(**kwargs):
    defaults = {
        "collection_name": "test_collection",
        "embedding_model_dims": 1536,
        "dbname": "test_db",
        "user": "user",
        "password": "password",
        "host": "localhost",
        "port": 5432,
        "diskann": False,
        "hnsw": False,
    }
    defaults.update(kwargs)
    return PGVector(**defaults)


def test_close_closes_owned_pool(pools, monkeypatch):
    monkeypatch.setattr(pgvector_module, "PSYCOPG_VERSION", 3)
    store = make_store()

    store.close()

    pools[0].close.assert_called_once()
    pools[0].closeall.assert_not_called()


def test_close_closes_owned_pool_psycopg2(pools, monkeypatch):
    monkeypatch.setattr(pgvector_module, "PSYCOPG_VERSION", 2)
    store = make_store()

    store.close()

    pools[0].closeall.assert_called_once()
    pools[0].close.assert_not_called()


def test_close_is_idempotent(pools, monkeypatch):
    monkeypatch.setattr(pgvector_module, "PSYCOPG_VERSION", 3)
    store = make_store()

    store.close()
    store.close()

    pools[0].close.assert_called_once()


def test_close_leaves_borrowed_pool_open(pools):
    borrowed = MagicMock()
    store = make_store(connection_pool=borrowed)

    store.close()
    store.__del__()

    borrowed.close.assert_not_called()
    borrowed.closeall.assert_not_called()


def test_del_closes_owned_pool_once(pools, monkeypatch):
    monkeypatch.setattr(pgvector_module, "PSYCOPG_VERSION", 3)
    store = make_store()

    store.__del__()
    store.__del__()

    pools[0].close.assert_called_once()


def _fake_store():
    return MagicMock(spec=VectorStoreBase)


def test_memory_close_closes_vector_stores():
    "Memory.close() closes the vector stores it holds (issue #7771 wiring)."
    memory = Memory.__new__(Memory)
    memory.db = None
    memory.vector_store = _fake_store()
    memory._telemetry_vector_store = _fake_store()
    memory._entity_store = None

    memory.close()

    memory.vector_store.close.assert_called_once()
    memory._telemetry_vector_store.close.assert_called_once()


def test_memory_close_survives_store_close_failures():
    "A failing store close does not prevent the other stores and db from closing."
    memory = Memory.__new__(Memory)
    failing = _fake_store()
    failing.close.side_effect = RuntimeError("pool already closed")
    memory.db = None
    memory.vector_store = failing
    memory._telemetry_vector_store = _fake_store()
    memory._entity_store = None

    memory.close()

    memory._telemetry_vector_store.close.assert_called_once()
    assert memory.db is None


def test_memory_close_without_stores_is_noop():
    "close() on a minimally constructed instance closes nothing and does not raise."
    memory = Memory.__new__(Memory)
    memory.db = None

    memory.close()

    assert memory.db is None


@pytest.mark.skipif(
    not all(os.environ.get(v) for v in ("PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE")),
    reason="requires a live PostgreSQL server (set PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE)",
)
def test_memory_close_returns_server_sessions_to_baseline():
    "End-to-end against a real server: N Memory objects, close() on each with\n    references kept, and pg_stat_activity returns to baseline (issue #7771).\n\n    Mirrors the reproduction script from the issue triage. Requires psycopg2."
    import gc

    import psycopg2

    def server_sessions():
        con = psycopg2.connect(
            host=os.environ["PGHOST"],
            port=os.environ.get("PGPORT", "5432"),
            user=os.environ["PGUSER"],
            password=os.environ["PGPASSWORD"],
            dbname=os.environ["PGDATABASE"],
        )
        try:
            with con.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND pid <> pg_backend_pid()"
                )
                return cur.fetchone()[0]
        finally:
            con.close()

    def make():
        from mem0 import Memory

        return Memory.from_config(
            {
                "vector_store": {
                    "provider": "pgvector",
                    "config": {
                        "host": os.environ["PGHOST"],
                        "port": int(os.environ.get("PGPORT", "5432")),
                        "user": os.environ["PGUSER"],
                        "password": os.environ["PGPASSWORD"],
                        "dbname": os.environ["PGDATABASE"],
                        "collection_name": "repro_close_pool",
                        "embedding_model_dims": 1536,
                    },
                },
                "history_db_path": os.path.join(tempfile.mkdtemp(), "history.db"),
            }
        )

    baseline = server_sessions()
    memories = [make() for _ in range(3)]
    time.sleep(1)
    after_construct = server_sessions()
    assert after_construct > baseline

    for m in memories:
        m.close()
    time.sleep(1)
    gc.collect()
    time.sleep(1)

    assert server_sessions() == baseline
