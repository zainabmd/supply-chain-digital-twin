"""
DuckDB warehouse: raw load, SQL model execution and connection helpers.

Layering (same convention as the Snowflake deployment described in docs/ARCHITECTURE.md):

    raw.*       landing files loaded as-is (append-only, typed by Parquet)
    staging.*   cleaned, conformed views (1:1 with sources)
    marts.*     business-level tables consumed by analytics & the dashboard
    ml.*        feature store, forecasts, graph metrics, risk, optimisation outputs
    ops.*       pipeline runs and data-quality results

SQL models live in `sql/<layer>/NN_name.sql` and run in lexical order, so
dependencies are explicit in the file names. They use ANSI SQL (date_trunc,
window functions, CTEs, QUALIFY) that runs on both DuckDB and Snowflake.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd

from src.config import LANDING_DIR, SQL_DIR, WAREHOUSE_PATH, get_logger

log = get_logger("warehouse")
SCHEMAS = ("raw", "staging", "marts", "ml", "ops")


@contextmanager
def connect(read_only: bool = False, path: Path = WAREHOUSE_PATH, retries: int = 10) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open the warehouse. Readers retry briefly if a pipeline task holds the write lock."""
    last = None
    for i in range(retries):
        try:
            con = duckdb.connect(str(path), read_only=read_only)
            break
        except duckdb.IOException as exc:  # lock held by another process
            last = exc
            time.sleep(0.3 * (i + 1))
    else:
        raise last  # type: ignore[misc]
    try:
        if not read_only:
            for s in SCHEMAS:
                con.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")
        yield con
    finally:
        con.close()


def load_raw(landing: Path = LANDING_DIR) -> dict[str, int]:
    """Load every landing Parquet file into raw.<name> (full refresh, idempotent)."""
    files = sorted(p for p in landing.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No landing files in {landing} – run the extract task first")
    counts = {}
    with connect() as con:
        for f in files:
            con.execute(f"CREATE OR REPLACE TABLE raw.{f.stem} AS SELECT * FROM read_parquet(?)", [str(f)])
            counts[f.stem] = con.execute(f"SELECT count(*) FROM raw.{f.stem}").fetchall()[0][0]
    log.info("Loaded %d raw tables (%s rows)", len(counts), f"{sum(counts.values()):,}")
    return counts


def run_sql_layer(layer: str) -> list[str]:
    """Execute every model in sql/<layer>/ in file-name order."""
    files = sorted((SQL_DIR / layer).glob("*.sql"))
    built = []
    with connect() as con:
        for f in files:
            t = time.perf_counter()
            con.execute(f.read_text(encoding="utf-8"))
            built.append(f.stem)
            log.info("  %-8s %-36s %.2fs", layer, f.stem, time.perf_counter() - t)
    return built


def write_df(df: pd.DataFrame, table: str) -> int:
    """Replace a table with a DataFrame (used by Python tasks that write to ml.*)."""
    with connect() as con:
        con.register("_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _df")
        con.unregister("_df")
    return len(df)


def read_sql(sql: str, params: list | None = None) -> pd.DataFrame:
    with connect(read_only=True) as con:
        return con.execute(sql, params or []).df()
