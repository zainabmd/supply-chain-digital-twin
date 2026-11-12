"""
Task graph shared by the Airflow DAG and the local runner.

`TASKS` is the single source of truth for what runs, in what order, with
which retries. The Airflow DAG (dags/supply_chain_digital_twin.py) builds one
operator per entry; `python -m src.pipeline` executes the same graph locally
in topological order. That keeps development, CI and production identical.

Every task execution is logged to ops.task_runs (status, duration, attempt),
which the dashboard surfaces on the Pipeline page.

    python -m src.pipeline                     # full run
    python -m src.pipeline --from build_marts  # resume from a task (and everything downstream)
    python -m src.pipeline --only train_and_forecast
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from typing import Callable

import pandas as pd

from src.config import get_logger

log = get_logger("pipeline")


def _t_extract():
    from src.generate import run
    return run()


def _t_load():
    from src.warehouse import load_raw
    return load_raw()


def _t_validate():
    from src.quality import run_checks
    res = run_checks(fail_on_error=True)
    return {"passed": int((res["status"] == "pass").sum()), "warn": int((res["status"] == "warn").sum())}


def _t_staging():
    from src.warehouse import run_sql_layer
    return run_sql_layer("staging")


def _t_marts():
    from src.warehouse import run_sql_layer
    return run_sql_layer("marts")


def _t_features():
    from src.features import build_feature_store
    return len(build_feature_store())


def _t_forecast():
    from src.forecast import run
    return run()


def _t_graph():
    from src.graph_analytics import run
    return len(run())


def _t_optimize():
    from src.optimize import run
    return run()


def _t_risk():
    from src.twin import publish_baseline
    return len(publish_baseline())


def _t_publish():
    from src.twin import publish_snapshot
    return publish_snapshot()


@dataclass(frozen=True)
class Task:
    fn: Callable
    deps: tuple[str, ...] = ()
    retries: int = 1
    doc: str = ""


TASKS: dict[str, Task] = {
    "extract_operational_data": Task(_t_extract, (), 2, "Pull demand, shipments, inventory and events (ERP/TMS/WMS extracts)."),
    "load_raw": Task(_t_load, ("extract_operational_data",), 2, "Load landing Parquet into warehouse raw schema."),
    "validate_raw": Task(_t_validate, ("load_raw",), 0, "Data-quality gate; blocking checks stop the run."),
    "transform_staging": Task(_t_staging, ("validate_raw",), 1, "Build conformed staging views."),
    "build_marts": Task(_t_marts, ("transform_staging",), 1, "Build business marts (service KPIs, lanes, suppliers, health)."),
    "update_feature_store": Task(_t_features, ("build_marts",), 1, "Point-in-time weekly features + known covariates."),
    "train_and_forecast": Task(_t_forecast, ("update_feature_store",), 1, "Backtest, retrain, publish 13-week quantile forecast."),
    "network_graph_analytics": Task(_t_graph, ("train_and_forecast",), 1, "SPOFs, max-flow, min-cut, TTR/TTS exposure."),
    "optimize_inventory_policy": Task(_t_optimize, ("train_and_forecast",), 1, "SLSQP safety-stock optimisation + frontier."),
    "stockout_risk_outlook": Task(_t_risk, ("optimize_inventory_policy",), 1, "Monte Carlo 8-week stockout outlook (both policies)."),
    "publish_control_tower": Task(_t_publish, ("network_graph_analytics", "stockout_risk_outlook"), 1, "Headline KPI snapshot."),
}


def downstream(start: str) -> set[str]:
    out, frontier = {start}, [start]
    while frontier:
        cur = frontier.pop()
        for name, t in TASKS.items():
            if cur in t.deps and name not in out:
                out.add(name)
                frontier.append(name)
    return out


def _record(rows: list[dict]) -> None:
    try:
        from src.warehouse import connect
        df = pd.DataFrame(rows)
        with connect() as con:
            con.register("_r", df)
            con.execute("CREATE TABLE IF NOT EXISTS ops.task_runs AS SELECT * FROM _r WHERE false")
            con.execute("INSERT INTO ops.task_runs SELECT * FROM _r")
    except Exception as exc:  # noqa: BLE001 – logging run metadata must never fail the pipeline
        log.warning("Could not record task runs: %s", exc)


def run(only: str | None = None, from_task: str | None = None) -> bool:
    order = list(TopologicalSorter({k: set(v.deps) for k, v in TASKS.items()}).static_order())
    selected = {only} if only else (downstream(from_task) if from_task else set(order))
    run_id = uuid.uuid4().hex[:10]
    failed: set[str] = set()
    rows = []
    t_run = time.perf_counter()
    for name in order:
        if name not in selected:
            continue
        task = TASKS[name]
        if any(d in failed for d in task.deps):
            log.error("⏭  %s skipped (upstream failed)", name)
            failed.add(name)
            rows.append({"run_id": run_id, "task": name, "status": "upstream_failed", "attempt": 0, "seconds": 0.0,
                         "error": "", "finished_at": pd.Timestamp.now(tz="UTC")})
            continue
        for attempt in range(1, task.retries + 2):
            t0 = time.perf_counter()
            log.info("▶ %-28s attempt %d · %s", name, attempt, task.doc)
            try:
                task.fn()
                secs = time.perf_counter() - t0
                log.info("✔ %-28s %.1fs", name, secs)
                rows.append({"run_id": run_id, "task": name, "status": "success", "attempt": attempt, "seconds": secs,
                             "error": "", "finished_at": pd.Timestamp.now(tz="UTC")})
                break
            except Exception as exc:  # noqa: BLE001
                secs = time.perf_counter() - t0
                log.error("✖ %s failed (attempt %d): %s", name, attempt, exc)
                last = attempt == task.retries + 1
                rows.append({"run_id": run_id, "task": name, "status": "failed" if last else "retry", "attempt": attempt,
                             "seconds": secs, "error": f"{type(exc).__name__}: {exc}"[:500],
                             "finished_at": pd.Timestamp.now(tz="UTC")})
                if last:
                    log.debug(traceback.format_exc())
                    failed.add(name)
                else:
                    time.sleep(min(30, 2 ** attempt))
    _record(rows)
    ok = not failed
    log.info("Pipeline %s %s in %.1fs", run_id, "SUCCEEDED" if ok else f"FAILED ({sorted(failed)})", time.perf_counter() - t_run)
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the supply-chain digital twin pipeline locally")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--only", choices=list(TASKS))
    g.add_argument("--from", dest="from_task", choices=list(TASKS))
    args = ap.parse_args(argv)
    return 0 if run(args.only, args.from_task) else 1


if __name__ == "__main__":
    sys.exit(main())
