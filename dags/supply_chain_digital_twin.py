"""
Airflow DAG: supply-chain digital twin, daily refresh.

    extract_operational_data → load_raw → validate_raw → transform_staging → build_marts
        → update_feature_store → train_and_forecast ┬→ network_graph_analytics ───────────────┐
                                                    └→ optimize_inventory_policy → stockout_risk_outlook ┴→ publish_control_tower

The DAG is generated from `src.pipeline.TASKS`, the same registry the local
runner and CI use, so task code, dependencies and retry policy are defined
exactly once. Each task is a thin PythonOperator around a pure function.

Operational features:
- daily at 04:00 UTC, after the ERP/TMS overnight extracts land; `catchup=False`
- retries with exponential backoff per task (from the registry)
- an SLA on the forecast task, with a miss callback to the on-call channel
- `max_active_runs=1`: the DuckDB/Snowflake loads are full refreshes
- failure callback with run context for alerting
- pools: `warehouse_write` (1 slot) serialises tasks that write the warehouse
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

# Make the project importable inside the Airflow container (repo mounted at /opt/airflow/project).
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.pipeline import TASKS  # noqa: E402

log = logging.getLogger(__name__)


def _on_failure(context) -> None:
    ti = context["task_instance"]
    log.error("Task %s failed in DAG run %s (try %s). Log: %s", ti.task_id, context["run_id"], ti.try_number, ti.log_url)
    # Production: post to Slack/PagerDuty here (e.g. SlackWebhookHook) with the log URL and run id.


def _sla_miss(dag, task_list, blocking_task_list, slas, blocking_tis) -> None:
    log.warning("SLA missed for %s (blocking: %s)", task_list, blocking_task_list)


def _run(task_name: str, **_):
    """Execute one registry task. Raising marks the Airflow task failed and triggers retries."""
    result = TASKS[task_name].fn()
    log.info("%s finished: %s", task_name, str(result)[:500])
    return str(result)[:2000]   # pushed to XCom for lineage/debugging (kept small)


default_args = {
    "owner": "supply-chain-analytics",
    "depends_on_past": False,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=20),
    "on_failure_callback": _on_failure,
    "execution_timeout": timedelta(minutes=45),
}

WAREHOUSE_WRITERS = set(TASKS)  # every task writes; DuckDB allows one writer at a time

with DAG(
    dag_id="supply_chain_digital_twin",
    description="ETL → feature store → quantile forecast → graph resilience → inventory optimisation → risk outlook",
    start_date=datetime(2025, 1, 1),
    schedule="0 4 * * *",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    sla_miss_callback=_sla_miss,
    tags=["supply-chain", "digital-twin", "forecasting", "optimization"],
    doc_md=__doc__,
) as dag:
    start = EmptyOperator(task_id="start")
    done = EmptyOperator(task_id="done", trigger_rule="all_success")

    ops: dict[str, PythonOperator] = {}
    for name, spec in TASKS.items():
        ops[name] = PythonOperator(
            task_id=name,
            python_callable=_run,
            op_kwargs={"task_name": name},
            retries=spec.retries,
            pool="warehouse_write" if name in WAREHOUSE_WRITERS else "default_pool",
            sla=timedelta(hours=1) if name == "train_and_forecast" else None,
            doc_md=spec.doc,
        )

    for name, spec in TASKS.items():
        if not spec.deps:
            start >> ops[name]
        for dep in spec.deps:
            ops[dep] >> ops[name]
    terminal = [n for n in TASKS if not any(n in t.deps for t in TASKS.values())]
    for n in terminal:
        ops[n] >> done
