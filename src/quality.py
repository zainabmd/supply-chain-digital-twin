"""
Data-quality gate (runs between load and transform in the DAG).

Each check is a SQL query returning the number of offending rows, plus a
severity. `error` failures stop the pipeline (bad data never reaches the
models); `warn` failures are recorded and surfaced on the dashboard. Results
are appended to ops.dq_results for trend monitoring.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import HISTORY_END, get_logger
from src.warehouse import connect

log = get_logger("quality")


class DataQualityError(RuntimeError):
    pass


@dataclass(frozen=True)
class Check:
    name: str
    sql: str            # returns a single integer: number of violating rows
    severity: str = "error"
    description: str = ""


CHECKS = [
    Check("demand_grain_unique",
          "SELECT count(*) FROM (SELECT date, sku_id, dc_id FROM raw.demand_daily GROUP BY ALL HAVING count(*) > 1)",
          description="One demand row per date x SKU x DC"),
    Check("demand_grid_complete",
          "SELECT (SELECT count(DISTINCT date) FROM raw.demand_daily) * (SELECT count(*) FROM raw.skus) * "
          "(SELECT count(*) FROM raw.nodes WHERE node_type = 'dc') - (SELECT count(*) FROM raw.demand_daily)",
          description="No missing date x SKU x DC combinations"),
    Check("demand_non_negative", "SELECT count(*) FROM raw.demand_daily WHERE demand_units < 0"),
    Check("inventory_non_negative",
          "SELECT count(*) FROM raw.inventory_daily WHERE on_hand_units < 0 OR on_order_units < 0 OR sales_units < 0"),
    Check("sales_not_above_demand",
          "SELECT count(*) FROM raw.inventory_daily i JOIN raw.demand_daily d USING (date, sku_id, dc_id) "
          "WHERE i.sales_units > d.demand_units + 1e-6",
          description="Cannot sell more than was demanded"),
    Check("shipment_lead_time_positive", "SELECT count(*) FROM raw.shipments WHERE actual_lead_time_days <= 0"),
    Check("shipment_fk_plant",
          "SELECT count(*) FROM raw.shipments s LEFT JOIN raw.nodes n ON n.node_id = s.plant_id WHERE n.node_id IS NULL"),
    Check("shipment_fk_sku",
          "SELECT count(*) FROM raw.shipments s LEFT JOIN raw.skus k USING (sku_id) WHERE k.sku_id IS NULL"),
    Check("edge_fk_nodes",
          "SELECT count(*) FROM raw.edges e LEFT JOIN raw.nodes a ON a.node_id = e.src LEFT JOIN raw.nodes b "
          "ON b.node_id = e.dst WHERE a.node_id IS NULL OR b.node_id IS NULL"),
    Check("every_dc_sourced",
          "SELECT count(*) FROM (SELECT d.node_id FROM raw.nodes d WHERE d.node_type = 'dc' "
          "AND d.node_id NOT IN (SELECT dc_id FROM raw.sourcing))"),
    Check("bom_components_have_supplier",
          "SELECT count(*) FROM (SELECT DISTINCT component FROM raw.bom WHERE component NOT IN "
          "(SELECT component FROM raw.component_sources))"),
    Check("freshness_history_end",
          f"SELECT CASE WHEN max(CAST(date AS DATE)) < DATE '{HISTORY_END}' THEN 1 ELSE 0 END FROM raw.demand_daily",
          description="Extract must cover the full history window"),
    Check("lead_time_outliers",
          "SELECT count(*) FROM raw.shipments WHERE actual_lead_time_days > 3 * planned_lead_time_days",
          severity="warn", description="Shipments > 3x planned lead time (investigate, but don't block)"),
    Check("single_source_components",
          "SELECT count(*) FROM (SELECT component FROM raw.component_sources GROUP BY ALL HAVING count(*) = 1)",
          severity="warn", description="Structural risk: components with a single qualified supplier"),
]


def run_checks(fail_on_error: bool = True) -> pd.DataFrame:
    rows = []
    with connect() as con:
        for c in CHECKS:
            try:
                n = int(con.execute(c.sql).fetchall()[0][0])  # fetchall: DuckDB 2 requires fully consumed results
                status = "pass" if n == 0 else ("fail" if c.severity == "error" else "warn")
            except Exception as exc:  # noqa: BLE001 – a broken check is itself a failure
                n, status = -1, "fail"
                log.error("Check %s errored: %s", c.name, exc)
            rows.append({"check_name": c.name, "severity": c.severity, "violations": n, "status": status,
                         "description": c.description, "checked_at": pd.Timestamp.now(tz="UTC")})
        res = pd.DataFrame(rows)
        con.register("_res", res)
        con.execute("CREATE TABLE IF NOT EXISTS ops.dq_results AS SELECT * FROM _res WHERE false")
        con.execute("INSERT INTO ops.dq_results SELECT * FROM _res")
    for r in res.itertuples():
        (log.info if r.status == "pass" else log.warning)("  DQ %-30s %-4s violations=%s", r.check_name, r.status.upper(), r.violations)
    failed = res[res["status"] == "fail"]
    if fail_on_error and len(failed):
        raise DataQualityError(f"{len(failed)} blocking data-quality checks failed: {failed['check_name'].tolist()}")
    return res
