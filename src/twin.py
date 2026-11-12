"""
Digital-twin façade: one call from "scenario + policy" to network effects and stockout risk.

Used both by the pipeline (to publish the baseline outlook) and live by the
dashboard's scenario sliders, so the two can never disagree.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import RISK_HORIZON_DAYS, get_logger
from src.risk import RiskResult, simulate
from src.scenarios import ScenarioParams, ScenarioResult, evaluate
from src.warehouse import read_sql, write_df

log = get_logger("twin")


def load_inputs() -> dict[str, pd.DataFrame]:
    q = {
        "nodes": "SELECT * FROM raw.nodes", "edges": "SELECT * FROM raw.edges", "bom": "SELECT * FROM raw.bom",
        "comp_src": "SELECT * FROM raw.component_sources", "mat_src": "SELECT * FROM raw.material_sources",
        "sku_plants": "SELECT * FROM raw.sku_plants", "sourcing": "SELECT * FROM raw.sourcing",
        "health": "SELECT * FROM marts.mart_inventory_health",
        "fc": "SELECT sku_id, dc_id, avg(p50) AS weekly_demand FROM ml.forecast_weekly GROUP BY ALL",
        "forecast": "SELECT * FROM ml.forecast_weekly", "pipeline": "SELECT * FROM raw.pipeline_open",
        "policy": "SELECT sku_id, dc_id, current_z, optimal_z FROM ml.inventory_policy",
    }
    return {k: read_sql(v) for k, v in q.items()}


@dataclass
class TwinRun:
    scenario: ScenarioResult
    risk: RiskResult
    series: pd.DataFrame       # scenario series joined with risk summary
    seconds: float


def run_scenario(params: ScenarioParams, d: dict[str, pd.DataFrame], policy: str = "current",
                 sims: int | None = None) -> TwinRun:
    t0 = time.perf_counter()
    scn = evaluate(params, d)
    ser = scn.series.merge(d["health"][["sku_id", "dc_id", "on_hand_units", "category", "dc_region"]], on=["sku_id", "dc_id"])
    z = ser[["sku_id", "dc_id"]].merge(d["policy"], on=["sku_id", "dc_id"], how="left")[f"{policy}_z"].fillna(1.65).to_numpy()
    kwargs = {"sims": sims} if sims else {}
    rr = simulate(ser, d["forecast"], d["pipeline"], z=z, duration_days=min(7 * params.duration_weeks, RISK_HORIZON_DAYS),
                  demand_shock=params.demand_shock, lead_time_multiplier=params.lead_time_multiplier, **kwargs)
    merged = ser.merge(rr.summary, on=["sku_id", "dc_id"])
    return TwinRun(scn, rr, merged, time.perf_counter() - t0)


def publish_baseline() -> pd.DataFrame:
    """Pipeline task: baseline 8-week stockout outlook under current and optimised policies."""
    d = load_inputs()
    frames = []
    for policy in ("current", "optimal"):
        r = run_scenario(ScenarioParams(), d, policy=policy)
        s = r.series[["sku_id", "dc_id", "category", "dc_region", "plant_id", "p_stockout", "p_any_stockout",
                      "expected_fill_rate", "expected_lost_units", "expected_lost_revenue", "expected_demand_units",
                      "median_days_to_stockout", "projected_end_on_hand_p50", "risk_band", "lt_base"]].copy()
        s["policy"] = policy
        frames.append(s)
        log.info("Baseline outlook (%s policy): fill %.2f%%, %d SKU-DCs high/critical, expected lost revenue $%.1fM",
                 policy, 100 * (1 - s["expected_lost_units"].sum() / s["expected_demand_units"].sum()),
                 int(s["risk_band"].isin(["High", "Critical"]).sum()), s["expected_lost_revenue"].sum() / 1e6)
    out = pd.concat(frames, ignore_index=True)
    write_df(out, "ml.stockout_risk_baseline")
    return out


def publish_snapshot() -> dict:
    """Pipeline task: headline KPIs for the control-tower header (one row, JSON-friendly)."""
    kpi = read_sql("""
        WITH last13 AS (SELECT * FROM marts.mart_service_kpis_weekly
                        WHERE week_start > (SELECT max(week_start) FROM marts.mart_service_kpis_weekly) - INTERVAL 91 DAY)
        SELECT sum(sales_units) / sum(demand_units) AS fill_rate_13w, sum(otif * shipments) / sum(shipments) AS otif_13w,
               sum(lost_revenue) AS lost_revenue_13w, sum(revenue) AS revenue_13w,
               avg(days_of_supply) AS days_of_supply FROM last13""").iloc[0].to_dict()
    inv = read_sql("SELECT sum(inventory_value) AS inv_value FROM marts.mart_inventory_health").iloc[0, 0]
    risk = read_sql("SELECT count(*) FILTER (WHERE risk_band IN ('High','Critical')) AS high, sum(expected_lost_revenue) AS rev "
                    "FROM ml.stockout_risk_baseline WHERE policy = 'current'").iloc[0]
    spof = read_sql("SELECT count(*) FILTER (WHERE is_single_point_of_failure) AS n FROM ml.graph_node_metrics").iloc[0, 0]
    fc = read_sql("SELECT metrics_json FROM ops.model_registry WHERE status = 'production' ORDER BY trained_at DESC LIMIT 1")
    import json
    wape = json.loads(fc.iloc[0, 0])["wape"] if len(fc) else np.nan
    snap = {**{k: float(v) for k, v in kpi.items()}, "inventory_value": float(inv), "high_risk_series": int(risk["high"]),
            "expected_lost_revenue_8w": float(risk["rev"]), "single_points_of_failure": int(spof), "forecast_wape": float(wape),
            "published_at": pd.Timestamp.now(tz="UTC")}
    write_df(pd.DataFrame([snap]), "ml.control_tower_snapshot")
    log.info("Control-tower snapshot published: %s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in snap.items()})
    return snap
