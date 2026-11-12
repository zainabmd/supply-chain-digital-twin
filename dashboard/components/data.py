"""
Dashboard data access.

* Warehouse tables are read once, read-only, and cached (`st.cache_data`, 10 min).
* Digital-twin runs are cached per (scenario parameters, policy). Moving a slider
  back to an earlier value is instant, and every page shares one run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src.scenarios import SCENARIOS, ScenarioParams
from src.twin import TwinRun, load_inputs, run_scenario
from src.warehouse import read_sql

TTL = 600


class WarehouseMissing(RuntimeError):
    pass


@st.cache_data(ttl=TTL, show_spinner="Reading the warehouse…")
def table(name: str) -> pd.DataFrame:
    try:
        df = read_sql(f"SELECT * FROM {name}")
    except Exception as exc:  # noqa: BLE001 – surface a friendly message in the UI
        raise WarehouseMissing(f"`{name}` is not available ({exc.__class__.__name__}). "
                               f"Run `python -m src.pipeline` to build the warehouse.") from exc
    for c in df.columns:
        if c in ("week_start", "date", "month", "origin_week", "target_week", "start_date", "end_date"):
            df[c] = pd.to_datetime(df[c])
    return df


@st.cache_resource(show_spinner="Loading the digital twin…")
def twin_inputs() -> dict[str, pd.DataFrame]:
    return load_inputs()


@st.cache_data(max_entries=32, show_spinner="Simulating the network (routing · flow LP · Monte Carlo)…")
def twin(params: ScenarioParams, policy: str) -> TwinRun:
    return run_scenario(params, twin_inputs(), policy=policy)


@st.cache_data(ttl=TTL)
def snapshot() -> dict:
    s = table("ml.control_tower_snapshot")
    return s.iloc[0].to_dict() if len(s) else {}


@st.cache_data(ttl=TTL)
def forecast_metrics() -> dict:
    r = table("ops.model_registry")
    r = r[r["status"] == "production"].sort_values("trained_at")
    return json.loads(r.iloc[-1]["metrics_json"]) if len(r) else {}


@dataclass(frozen=True)
class Filters:
    supplier_regions: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    dc_regions: tuple[str, ...] = ()

    def series_mask(self, df: pd.DataFrame) -> pd.Series:
        m = pd.Series(True, index=df.index)
        if self.categories and "category" in df:
            m &= df["category"].isin(self.categories)
        if self.dc_regions and "dc_region" in df:
            m &= df["dc_region"].isin(self.dc_regions)
        return m

    def node_mask(self, nodes: pd.DataFrame) -> pd.Series:
        """Supplier-region filter applies to supplier nodes only; everything else stays visible."""
        if not self.supplier_regions:
            return pd.Series(True, index=nodes.index)
        is_sup = nodes["node_type"].isin(["supplier_t2", "supplier_t3"])
        return ~is_sup | nodes["region"].isin(self.supplier_regions)


def scenario_meta(name: str) -> dict:
    return SCENARIOS.get(name, SCENARIOS["Baseline (no disruption)"])
