"""Scenario Simulator – what-if analysis: reroutes, flows, cost, service and revenue at risk vs baseline."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import WarehouseMissing, table, twin
from dashboard.components.layout import controls, page_hero, scenario_banner
from dashboard.components.maps import network_map
from dashboard.components.theme import (DIVERGING, NAVY, STATUS, TEAL, kpi, kpis, money, pct, section, style)
from src.scenarios import SCENARIOS, ScenarioParams

c = controls()
page_hero("Digital twin · what-if lab", "Scenario Simulator",
          "Use the sidebar dials. Every change re-routes the graph, re-solves the flow LP and re-runs the Monte Carlo.")
try:
    nodes, edges = table("raw.nodes"), table("raw.edges")
except WarehouseMissing as exc:
    st.error(str(exc))
    st.stop()

run = twin(c.params, c.policy)
base = twin(ScenarioParams(), c.policy)
scenario_banner(run.scenario.kpis)
m = c.filters.series_mask
s, b = run.series[m(run.series)], base.series[m(base.series)]


def fill(df):
    return 1 - df["expected_lost_units"].sum() / max(df["expected_demand_units"].sum(), 1)


k, kb = run.scenario.kpis, base.scenario.kpis
lost_delta = s["expected_lost_revenue"].sum() - b["expected_lost_revenue"].sum()
kpis([
    kpi("Revenue at risk Δ", money(lost_delta), f"{money(s['expected_lost_revenue'].sum())} total · 8 wks", "bad" if lost_delta > 0 else "good"),
    kpi("Projected fill rate", pct(fill(s)), f"baseline {pct(fill(b))}", "warn" if fill(s) < fill(b) - 0.005 else "good",
        f"{100 * (fill(s) - fill(b)):+.1f} pts", fill(s) >= fill(b) - 1e-9),
    kpi("Avg replenishment lead time", f"{k['avg_lead_time_scenario']:.1f} d", f"baseline {kb['avg_lead_time_base']:.1f} d", "serious",
        f"{k['avg_lead_time_scenario'] - kb['avg_lead_time_base']:+.1f} d", k['avg_lead_time_scenario'] <= kb['avg_lead_time_base'] + 0.05),
    kpi("Demand served (flow plan)", pct(k["network_served_share"]), "weekly capacity while disrupted", "good" if k["network_served_share"] > 0.99 else "bad",
        f"{100 * (k['network_served_share'] - kb['network_served_share']):+.1f} pts", k["network_served_share"] >= kb["network_served_share"] - 1e-9),
    # A lower unit cost is only "good" if the network still serves demand; otherwise it just means
    # the expensive long-haul volume can't ship at all.
    kpi("Cost to serve", f"${k['cost_per_unit']:,.2f}/u", f"baseline ${kb['cost_per_unit']:,.2f}/u"
        + ("" if k["network_served_share"] >= 0.95 else " · on shipped volume only"), "navy",
        f"{k['cost_per_unit'] / kb['cost_per_unit'] - 1:+.1%}",
        (k["cost_per_unit"] <= kb["cost_per_unit"]) if k["network_served_share"] >= 0.95 else None),
    kpi("Lanes rerouted / cut", f"{k['lanes_rerouted']} / {k['lanes_unreachable']}", "plant → DC lanes", "warn" if k["lanes_rerouted"] else "good"),
    kpi("SKUs supply-impaired", f"{k['skus_supply_impaired']}", "via BOM / supplier capacity", "bad" if k["skus_supply_impaired"] else "good"),
])

# ---- Baseline vs scenario maps -------------------------------------------------------
lanes = run.scenario.lanes
rerouted = lanes[(lanes["base_path"] != lanes["scn_path"]) & lanes["scn_reachable"].fillna(False)]
hl = set()
for p in rerouted["scn_path"].dropna():
    hops = p.split("→")
    hl |= set(zip(hops, hops[1:]))
tab_scn, tab_base = st.tabs(["🌪️ Scenario flow plan", "🟢 Baseline flow plan"])
with tab_scn:
    st.plotly_chart(network_map(nodes, edges, run.scenario.flows, closed=run.scenario.closed_nodes, highlight_edges=hl,
                                height=500, title="Scenario · orange = rerouted lanes · red rings = disrupted nodes",
                                flow_color="#256abf"), key="sl_map_s")
with tab_base:
    st.plotly_chart(network_map(nodes, edges, base.scenario.flows, closed=base.scenario.closed_nodes, height=500,
                                title="Baseline (as-is) weekly flow plan · red ring = ongoing disruption"), key="sl_map_b")

# ---- DC impact (diverging) + utilisation -------------------------------------------------
a, b2 = st.columns([1.2, 1])
with a:
    dc_s = s.groupby("dc_id").agg(lost=("expected_lost_revenue", "sum"), risk=("p_stockout", "mean"))
    dc_b = b.groupby("dc_id").agg(lost=("expected_lost_revenue", "sum"), risk=("p_stockout", "mean"))
    d = (dc_s - dc_b).reset_index().sort_values("lost")
    d["label"] = d["dc_id"].str.replace("DC_", "").str.replace("_", " ").str.title()
    lim = max(abs(d["lost"]).max(), 1)
    fig = go.Figure(go.Bar(x=d["lost"], y=d["label"], orientation="h",
                           marker=dict(color=d["lost"], colorscale=DIVERGING, cmin=-lim, cmax=lim),
                           customdata=d["risk"], hovertemplate="%{y}<br>Δ lost revenue %{x:$,.0f}<br>Δ mean P(stockout) %{customdata:+.1%}<extra></extra>"))
    fig.update_xaxes(tickprefix="$", tickformat="~s", zeroline=True, zerolinecolor="#9a958a")
    st.plotly_chart(style(fig, "Change in 8-week revenue at risk by DC (red = worse)", 380, legend=False), key="sl_dc")
with b2:
    u = run.scenario.node_util
    if len(u):
        ub = base.scenario.node_util.set_index("node_id")["utilisation"]
        u = u[u["node_type"].isin(["chokepoint", "port", "plant"])].copy()
        u["base"] = u["node_id"].map(ub)
        u = u.sort_values("utilisation", ascending=False).head(12)
        u["name"] = u["node_id"].map(nodes.set_index("node_id")["name"])
        fig = go.Figure()
        fig.add_bar(y=u["name"], x=u["base"], orientation="h", name="Baseline", marker_color="#c9c2b1")
        fig.add_bar(y=u["name"], x=u["utilisation"], orientation="h", name="Scenario", marker_color=TEAL)
        fig.add_vline(x=1.0, line_dash="dot", line_color=STATUS["Critical"], annotation_text="capacity", annotation_font_color=STATUS["Critical"])
        fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"))
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(style(fig, "Busiest nodes: utilisation under the flow plan", 380), key="sl_util")

# ---- Lane reroute table ------------------------------------------------------------------
section("Lane-level impact", "plant → DC replenishment lanes whose route or reachability changed")
chg = lanes[(lanes["base_path"] != lanes["scn_path"]) | ~lanes["scn_reachable"].fillna(False).astype(bool)].copy()
if chg.empty:
    st.success("No lane changes under this scenario.")
else:
    chg["status"] = np.where(chg["scn_reachable"].fillna(False).astype(bool), "Rerouted", "Unreachable")
    st.dataframe(chg[["plant_id", "dc_id", "status", "base_chokepoints", "scn_chokepoints", "base_transit_days", "scn_transit_days",
                      "delta_days", "lane_capacity_factor"]], hide_index=True, height=280, column_config={
        "base_chokepoints": "Baseline chokepoints", "scn_chokepoints": "Scenario chokepoints",
        "base_transit_days": st.column_config.NumberColumn("Base transit (d)", format="%.1f"),
        "scn_transit_days": st.column_config.NumberColumn("Scenario transit (d)", format="%.1f"),
        "delta_days": st.column_config.NumberColumn("Δ days", format="%+.1f"),
        "lane_capacity_factor": st.column_config.ProgressColumn("Lane capacity", min_value=0, max_value=1, format="percent")})

# ---- Compare all scenarios -----------------------------------------------------------------
section("Scenario league table", "same severity, duration and dials applied to every predefined scenario")
if st.button("▶ Run all scenarios (≈ 30–40 s, then cached)", type="primary"):
    st.session_state["sl_compare"] = True
if st.session_state.get("sl_compare"):
    rows = []
    prog = st.progress(0.0, text="Simulating…")
    names = [n for n, v in SCENARIOS.items() if n != "Regional supplier outage" and not v.get("hidden")]
    for i, name in enumerate(names):
        p = replace(c.params, scenario=name, demand_shock=SCENARIOS[name].get("default_shock", c.params.demand_shock)
                    if name == "Demand surge" else c.params.demand_shock)
        r = twin(p, c.policy)
        ss = r.series[m(r.series)]
        rows.append({"scenario": f"{SCENARIOS[name]['icon']} {name}", "revenue_at_risk": ss["expected_lost_revenue"].sum(),
                     "fill_rate": fill(ss), "lead_time": r.scenario.kpis["avg_lead_time_scenario"],
                     "served": r.scenario.kpis["network_served_share"], "cost_per_unit": r.scenario.kpis["cost_per_unit"],
                     "high_risk": int(ss["risk_band"].isin(["High", "Critical"]).sum())})
        prog.progress((i + 1) / len(names), text=f"Simulated {name}")
    prog.empty()
    lt = pd.DataFrame(rows).sort_values("revenue_at_risk")
    fig = go.Figure(go.Bar(x=lt["revenue_at_risk"], y=lt["scenario"], orientation="h",
                           marker=dict(color=lt["revenue_at_risk"], colorscale=[[0, "#cde2fb"], [1, NAVY]]),
                           text=[money(v) for v in lt["revenue_at_risk"]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>%{x:$,.0f} at risk<extra></extra>"))
    fig.update_xaxes(tickprefix="$", tickformat="~s", range=[0, lt["revenue_at_risk"].max() * 1.2])
    st.plotly_chart(style(fig, "8-week revenue at risk by scenario", 420, legend=False), key="sl_league")
    st.dataframe(lt.sort_values("revenue_at_risk", ascending=False), hide_index=True, column_config={
        "revenue_at_risk": st.column_config.NumberColumn("Revenue at risk", format="$%,.0f"),
        "fill_rate": st.column_config.NumberColumn("Fill rate", format="percent"),
        "lead_time": st.column_config.NumberColumn("Lead time (d)", format="%.1f"),
        "served": st.column_config.ProgressColumn("Served", min_value=0, max_value=1, format="percent"),
        "cost_per_unit": st.column_config.NumberColumn("Cost/unit", format="$%.2f"), "high_risk": "High-risk SKU×DC"})
