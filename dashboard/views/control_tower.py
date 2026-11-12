"""Control Tower – the executive one-pager: service, risk, resilience and the live network under the chosen scenario."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import WarehouseMissing, forecast_metrics, snapshot, table, twin
from dashboard.components.layout import controls, page_hero, scenario_banner
from dashboard.components.maps import network_map
from dashboard.components.theme import (NAVY, REGION_COLORS, STATUS, TEAL, alert, kpi, kpis, money, num, pct, section,
                                        style)
from src.scenarios import ScenarioParams

c = controls()
page_hero("Global supply-chain control tower", "Network health, risk outlook & resilience",
          "60 SKUs · 12 distribution centres · 7 plants · 36 tier-2/3 suppliers · 15 ports · 8 maritime chokepoints")
try:
    snap = snapshot()
    kpi_wk = table("marts.mart_service_kpis_weekly")
    events = table("marts.mart_disruption_impact")
    nodes = table("raw.nodes")
    edges = table("raw.edges")
    gm = table("ml.graph_node_metrics")
except WarehouseMissing as exc:
    st.error(str(exc))
    st.stop()

run = twin(c.params, c.policy)
base = twin(ScenarioParams(), c.policy) if not c.params.scenario.startswith("Baseline") or c.params.demand_shock \
    or c.params.lead_time_multiplier != 1 else run
scenario_banner(run.scenario.kpis)
fm = forecast_metrics()

s, b = run.series[c.filters.series_mask(run.series)], base.series[c.filters.series_mask(base.series)]
fill_s = 1 - s["expected_lost_units"].sum() / max(s["expected_demand_units"].sum(), 1)
fill_b = 1 - b["expected_lost_units"].sum() / max(b["expected_demand_units"].sum(), 1)
rar_s, rar_b = s["expected_lost_revenue"].sum(), b["expected_lost_revenue"].sum()
high_s = int(s["risk_band"].isin(["High", "Critical"]).sum())
high_b = int(b["risk_band"].isin(["High", "Critical"]).sum())
served = run.scenario.kpis.get("network_served_share", float("nan"))
cpu_s, cpu_b = run.scenario.kpis.get("cost_per_unit"), base.scenario.kpis.get("cost_per_unit")
is_what_if = run is not base

kpis([
    kpi("Fill rate · last 13 wks", pct(snap.get("fill_rate_13w", float("nan"))), "actual, all DCs", "teal"),
    kpi("OTIF · last 13 wks", pct(snap.get("otif_13w", float("nan"))), "vs contracted lead time", "serious"
        if snap.get("otif_13w", 1) < 0.8 else "good"),
    kpi("Projected fill · 8 wks", pct(fill_s), "Monte Carlo outlook", "good" if fill_s > 0.97 else "warn",
        f"{100 * (fill_s - fill_b):+.1f} pts" if is_what_if else None, fill_s >= fill_b),
    kpi("Revenue at risk · 8 wks", money(rar_s), "expected lost sales", "bad" if rar_s > 2 * rar_b else "serious",
        f"{money(rar_s - rar_b)} vs base" if is_what_if else None, rar_s <= rar_b),
    kpi("SKU×DC at high risk", num(high_s), "P(lose ≥1 day of demand) > 50%", "bad" if high_s > high_b * 1.5 else "warn",
        f"{high_s - high_b:+d}" if is_what_if else None, high_s <= high_b),
    kpi("Demand served (network)", pct(served), "flow plan, next 13 wks", "good" if served > 0.99 else "bad"),
    kpi("Cost to serve", f"${cpu_s:,.2f}/u" if cpu_s == cpu_s else "–", "logistics, per unit", "navy",
        f"{(cpu_s / cpu_b - 1):+.1%}" if is_what_if and cpu_b else None,
        ((cpu_s or 0) <= (cpu_b or 0)) if served >= 0.95 else None),   # cheaper only because volume can't ship ≠ good
    kpi("Forecast WAPE", pct(fm.get("wape", float("nan"))), f"vs {pct(fm.get('wape_seasonal_naive', float('nan')))} seasonal-naive", "violet"),
    kpi("Inventory value", money(snap.get("inventory_value", float("nan"))), f"{snap.get('days_of_supply', 0):.1f} days of supply", "navy"),
    kpi("Single points of failure", f"{int(snap.get('single_points_of_failure', 0))}", "nodes > 5% revenue or > 10% flow", "bad"),
])

# ---- Map + alert rail ----------------------------------------------------------------
left, right = st.columns([2.6, 1])
with left:
    flows = run.scenario.flows
    closed = run.scenario.closed_nodes & set(nodes["node_id"])
    lanes = run.scenario.lanes
    rerouted = lanes[(lanes["base_path"] != lanes["scn_path"]) & lanes["scn_reachable"].fillna(False)]
    hl = set()
    for p in rerouted["scn_path"].dropna():
        hops = p.split("→")
        hl |= set(zip(hops, hops[1:]))
    nshow = nodes[c.filters.node_mask(nodes)]
    crit = gm.set_index("node_id")["criticality_score"]
    fig = network_map(nshow, edges, flows, closed=closed, size_by=crit, highlight_edges=hl if is_what_if else None,
                      height=540, title="Live network · lane width = planned weekly flow · node size = criticality")
    st.plotly_chart(fig, key="ct_map")
with right:
    st.markdown("**Priority alerts**")
    cards = []
    for n in run.scenario.affected_nodes[:4]:
        name = nodes.set_index("node_id")["name"].get(n, n)
        cards.append(alert(f"{name} disrupted", f"{c.params.scenario} · severity {c.params.severity:.0%}", "Critical"))
    top = s.sort_values("expected_lost_revenue", ascending=False).head(5)
    for r in top.itertuples():
        band = r.risk_band if r.risk_band in STATUS else "Watch"
        cards.append(alert(f"{r.sku_id} @ {r.dc_id.replace('DC_', '').replace('_', ' ').title()}",
                           f"P(stockout) {r.p_stockout:.0%} · {money(r.expected_lost_revenue)} at risk · LT {r.lt_scenario:.0f}d", band))
    spof = gm[gm["is_single_point_of_failure"]].head(3)
    for r in spof.itertuples():
        cards.append(alert(f"SPOF: {r.name}", f"{pct(r.revenue_at_risk_share, 0)} of revenue dependent · TTR {r.ttr_days}d vs TTS {r.tts_days:.0f}d",
                           "High" if r.uncovered_days > 0 else "Watch"))
    st.markdown("".join(cards), unsafe_allow_html=True)

# ---- Service trend with disruption bands --------------------------------------------
section("Service performance through two years of disruption", "weekly · shaded bands = historical events")
w = kpi_wk.copy()
if c.filters.dc_regions:
    w = w[w["dc_region"].isin(c.filters.dc_regions)]
agg = w.groupby("week_start").agg(sales=("sales_units", "sum"), demand=("demand_units", "sum"),
                                  ship=("shipments", "sum"), ot=("otif", lambda x: (x * w.loc[x.index, "shipments"]).sum())).reset_index()
agg["fill"] = agg["sales"] / agg["demand"]
agg["otif"] = agg["ot"] / agg["ship"]
fig = go.Figure()
for e in events.itertuples():
    fig.add_vrect(x0=e.start_date, x1=e.end_date, fillcolor="#f2c26b", opacity=0.13, line_width=0)
    short = " ".join(e.event_name.replace("–", " ").split()[:2])
    fig.add_annotation(x=e.start_date, y=1.02, yref="paper", text=short, showarrow=False,
                       font=dict(size=9, color="#8a5300"), xanchor="left", textangle=-30)
fig.add_scatter(x=agg["week_start"], y=agg["fill"], name="Fill rate", mode="lines", line=dict(color=TEAL, width=2.4),
                hovertemplate="%{x|%d %b %Y}<br>fill %{y:.1%}<extra></extra>")
fig.add_scatter(x=agg["week_start"], y=agg["otif"], name="OTIF (on-time arrivals)", mode="lines", line=dict(color=NAVY, width=1.6, dash="dot"),
                hovertemplate="%{x|%d %b %Y}<br>OTIF %{y:.1%}<extra></extra>")
fig.update_yaxes(tickformat=".0%", range=[0.3, 1.02])
st.plotly_chart(style(fig, None, 340), key="ct_trend")

# ---- Regional risk + events table ------------------------------------------------------
a, b2 = st.columns([1, 1.25])
with a:
    reg = pd.DataFrame({"Scenario": s.groupby("dc_region")["expected_lost_revenue"].sum(),
                        "Baseline": b.groupby("dc_region")["expected_lost_revenue"].sum()}).fillna(0).reset_index()
    reg = reg.sort_values("Scenario")
    fig = go.Figure()
    fig.add_bar(y=reg["dc_region"], x=reg["Baseline"], orientation="h", name="Baseline", marker_color="#c9c2b1",
                hovertemplate="%{y}: %{x:$,.0f}<extra>baseline</extra>")
    fig.add_bar(y=reg["dc_region"], x=reg["Scenario"], orientation="h", name="Selected scenario",
                marker_color=[REGION_COLORS.get(r, TEAL) for r in reg["dc_region"]],
                hovertemplate="%{y}: %{x:$,.0f}<extra>scenario</extra>")
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickprefix="$", tickformat="~s")
    st.plotly_chart(style(fig, "8-week revenue at risk by DC region", 330), key="ct_region")
with b2:
    ev = events[["event_name", "category", "start_date", "duration_days", "avg_delay_days", "on_time_rate", "lost_share_during",
                 "lost_revenue_during"]].sort_values("start_date")
    st.markdown("**Historical disruptions · measured impact**")
    st.dataframe(ev, hide_index=True, height=300, column_config={
        "event_name": "Event", "category": "Type", "start_date": st.column_config.DateColumn("Start", format="MMM YYYY"),
        "duration_days": "Days", "avg_delay_days": st.column_config.NumberColumn("Avg delay (d)", format="%.1f"),
        "on_time_rate": st.column_config.ProgressColumn("On-time", min_value=0, max_value=1, format="percent"),
        "lost_share_during": st.column_config.NumberColumn("Lost sales %", format="percent"),
        "lost_revenue_during": st.column_config.NumberColumn("Lost revenue", format="$%,.0f")})
st.caption(f"Twin run: {run.seconds:.1f}s · routing (NetworkX) → flow plan (SciPy HiGHS) → {len(s):,} series × Monte Carlo. "
           f"Snapshot published {pd.Timestamp(snap.get('published_at')).strftime('%d %b %Y %H:%M UTC') if snap.get('published_at') else '–'}.")
