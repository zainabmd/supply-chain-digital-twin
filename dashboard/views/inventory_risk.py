"""Stockout Risk – scenario-aware 8-week Monte Carlo outlook: heatmap, ranking and per-SKU projected inventory fan."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import WarehouseMissing, table, twin
from dashboard.components.layout import controls, page_hero, scenario_banner
from dashboard.components.theme import (CATEGORY_COLORS, NAVY, SEQ_SCALE, STATUS, STATUS_ORDER, TEAL, kpi, kpis, money, num,
                                        pct, section, style)

c = controls()
page_hero("Inventory risk", "8-week stockout outlook",
          "Monte Carlo over forecast quantiles, pipeline in transit, scenario lead times and the replenishment policy")
try:
    health = table("marts.mart_inventory_health")
except WarehouseMissing as exc:
    st.error(str(exc))
    st.stop()
run = twin(c.params, c.policy)
scenario_banner(run.scenario.kpis)
s = run.series
mask = c.filters.series_mask(s)
s = s[mask]

kpis([
    kpi("Critical", f"{int((s['risk_band'] == 'Critical').sum())}", "P(material stockout) > 80%", "bad"),
    kpi("High", f"{int((s['risk_band'] == 'High').sum())}", "50–80%", "serious"),
    kpi("Watch", f"{int((s['risk_band'] == 'Watch').sum())}", "20–50%", "warn"),
    kpi("Projected fill", pct(1 - s["expected_lost_units"].sum() / max(s["expected_demand_units"].sum(), 1)), "8 weeks", "teal"),
    kpi("Revenue at risk", money(s["expected_lost_revenue"].sum()), "expected lost sales", "bad"),
    kpi("Median cover", f"{health['days_of_cover'].median():.0f} d", "on hand today", "navy"),
])

# ---- Heatmap SKU x DC ---------------------------------------------------------------------
a, b = st.columns([1.6, 1])
with a:
    pv = s.pivot_table(index="sku_id", columns="dc_id", values="p_stockout")
    cat = s.drop_duplicates("sku_id").set_index("sku_id")["category"]
    order = cat.reindex(pv.index).sort_values().index
    pv = pv.loc[order]
    pv.columns = [cname.replace("DC_", "").replace("_", " ").title() for cname in pv.columns]
    fig = px.imshow(pv, color_continuous_scale=[[0, "#f0efec"], [0.2, "#fde6b0"], [0.5, "#ec835a"], [1, "#b2302f"]],
                    zmin=0, zmax=1, aspect="auto", labels=dict(color="P(stockout)"))
    fig.update_traces(xgap=1, ygap=1, hovertemplate="%{y} @ %{x}<br>P(material stockout) %{z:.0%}<extra></extra>")
    fig.update_yaxes(tickfont=dict(size=8))
    fig.update_layout(coloraxis_colorbar=dict(tickformat=".0%"))
    st.plotly_chart(style(fig, "Risk heatmap · SKU (grouped by category) × DC", 640, legend=False), key="ir_heat")
with b:
    band = s["risk_band"].value_counts().reindex(STATUS_ORDER).fillna(0)
    fig = go.Figure(go.Bar(x=band.index, y=band.values, marker_color=[STATUS[k] for k in band.index],
                           text=[f"{int(v)}" for v in band.values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}: %{y} SKU×DC<extra></extra>"))
    fig.update_yaxes(range=[0, band.max() * 1.2])
    st.plotly_chart(style(fig, "Risk bands (label + colour)", 280, legend=False), key="ir_band")
    bycat = s.groupby("category")["expected_lost_revenue"].sum().sort_values()
    fig = go.Figure(go.Bar(x=bycat.values, y=bycat.index, orientation="h", marker_color=[CATEGORY_COLORS.get(k, TEAL) for k in bycat.index],
                           hovertemplate="%{y}: %{x:$,.0f}<extra></extra>"))
    fig.update_xaxes(tickprefix="$", tickformat="~s")
    st.plotly_chart(style(fig, "Revenue at risk by category", 330, legend=False), key="ir_cat")

# ---- Ranking + drill-down ------------------------------------------------------------------
section("Risk ranking & drill-down", "select a row to see its projected inventory fan")
rank = s.sort_values(["expected_lost_revenue"], ascending=False).reset_index(drop=True)
cols = ["sku_id", "dc_id", "category", "risk_band", "p_stockout", "expected_fill_rate", "expected_lost_revenue",
        "median_days_to_stockout", "on_hand_units", "lt_base", "lt_scenario", "fulfil_factor"]
ev = st.dataframe(rank[cols].head(300), hide_index=True, height=330, on_select="rerun", selection_mode="single-row", key="ir_rank",
                  column_config={
                      "risk_band": "Band", "p_stockout": st.column_config.ProgressColumn("P(stockout)", min_value=0, max_value=1, format="percent"),
                      "expected_fill_rate": st.column_config.NumberColumn("Exp. fill", format="percent"),
                      "expected_lost_revenue": st.column_config.NumberColumn("Revenue at risk", format="$%,.0f"),
                      "median_days_to_stockout": st.column_config.NumberColumn("Days to stockout", format="%.0f"),
                      "on_hand_units": st.column_config.NumberColumn("On hand", format="%,.0f"),
                      "lt_base": st.column_config.NumberColumn("LT base", format="%.0f d"),
                      "lt_scenario": st.column_config.NumberColumn("LT scenario", format="%.0f d"),
                      "fulfil_factor": st.column_config.ProgressColumn("Supply avail.", min_value=0, max_value=1, format="percent")})
sel = rank.iloc[ev.selection.rows[0]] if ev.selection.rows else rank.iloc[0]
pos = run.series.reset_index(drop=True)
i = pos.index[(pos["sku_id"] == sel["sku_id"]) & (pos["dc_id"] == sel["dc_id"])][0]
days = np.arange(1, run.risk.traj_p50.shape[1] + 1)
fig = go.Figure()
fig.add_scatter(x=list(days) + list(days[::-1]), y=list(run.risk.traj_p90[i]) + list(run.risk.traj_p10[i][::-1]), fill="toself",
                fillcolor="rgba(57,135,229,0.18)", line=dict(width=0), name="On hand P10–P90", hoverinfo="skip")
fig.add_scatter(x=days, y=run.risk.traj_p50[i], name="On hand P50", mode="lines", line=dict(color=NAVY, width=2.6),
                hovertemplate="day %{x}<br>P50 on hand %{y:,.0f}<extra></extra>")
fig.add_hline(y=0, line_color=STATUS["Critical"], line_width=1.5)
if c.params.duration_weeks and not c.params.scenario.startswith("Baseline"):
    fig.add_vrect(x0=0, x1=min(7 * c.params.duration_weeks, days[-1]), fillcolor="#f2c26b", opacity=0.12, line_width=0,
                  annotation_text="disruption window", annotation_position="top left")
fig.update_xaxes(title="days ahead")
fig.update_yaxes(title="units on hand", rangemode="tozero")
st.plotly_chart(style(fig, f"{sel['sku_id']} @ {sel['dc_id']} · P(stockout) {sel['p_stockout']:.0%} · "
                           f"lead time {sel['lt_scenario']:.0f}d (base {sel['lt_base']:.0f}d)", 360), key="ir_fan")
st.caption(f"{num(len(s))} SKU×DC series × Monte Carlo paths · policy: {c.policy}. "
           "‘Material stockout’ = losing at least one full day of demand in the 8-week window.")
