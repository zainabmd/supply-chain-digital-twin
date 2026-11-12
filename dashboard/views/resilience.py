"""Resilience & SPOF – single points of failure, TTR vs TTS exposure, concentration, stress test, chokepoint history."""
from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import WarehouseMissing, table
from dashboard.components.layout import controls, page_hero
from dashboard.components.theme import (NAVY, NODE_STYLE, SEQ_SCALE, STATUS, TEAL, kpi, kpis, money, pct, section, style)

c = controls()
page_hero("Network resilience engineering", "Resilience & single points of failure",
          "Four lenses per node: max-flow loss · BOM / routing dependency · TTR vs TTS exposure · centrality")
try:
    gm, cut, stress = table("ml.graph_node_metrics"), table("ml.graph_min_cut"), table("ml.graph_stress_test")
    conc, cpx = table("ml.supplier_concentration"), table("marts.mart_chokepoint_exposure")
    sup = table("marts.mart_supplier_scorecard")
except WarehouseMissing as exc:
    st.error(str(exc))
    st.stop()

if c.filters.supplier_regions:
    gm = gm[~gm["node_type"].isin(["supplier_t2", "supplier_t3"]) | gm["region"].isin(c.filters.supplier_regions)]
gm = gm.assign(label=[NODE_STYLE.get(t, (t,))[0] for t in gm["node_type"]])
# Exact headroom = network max-flow ceiling / forecast demand (the stress grid steps in 20% increments, too coarse).
base_demand = stress.loc[stress["demand_multiplier"] == 1.0, "demand"].iloc[0]
headroom = stress["max_flow"].max() / base_demand
spof = gm[gm["is_single_point_of_failure"]]
kpis([
    kpi("Single points of failure", f"{len(spof)}", "> 5% revenue dependent or > 10% flow loss", "bad"),
    kpi("Largest dependency", spof.sort_values("revenue_at_risk_share").iloc[-1]["name"] if len(spof) else "–",
        f"{pct(spof['revenue_at_risk_share'].max())} of revenue single-sourced" if len(spof) else "", "bad"),
    kpi("Total risk exposure", money(gm["risk_exposure_usd"].sum()), "Σ revenue lost while TTR > TTS", "serious"),
    kpi("Nodes with TTR > TTS", f"{int((gm['ttr_days'] > gm['tts_days']).sum())}", "recovery outlasts inventory runway", "warn"),
    kpi("Demand surge headroom", f"+{100 * (headroom - 1):.0f}%", "before max-flow < demand", "warn" if headroom < 1.2 else "good"),
    kpi("Single-sourced items", f"{int((conc['n_suppliers'] == 1).sum())}", "components / materials with 1 supplier", "bad"),
])

# ---- Criticality ranking + TTR/TTS -------------------------------------------------------
a, b = st.columns([1, 1.15])
with a:
    top = gm.head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["criticality_score"], y=top["name"], orientation="h",
                           marker_color=[NODE_STYLE.get(t, ("", TEAL))[1] for t in top["node_type"]],
                           customdata=list(zip(top["label"], top["revenue_at_risk_share"], top["flow_loss_pct"].fillna(0), top["risk_exposure_usd"])),
                           hovertemplate="<b>%{y}</b> · %{customdata[0]}<br>criticality %{x:.0f}<br>revenue dependent %{customdata[1]:.0%}"
                                         "<br>flow loss %{customdata[2]:.0%}<br>exposure %{customdata[3]:$,.0f}<extra></extra>"))
    fig.update_xaxes(range=[0, 100], title="criticality (0–100)")
    st.plotly_chart(style(fig, "Top 15 critical nodes (colour = node type)", 470, legend=False), key="rs_rank")
with b:
    g = gm[gm["revenue_at_risk_weekly"] > 0].copy()
    g["bubble"] = np.sqrt(g["revenue_at_risk_weekly"] / g["revenue_at_risk_weekly"].max()) * 40 + 6
    mx = max(g["ttr_days"].max(), g["tts_days"].max()) * 1.08
    fig = go.Figure()
    fig.add_shape(type="path", path=f"M 0 0 L {mx} {mx} L {mx} 0 Z", fillcolor="rgba(208,59,59,0.07)", line_width=0)
    fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(color="#9a958a", dash="dot"))
    fig.add_annotation(x=mx * 0.8, y=mx * 0.2, text="TTR > TTS<br><b>revenue lost</b>", showarrow=False, font=dict(color=STATUS["Critical"]))
    fig.add_annotation(x=mx * 0.2, y=mx * 0.85, text="inventory outlasts<br>the outage", showarrow=False, font=dict(color="#0b6b3a"))
    for t, (lab, col, sym) in NODE_STYLE.items():
        sub = g[g["node_type"] == t]
        if len(sub):
            fig.add_scatter(x=sub["ttr_days"], y=sub["tts_days"], mode="markers", name=lab,
                            marker=dict(size=sub["bubble"], color=col, symbol=sym if sym != "x" else "x", opacity=0.8,
                                        line=dict(width=1, color="white")),
                            text=sub["name"], customdata=list(zip(sub["revenue_at_risk_weekly"], sub["risk_exposure_usd"])),
                            hovertemplate="<b>%{text}</b><br>TTR %{x}d · TTS %{y:.0f}d<br>%{customdata[0]:$,.0f}/wk dependent"
                                          "<br>exposure %{customdata[1]:$,.0f}<extra></extra>")
    fig.update_xaxes(title="time to recover (days)", range=[0, mx])
    fig.update_yaxes(title="time to survive (days of cover)", range=[0, mx])
    st.plotly_chart(style(fig, "Time-to-recover vs time-to-survive (bubble = weekly revenue dependent)", 470), key="rs_ttr")

# ---- Concentration + stress test ---------------------------------------------------------
a, b = st.columns(2)
with a:
    cc = conc.sort_values("hhi_country")
    fig = go.Figure(go.Bar(x=cc["hhi_country"], y=cc["item"], orientation="h",
                           marker=dict(color=cc["hhi_country"], colorscale=SEQ_SCALE, cmin=0, cmax=1),
                           customdata=list(zip(cc["n_suppliers"], cc["top_country"], cc["tier"])),
                           hovertemplate="<b>%{y}</b> (tier %{customdata[2]})<br>HHI %{x:.2f} · %{customdata[0]} supplier(s)"
                                         "<br>top country %{customdata[1]}<extra></extra>"))
    fig.add_vline(x=0.25, line_dash="dot", line_color=STATUS["High"], annotation_text="highly concentrated", annotation_font_color=STATUS["High"])
    fig.update_xaxes(range=[0, 1.05], title="Herfindahl index by supplier country (1 = single country)")
    st.plotly_chart(style(fig, "Sourcing concentration per component / material", 430, legend=False), key="rs_hhi")
with b:
    fig = go.Figure()
    fig.add_scatter(x=stress["demand_multiplier"], y=stress["served_share"], mode="lines+markers", line=dict(color=NAVY, width=2.4),
                    marker=dict(size=9), hovertemplate="demand ×%{x:.1f}<br>served %{y:.0%}<extra></extra>", name="served share")
    fig.add_hline(y=1.0, line_dash="dot", line_color="#9a958a")
    fig.update_xaxes(title="demand multiplier vs forecast")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], title="share of demand the network can serve")
    st.plotly_chart(style(fig, "Stress test: max-flow vs demand surge", 300, legend=False), key="rs_stress")
    cut_n = cut.assign(name=cut["from_node"].map(gm.set_index("node_id")["name"]).fillna(cut["from_node"]))
    st.markdown(f"**Binding min-cut at ×{cut['at_demand_multiplier'].iloc[0]:.1f} demand:** "
                + ", ".join(f"{n} ({cap:,.0f} u/wk)" for n, cap in zip(cut_n["name"], cut_n["capacity"])))

# ---- Chokepoint history + supplier scorecard --------------------------------------------
section("Chokepoint exposure history", "average delay of shipments transiting each chokepoint, by month")
pv = cpx.pivot_table(index="chokepoint", columns="month", values="avg_delay", aggfunc="mean")
vol = cpx.groupby("chokepoint")["units"].sum()
fig = px.imshow(pv, color_continuous_scale=[[0, "#f0efec"], [0.5, "#ec835a"], [1, "#b2302f"]], aspect="auto",
                labels=dict(color="avg delay (d)", x="", y=""))
fig.update_traces(xgap=2, ygap=2, hovertemplate="%{y} · %{x|%b %Y}<br>avg delay %{z:.1f} d<extra></extra>")
fig.update_yaxes(ticktext=[f"{i.replace('CP_', '').replace('_', ' ').title()} ({vol.get(i, 0) / 1e3:,.0f}K u)" for i in pv.index],
                 tickvals=list(pv.index))
st.plotly_chart(style(fig, None, 300, legend=False), key="rs_cp")

section("Supplier scorecard", "delivery performance + structural risk")
s2 = sup if not c.filters.supplier_regions else sup[sup["region"].isin(c.filters.supplier_regions)]
s2 = s2.merge(gm[["node_id", "criticality_score", "revenue_at_risk_share"]], left_on="supplier_id", right_on="node_id", how="left")
st.dataframe(s2.sort_values("criticality_score", ascending=False)[
    ["supplier_name", "tier", "country", "supplies", "single_source_items", "on_time_rate", "on_time_rate_h2_2024", "avg_quality_ppm",
     "ttr_days", "risk_score", "revenue_at_risk_share", "criticality_score"]], hide_index=True, height=360, column_config={
    "supplier_name": "Supplier", "single_source_items": "Single-sourced",
    "on_time_rate": st.column_config.ProgressColumn("On-time (2y)", min_value=0, max_value=1, format="percent"),
    "on_time_rate_h2_2024": st.column_config.NumberColumn("On-time H2-24", format="percent"),
    "avg_quality_ppm": st.column_config.NumberColumn("Quality (ppm)", format="%.0f"), "ttr_days": "TTR (d)",
    "risk_score": st.column_config.NumberColumn("Geo/fin risk", format="%.2f"),
    "revenue_at_risk_share": st.column_config.ProgressColumn("Revenue dependent", min_value=0, max_value=1, format="percent"),
    "criticality_score": st.column_config.NumberColumn("Criticality", format="%.0f")})
