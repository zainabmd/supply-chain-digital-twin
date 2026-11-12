"""Network Twin – the multi-tier graph, coloured by risk, with a node drill-down and a one-click failure simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import WarehouseMissing, table, twin
from dashboard.components.layout import controls, page_hero
from dashboard.components.maps import network_map
from dashboard.components.theme import (INK_2, NODE_STYLE, SEQ_SCALE, STATUS, kpi, kpis, money, num, pct, section, style)
from src.scenarios import ScenarioParams

c = controls()
page_hero("Network digital twin", "Multi-tier network topology",
          "Raw materials → components → assembly → ports → maritime chokepoints → distribution centres")
try:
    nodes, edges, gm = table("raw.nodes"), table("raw.edges"), table("ml.graph_node_metrics")
except WarehouseMissing as exc:
    st.error(str(exc))
    st.stop()

nodes = nodes[c.filters.node_mask(nodes)]
nodes = nodes[nodes["node_type"] != "waypoint"]
keep = set(nodes["node_id"])
# Collapse open-ocean waypoints so sea routes read as port → chokepoint → port.
wp = set(table("raw.nodes").query("node_type == 'waypoint'")["node_id"])
e = edges.copy()
for _ in range(3):
    via = e[e["dst"].isin(wp)].merge(e[e["src"].isin(wp)], left_on="dst", right_on="src", suffixes=("", "_2"))
    via = via.assign(dst=via["dst_2"])[e.columns]
    e = pd.concat([e[~e["dst"].isin(wp) & ~e["src"].isin(wp)], via]).drop_duplicates(["src", "dst"])
e = e[e["src"].isin(keep) & e["dst"].isin(keep) & (e["src"] != e["dst"])]

# ---- Tiered layout -----------------------------------------------------------------
origin_ports = set(e.loc[e["edge_type"] == "inland", "dst"]) & set(nodes.loc[nodes["node_type"] == "port", "node_id"])
col = {"supplier_t3": 0, "supplier_t2": 1, "plant": 2, "chokepoint": 4, "dc": 6}
nodes = nodes.assign(x=[col.get(t, 3 if n in origin_ports else 5) for n, t in zip(nodes["node_id"], nodes["node_type"])])
nodes = nodes.sort_values(["x", "region", "lat"])
nodes["y"] = nodes.groupby("x").cumcount()
nodes["y"] = nodes["y"] / nodes.groupby("x")["y"].transform("max").replace(0, 1)
P = nodes.set_index("node_id")[["x", "y"]]

metric_opts = {"Criticality score": "criticality_score", "Revenue at risk (share)": "revenue_at_risk_share",
               "Flow loss if removed": "flow_loss_pct", "Risk exposure ($)": "risk_exposure_usd", "Betweenness": "betweenness"}
a1, a2, a3 = st.columns([1.4, 1, 1])
metric_name = a1.selectbox("Colour nodes by", list(metric_opts), key="nt_metric")
show_supply = a2.toggle("Show supply links", True, key="nt_supply")
view = a3.segmented_control("View", ["Tiered graph", "Geographic"], default="Tiered graph", key="nt_view") or "Tiered graph"
mcol = metric_opts[metric_name]
vals = nodes["node_id"].map(gm.set_index("node_id")[mcol]).fillna(0.0)
if mcol == "risk_exposure_usd":
    vals = np.log10(vals + 1)

if view == "Tiered graph":
    fig = go.Figure()
    styles = {"material": ("#b3a9e6", 0.6), "component": ("#9ec5f4", 0.7), "inland": ("#9a958a", 1.0),
              "sea_leg": ("#256abf", 1.1), "last_mile": ("#9a958a", 1.0)}
    for et, (color, w) in styles.items():
        if et in ("material", "component") and not show_supply:
            continue
        sub = e[e["edge_type"] == et]
        xs, ys = [], []
        for r in sub.itertuples():
            xs += [P.at[r.src, "x"], P.at[r.dst, "x"], None]
            ys += [P.at[r.src, "y"], P.at[r.dst, "y"], None]
        fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=w), opacity=0.45, hoverinfo="skip",
                        name={"material": "Material flow", "component": "Component flow", "inland": "Inland / rail",
                              "sea_leg": "Sea lane", "last_mile": "Last mile"}[et])
    # ttr_days already comes from the node table: take only analytics columns from gm (avoids _x/_y suffixes)
    g = nodes.merge(gm[["node_id", "criticality_score", "revenue_at_risk_share", "flow_loss_pct", "tts_days"]],
                    on="node_id", how="left")
    g["v"] = vals.to_numpy()
    for ntype, (label, _, symbol) in NODE_STYLE.items():
        sub = g[g["node_type"] == ntype]
        if sub.empty:
            continue
        fig.add_scatter(x=sub["x"], y=sub["y"], mode="markers+text", name=label, text=sub["name"].str.slice(0, 18),
                        textposition="middle right", textfont=dict(size=9, color=INK_2),
                        marker=dict(size=15, symbol=symbol if symbol != "x" else "x-thin-open", color=sub["v"],
                                    colorscale=SEQ_SCALE, cmin=float(vals.min()), cmax=float(vals.max() or 1),
                                    line=dict(width=1.5, color="#14213d"),
                                    colorbar=dict(title=metric_name.split(" (")[0], len=0.6, thickness=12) if ntype == "plant" else None,
                                    showscale=ntype == "plant"),
                        # list of tuples (not np.stack) keeps numbers numeric for the hover formatting
                        customdata=list(zip(sub["node_id"], sub["criticality_score"].fillna(0), sub["revenue_at_risk_share"].fillna(0),
                                            sub["ttr_days"], sub["tts_days"].fillna(0))),
                        hovertemplate="<b>%{text}</b> (" + label + ")<br>criticality %{customdata[1]:.0f}<br>revenue dependent "
                                      "%{customdata[2]:.0%}<br>TTR %{customdata[3]}d · TTS %{customdata[4]:.0f}d<extra></extra>")
    for x, lbl in zip([0, 1, 2, 3, 4, 5, 6], ["Tier-3 raw materials", "Tier-2 components", "Assembly plants", "Origin ports",
                                              "Chokepoints", "Destination ports", "Distribution centres"]):
        fig.add_annotation(x=x, y=1.07, text=f"<b>{lbl}</b>", showarrow=False, font=dict(size=11, color="#0b2545"))
    fig.update_xaxes(visible=False, range=[-0.55, 7.7])   # room for the first header and the DC labels on the right
    fig.update_yaxes(visible=False, range=[-0.05, 1.12])
    ev = st.plotly_chart(style(fig, None, 760), key="nt_graph", on_select="rerun", selection_mode="points")
else:
    ev = None
    st.plotly_chart(network_map(nodes, edges, None, size_by=gm.set_index("node_id")["criticality_score"], show_supply_links=show_supply,
                                height=600, title="Geographic view · node size = criticality"), key="nt_geo")

# ---- Drill-down --------------------------------------------------------------------------
section("Node drill-down", "click a node in the tiered graph, or pick one here")
picked = None
try:
    pts = ev.selection.points if ev else []
    if pts and pts[0].get("customdata"):
        picked = pts[0]["customdata"][0]
except AttributeError:
    picked = None
names = gm.set_index("node_id")["name"]
order = gm["node_id"].tolist()
if picked in order:
    st.session_state["nt_node"] = picked
node = st.selectbox("Node", order, format_func=lambda n: f"{names.get(n, n)} · {n}", key="nt_node")
r = gm.set_index("node_id").loc[node]
kpis([
    kpi("Type", NODE_STYLE.get(r["node_type"], (r["node_type"],))[0], f"{r['region']} · {r['country']}", "navy"),
    kpi("Criticality", f"{r['criticality_score']:.0f}/100", "composite of the 4 lenses", "bad" if r["criticality_score"] > 50 else "warn"),
    kpi("Revenue dependent", pct(r["revenue_at_risk_share"]), f"{money(r['revenue_at_risk_weekly'])}/week with no alternative",
        "bad" if r["revenue_at_risk_share"] > 0.05 else "good"),
    kpi("Flow lost if removed", pct(r["flow_loss_pct"] if r["flow_loss_pct"] == r["flow_loss_pct"] else 0), "max-flow re-solve", "serious"),
    kpi("TTR vs TTS", f"{r['ttr_days']:.0f}d vs {r['tts_days']:.0f}d", "recover vs survive", "bad" if r["ttr_days"] > r["tts_days"] else "good"),
    kpi("Risk exposure", money(r["risk_exposure_usd"]), "revenue lost while uncovered", "bad" if r["risk_exposure_usd"] > 0 else "good"),
])
left, right = st.columns([1, 1.2])
with left:
    up = edges[edges["dst"] == node][["src", "edge_type", "mode", "transit_days", "capacity_per_week"]].rename(columns={"src": "node"})
    dn = edges[edges["src"] == node][["dst", "edge_type", "mode", "transit_days", "capacity_per_week"]].rename(columns={"dst": "node"})
    nb = pd.concat([up.assign(direction="⬅ upstream"), dn.assign(direction="➡ downstream")])
    nb["name"] = nb["node"].map(names).fillna(nb["node"])
    st.dataframe(nb[["direction", "name", "edge_type", "mode", "transit_days", "capacity_per_week"]], hide_index=True, height=300,
                 column_config={"transit_days": st.column_config.NumberColumn("Transit (d)", format="%.1f"),
                                "capacity_per_week": st.column_config.NumberColumn("Capacity /wk", format="%,.0f")})
with right:
    if st.button(f"🎯 Simulate a 6-week outage of {names.get(node, node)}", type="primary", key="nt_sim"):
        st.session_state["nt_sim_node"] = node
    if st.session_state.get("nt_sim_node") == node:
        p = ScenarioParams(scenario="Custom node outage", custom_nodes=(node,), severity=1.0, duration_weeks=6)
        run, base = twin(p, c.policy), twin(ScenarioParams(), c.policy)
        dl = run.series["expected_lost_revenue"].sum() - base.series["expected_lost_revenue"].sum()
        fs = 1 - run.series["expected_lost_units"].sum() / run.series["expected_demand_units"].sum()
        fb = 1 - base.series["expected_lost_units"].sum() / base.series["expected_demand_units"].sum()
        kpis([kpi("Revenue at risk Δ", money(dl), "8-week outlook", "bad" if dl > 0 else "good"),
              kpi("Fill rate", pct(fs), f"baseline {pct(fb)}", "warn", f"{100 * (fs - fb):+.1f} pts", fs >= fb),
              kpi("Lanes rerouted / cut", f"{run.scenario.kpis['lanes_rerouted']} / {run.scenario.kpis['lanes_unreachable']}", "", "serious"),
              kpi("SKUs supply-impaired", f"{run.scenario.kpis['skus_supply_impaired']}", "BOM", "bad")])
        worst = run.series[["sku_id", "dc_id", "expected_lost_revenue"]].merge(
            base.series[["sku_id", "dc_id", "expected_lost_revenue"]], on=["sku_id", "dc_id"], suffixes=("", "_base"))
        worst["delta"] = worst["expected_lost_revenue"] - worst["expected_lost_revenue_base"]
        worst = worst.groupby("dc_id")["delta"].sum().sort_values(ascending=False).head(8).reset_index()
        fig = go.Figure(go.Bar(x=worst["delta"], y=worst["dc_id"], orientation="h", marker_color=STATUS["High"],
                               hovertemplate="%{y}: %{x:$,.0f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickprefix="$", tickformat="~s")
        st.plotly_chart(style(fig, "Most affected DCs (Δ revenue at risk)", 280, legend=False), key="nt_sim_chart")
    else:
        st.info("Run a single-node failure through the full twin: routing, flow plan and Monte Carlo.")
st.caption(f"{num(len(nodes))} nodes · {num(len(e))} links shown (open-ocean waypoints collapsed).")
