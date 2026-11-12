"""Inventory Optimisation – SLSQP safety-stock policy vs uniform z, efficient frontier, space use and recommendations."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import WarehouseMissing, table
from dashboard.components.layout import controls, page_hero
from dashboard.components.theme import (CATEGORY_COLORS, NAVY, STATUS, TEAL, kpi, kpis, money, pct, section, style)

c = controls()
page_hero("Prescriptive analytics", "Inventory policy optimisation",
          "SciPy SLSQP per DC: minimise holding + expected shortage cost under shelf-space limits")
try:
    pol, dcs, fr, summ = (table("ml.inventory_policy"), table("ml.inventory_policy_dc"), table("ml.inventory_frontier"),
                          table("ml.inventory_policy_summary").iloc[0])
except WarehouseMissing as exc:
    st.error(str(exc))
    st.stop()
if c.filters.categories:
    pol = pol[pol["category"].isin(c.filters.categories)]

saving = summ["current_total_cost"] - summ["optimal_total_cost"]
kpis([
    kpi("Annual cost saving", money(saving), f"{saving / summ['current_total_cost']:.1%} of holding + shortage cost", "good"),
    kpi("Fill rate", pct(summ["optimal_fill"], 2), f"current {pct(summ['current_fill'], 2)}", "teal",
        f"{100 * (summ['optimal_fill'] - summ['current_fill']):+.2f} pts", True),
    kpi("Same budget, better mix", pct(summ["same_budget_fill"], 2), "fill if today's $ were reallocated", "good",
        f"{100 * (summ['same_budget_fill'] - summ['current_fill']):+.2f} pts", True),
    kpi("Safety-stock investment", money(summ["optimal_ss_value"]), f"current {money(summ['current_ss_value'])}", "navy",
        f"{summ['optimal_ss_value'] / summ['current_ss_value'] - 1:+.0%}", None),
    kpi("DCs solved to optimality", f"{int((dcs['solver_status'] == 'optimal').sum())}/{len(dcs)}", "SLSQP with analytic gradients", "good"),
])

a, b = st.columns([1.25, 1])
with a:
    fig = go.Figure()
    fig.add_scatter(x=fr["ss_investment"], y=fr["fill_rate"], mode="lines", line=dict(color=NAVY, width=2.6), name="Efficient frontier",
                    hovertemplate="$%{x:,.0f} safety stock<br>fill %{y:.2%}<extra></extra>")
    fig.add_scatter(x=[summ["current_ss_value"]], y=[summ["current_fill"]], mode="markers+text", name="Current (z = 1.65)",
                    marker=dict(size=14, color=STATUS["High"], symbol="circle"), text=["current"], textposition="bottom right")
    fig.add_scatter(x=[summ["current_ss_value"]], y=[summ["same_budget_fill"]], mode="markers+text", name="Same budget, optimal mix",
                    marker=dict(size=14, color=TEAL, symbol="diamond"), text=["same $"], textposition="top left")
    fig.add_scatter(x=[summ["optimal_ss_value"]], y=[summ["optimal_fill"]], mode="markers+text", name="Cost-optimal (SLSQP)",
                    marker=dict(size=15, color=NAVY, symbol="star"), text=["optimal"], textposition="top right")
    fig.add_annotation(x=summ["current_ss_value"], y=summ["same_budget_fill"], ax=summ["current_ss_value"], ay=summ["current_fill"],
                       xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowcolor=TEAL, text="")
    lo = max(0.9, fr["fill_rate"].min())
    fig.update_yaxes(tickformat=".1%", range=[lo, 1.001], title="expected fill rate")
    fig.update_xaxes(tickprefix="$", tickformat="~s", title="safety-stock investment", range=[0, summ["optimal_ss_value"] * 1.8])
    st.plotly_chart(style(fig, "Efficient frontier: service per inventory dollar", 430), key="op_frontier")
with b:
    smp = pol.sample(min(720, len(pol)), random_state=1)
    fig = go.Figure()
    for cat, g in smp.groupby("category"):
        fig.add_scatter(x=g["current_z"] + np.random.default_rng(0).normal(0, 0.015, len(g)), y=g["optimal_z"], mode="markers", name=cat,
                        marker=dict(size=np.clip(np.sqrt(g["unit_margin"]) * 1.1, 5, 22), color=CATEGORY_COLORS.get(cat), opacity=0.7,
                                    line=dict(width=1, color="white")),
                        customdata=list(zip(g["sku_id"], g["dc_id"], g["unit_margin"])),
                        hovertemplate="%{customdata[0]} @ %{customdata[1]}<br>z %{x:.2f} → %{y:.2f}<br>margin $%{customdata[2]:,.0f}<extra></extra>")
    fig.add_hline(y=1.65, line_dash="dot", line_color="#9a958a", annotation_text="uniform z = 1.65")
    fig.update_xaxes(title="current z (uniform)", range=[1.55, 1.75])
    fig.update_yaxes(title="optimised z")
    st.plotly_chart(style(fig, "Per-SKU service factor (bubble = unit margin)", 430), key="op_z")

section("Where the policy moves money", "per category and per DC")
a, b = st.columns(2)
with a:
    cat = pol.groupby("category").agg(cur=("current_ss_value", "sum"), opt=("optimal_ss_value", "sum"),
                                      cf=("current_fill_rate", "mean"), of=("optimal_fill_rate", "mean")).reset_index()
    fig = go.Figure()
    fig.add_bar(x=cat["category"], y=cat["cur"], name="Current", marker_color="#c9c2b1")
    fig.add_bar(x=cat["category"], y=cat["opt"], name="Optimised", marker_color=[CATEGORY_COLORS.get(k) for k in cat["category"]])
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    st.plotly_chart(style(fig, "Safety-stock value by category", 340), key="op_cat")
with b:
    d = dcs.assign(used=dcs["optimal_volume_m3"] / dcs["storage_m3"], label=dcs["dc_id"].str.replace("DC_", "").str.replace("_", " ").str.title())
    fig = go.Figure(go.Bar(x=d["label"], y=d["optimal_total_cost"] - d["current_total_cost"], marker_color=TEAL,
                           customdata=list(zip(d["used"], d["solver_status"])),
                           hovertemplate="%{x}<br>Δ annual cost %{y:$,.0f}<br>space used %{customdata[0]:.0%} · %{customdata[1]}<extra></extra>"))
    fig.update_yaxes(tickprefix="$", tickformat="~s", title="Δ annual cost (negative = saving)")
    st.plotly_chart(style(fig, "Annual cost change by DC (space constraint binding)", 340, legend=False), key="op_dc")

section("Recommendations", "largest changes in reorder point, with the economics behind them")
pol["rop_change"] = pol["optimal_reorder_point"] - pol["current_reorder_point"]
pol["cost_saving"] = pol["current_total_cost"] - pol["optimal_total_cost"]
rec = pol.reindex(pol["rop_change"].abs().sort_values(ascending=False).index)
st.dataframe(rec[["sku_id", "dc_id", "category", "unit_margin", "unit_cost", "current_z", "optimal_z", "current_reorder_point",
                  "optimal_reorder_point", "rop_change", "optimal_fill_rate", "cost_saving"]].head(200), hide_index=True, height=360,
             column_config={"unit_margin": st.column_config.NumberColumn("Margin", format="$%.0f"),
                            "unit_cost": st.column_config.NumberColumn("Cost", format="$%.0f"),
                            "current_z": st.column_config.NumberColumn("z now", format="%.2f"),
                            "optimal_z": st.column_config.NumberColumn("z opt", format="%.2f"),
                            "current_reorder_point": st.column_config.NumberColumn("ROP now", format="%,.0f"),
                            "optimal_reorder_point": st.column_config.NumberColumn("ROP opt", format="%,.0f"),
                            "rop_change": st.column_config.NumberColumn("Δ ROP", format="%+,.0f"),
                            "optimal_fill_rate": st.column_config.NumberColumn("Fill opt", format="percent"),
                            "cost_saving": st.column_config.NumberColumn("Saving/yr", format="$%,.0f")})
st.download_button("⬇ Download policy file for the ERP (CSV)", pol.to_csv(index=False), "inventory_policy.csv", "text/csv")
