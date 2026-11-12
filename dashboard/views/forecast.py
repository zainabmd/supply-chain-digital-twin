"""Demand Forecast – probabilistic 13-week outlook (fan chart), backtest vs baselines, accuracy and drivers."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import WarehouseMissing, forecast_metrics, table
from dashboard.components.layout import controls, page_hero
from dashboard.components.theme import (CATEGORY_COLORS, NAVY, SEQ, TEAL, kpi, kpis, num, pct, section, style)

c = controls()
page_hero("Demand planning", "Probabilistic demand forecast",
          "Global gradient-boosted quantile model · direct 13-week horizon · P10 / P50 / P90")
try:
    wk, fc, bt = table("marts.mart_weekly_sku_dc"), table("ml.forecast_weekly"), table("ml.forecast_backtest")
    imp, skus = table("ml.forecast_feature_importance"), table("raw.skus")
except WarehouseMissing as exc:
    st.error(str(exc))
    st.stop()
m = forecast_metrics()
if c.filters.categories:
    keep = set(skus.loc[skus["category"].isin(c.filters.categories), "sku_id"])
    wk, fc, bt = wk[wk["sku_id"].isin(keep)], fc[fc["sku_id"].isin(keep)], bt[bt["sku_id"].isin(keep)]
if c.filters.dc_regions:
    wk = wk[wk["dc_region"].isin(c.filters.dc_regions)]
    keep_dc = set(wk["dc_id"])
    fc, bt = fc[fc["dc_id"].isin(keep_dc)], bt[bt["dc_id"].isin(keep_dc)]

kpis([
    kpi("Backtest WAPE", pct(m.get("wape", float("nan"))), "13-week holdout · P50", "teal"),
    kpi("vs seasonal-naive", pct(m.get("wape_seasonal_naive", float("nan"))),
        f"{(1 - m.get('wape', 0) / m.get('wape_seasonal_naive', 1)):.0%} error reduction", "good"),
    kpi("vs 4-week average", pct(m.get("wape_ma4", float("nan"))), "planner's moving-average baseline", "navy"),
    kpi("Bias", f"{m.get('bias', 0):+.1%}", "P50 vs actual (− = under-forecast)", "warn" if abs(m.get("bias", 0)) > 0.05 else "good"),
    kpi("P10–P90 coverage", pct(m.get("coverage_p10_p90", float("nan"))), "target 80% (calibration)", "good"),
    kpi("Next-13-week demand", num(fc["p50"].sum()), f"P10 {num(fc['p10'].sum())} · P90 {num(fc['p90'].sum())}", "violet"),
])

# ---- Fan chart ---------------------------------------------------------------------------
section("Forecast explorer", "history (52 weeks) · backtest window · forward forecast with P10–P90 band")
a1, a2, a3 = st.columns(3)
level = a1.segmented_control("Level", ["Total network", "Category", "SKU × DC"], default="Category", key="fc_level") or "Category"
sel_cat = sel_sku = sel_dc = None
if level == "Category":
    sel_cat = a2.selectbox("Category", sorted(wk["category"].unique()), key="fc_cat")
elif level == "SKU × DC":
    top_skus = wk.groupby("sku_id")["demand_units"].sum().sort_values(ascending=False).index.tolist()
    sel_sku = a2.selectbox("SKU", top_skus, key="fc_sku")
    sel_dc = a3.selectbox("DC", sorted(wk["dc_id"].unique()), key="fc_dc")


def subset(df):
    if sel_cat is not None:
        cat_skus = set(skus.loc[skus["category"] == sel_cat, "sku_id"])
        return df[df["sku_id"].isin(cat_skus)]
    if sel_sku is not None:
        return df[(df["sku_id"] == sel_sku) & (df["dc_id"] == sel_dc)]
    return df


h = subset(wk).groupby("week_start")["demand_units"].sum()
h = h[h.index >= h.index.max() - pd.Timedelta(weeks=52)]
f = subset(fc).groupby("target_week")[["p10", "p50", "p90"]].sum()
b = subset(bt).groupby("target_week")[["p10", "p50", "p90", "y"]].sum()
color = CATEGORY_COLORS.get(sel_cat, TEAL)
fig = go.Figure()
fig.add_scatter(x=h.index, y=h.values, name="Actual demand", mode="lines", line=dict(color=NAVY, width=2),
                hovertemplate="%{x|%d %b %Y}<br>actual %{y:,.0f}<extra></extra>")
fig.add_scatter(x=list(b.index) + list(b.index[::-1]), y=list(b["p90"]) + list(b["p10"][::-1]), fill="toself",
                fillcolor="rgba(201,184,138,0.30)", line=dict(width=0), name="Backtest P10–P90", hoverinfo="skip")
fig.add_scatter(x=b.index, y=b["p50"], name="Backtest P50", mode="lines", line=dict(color="#a8935a", width=2, dash="dash"),
                hovertemplate="%{x|%d %b}<br>backtest P50 %{y:,.0f}<extra></extra>")
fig.add_scatter(x=list(f.index) + list(f.index[::-1]), y=list(f["p90"]) + list(f["p10"][::-1]), fill="toself",
                fillcolor="rgba(57,135,229,0.20)", line=dict(width=0), name="Forecast P10–P90", hoverinfo="skip")
fig.add_scatter(x=f.index, y=f["p50"], name="Forecast P50", mode="lines+markers", line=dict(color=SEQ[4], width=2.6),
                marker=dict(size=6), customdata=list(zip(f["p10"], f["p90"])),
                hovertemplate="%{x|%d %b %Y}<br>P50 %{y:,.0f}<br>P10–P90 %{customdata[0]:,.0f} – %{customdata[1]:,.0f}<extra></extra>")
fig.add_vline(x=h.index.max(), line_dash="dot", line_color="#9a958a")
fig.add_annotation(x=h.index.max(), y=1.03, yref="paper", text="forecast origin", showarrow=False, font=dict(size=10, color="#868b97"))
fig.update_yaxes(title="units / week", rangemode="tozero")
st.plotly_chart(style(fig, None, 430), key="fc_fan")

# ---- Accuracy breakdowns ---------------------------------------------------------------
a, b2, c3 = st.columns(3)
with a:
    bh = pd.Series(m.get("by_horizon", {})).sort_index()
    fig = go.Figure(go.Scatter(x=[int(k) for k in bh.index], y=bh.values, mode="lines+markers", line=dict(color=TEAL, width=2.4),
                               marker=dict(size=8), hovertemplate="h=%{x} wks · WAPE %{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_xaxes(title="weeks ahead", dtick=2)
    st.plotly_chart(style(fig, "Error grows gently with horizon", 300, legend=False), key="fc_h")
with b2:
    comp = pd.DataFrame({"model": ["Quantile GBM", "4-week MA", "Seasonal naive"],
                         "wape": [m.get("wape"), m.get("wape_ma4"), m.get("wape_seasonal_naive")]})
    fig = go.Figure(go.Bar(x=comp["model"], y=comp["wape"], marker_color=[TEAL, "#c9c2b1", "#c9c2b1"],
                           text=[pct(v) for v in comp["wape"]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", range=[0, comp["wape"].max() * 1.25])
    st.plotly_chart(style(fig, "Backtest WAPE vs planner baselines", 300, legend=False), key="fc_cmp")
with c3:
    bc = pd.Series(m.get("by_category", {})).sort_values()
    fig = go.Figure(go.Bar(x=bc.values, y=bc.index, orientation="h", marker_color=[CATEGORY_COLORS.get(k, TEAL) for k in bc.index],
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style(fig, "WAPE by category", 300, legend=False), key="fc_cat_wape")

a, b2 = st.columns([1, 1.2])
with a:
    ii = imp.head(14).iloc[::-1]
    fig = go.Figure(go.Bar(x=ii["importance"], y=ii["feature"], orientation="h", marker_color=NAVY,
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="permutation importance (Δ MAE on log demand)")
    st.plotly_chart(style(fig, "What drives the forecast", 420, legend=False), key="fc_imp")
with b2:
    t = fc.groupby(["sku_id", "dc_id"]).agg(p10=("p10", "sum"), p50=("p50", "sum"), p90=("p90", "sum")).reset_index()
    t["uncertainty"] = (t["p90"] - t["p10"]) / t["p50"].replace(0, float("nan"))
    st.markdown("**13-week forecast by SKU × DC** (sorted by volume)")
    st.dataframe(t.sort_values("p50", ascending=False), hide_index=True, height=380, column_config={
        "p10": st.column_config.NumberColumn("P10", format="%,.0f"), "p50": st.column_config.NumberColumn("P50", format="%,.0f"),
        "p90": st.column_config.NumberColumn("P90", format="%,.0f"),
        "uncertainty": st.column_config.ProgressColumn("Relative spread", min_value=0, max_value=2, format="%.2f")})
    st.download_button("⬇ Download forecast (CSV)", fc.to_csv(index=False), "forecast_13w.csv", "text/csv")
