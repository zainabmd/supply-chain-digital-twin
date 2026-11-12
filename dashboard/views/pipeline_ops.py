"""Pipeline & Data Quality – DAG structure, task run history, DQ gate results and model registry."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import WarehouseMissing, table
from dashboard.components.layout import page_hero
from dashboard.components.theme import NAVY, STATUS, TEAL, kpi, kpis, num, section, style
from src.pipeline import TASKS

page_hero("Operations", "Pipeline & data quality", "Airflow DAG `supply_chain_digital_twin` · daily 04:00 UTC")
try:
    runs, dq, reg = table("ops.task_runs"), table("ops.dq_results"), table("ops.model_registry")
except WarehouseMissing as exc:
    st.error(str(exc))
    st.stop()

last_run = runs.sort_values("finished_at").iloc[-1]["run_id"] if len(runs) else None
lr = runs[runs["run_id"] == last_run]
last_dq = dq[dq["checked_at"] == dq["checked_at"].max()]
kpis([
    kpi("Last run", "SUCCESS" if len(lr) and (lr["status"] == "success").all() else "ATTENTION", f"run {last_run}",
        "good" if len(lr) and (lr["status"] == "success").all() else "bad"),
    kpi("Tasks in DAG", f"{len(TASKS)}", "same registry for Airflow + local runner", "navy"),
    kpi("DQ checks passed", f"{int((last_dq['status'] == 'pass').sum())}/{len(last_dq)}", f"{int((last_dq['status'] == 'warn').sum())} warnings",
        "good" if not (last_dq["status"] == "fail").any() else "bad"),
    kpi("Model in production", reg[reg["status"] == "production"]["version"].iloc[-1] if len(reg) else "–", "demand forecaster", "violet"),
    kpi("Runs recorded", num(runs["run_id"].nunique()), "ops.task_runs", "teal"),
])

# ---- DAG graph ----------------------------------------------------------------------------
section("DAG", "dependencies from src.pipeline.TASKS")
from graphlib import TopologicalSorter  # noqa: E402

order = list(TopologicalSorter({k: set(v.deps) for k, v in TASKS.items()}).static_order())
level = {}
for t in order:
    level[t] = 1 + max([level[d] for d in TASKS[t].deps], default=-1)
by_lvl = pd.Series(level).groupby(pd.Series(level)).groups
ypos = {t: i - (len(ts) - 1) / 2 for lv, ts in by_lvl.items() for i, t in enumerate(sorted(ts))}
status = lr.set_index("task")["status"].to_dict() if len(lr) else {}
secs = lr.groupby("task")["seconds"].sum().to_dict() if len(lr) else {}
fig = go.Figure()
for t, spec in TASKS.items():
    for d in spec.deps:
        fig.add_scatter(x=[level[d], level[t]], y=[ypos[d], ypos[t]], mode="lines", line=dict(color="#c9c2b1", width=2), hoverinfo="skip",
                        showlegend=False)
xs, ys, cols, txt, hov = [], [], [], [], []
for t in TASKS:
    xs.append(level[t])
    ys.append(ypos[t])
    s = status.get(t, "not run")
    cols.append({"success": STATUS["Low"], "failed": STATUS["Critical"], "retry": STATUS["Watch"]}.get(s, "#9a958a"))
    txt.append(t.replace("_", " "))
    hov.append(f"<b>{t}</b><br>{TASKS[t].doc}<br>status: {s} · {secs.get(t, 0):.1f}s · retries {TASKS[t].retries}")
fig.add_scatter(x=xs, y=ys, mode="markers+text", text=txt, textposition="top center", textfont=dict(size=10, color=NAVY),
                marker=dict(size=22, color=cols, symbol="square", line=dict(width=2, color="white")), hovertext=hov, hoverinfo="text",
                showlegend=False)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False, range=[-1.6, 1.6])
st.plotly_chart(style(fig, None, 260, legend=False), key="po_dag")

a, b = st.columns([1.2, 1])
with a:
    if len(lr):
        d = lr.groupby("task", as_index=False)["seconds"].sum().set_index("task").reindex(order).dropna().reset_index()
        fig = go.Figure(go.Bar(x=d["seconds"], y=d["task"], orientation="h", marker_color=TEAL,
                               hovertemplate="%{y}: %{x:.1f}s<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="seconds")
        st.plotly_chart(style(fig, f"Task durations · last run ({d['seconds'].sum():.0f}s total)", 380, legend=False), key="po_dur")
with b:
    st.markdown("**Data-quality gate · latest results**")
    st.dataframe(last_dq[["check_name", "severity", "status", "violations", "description"]], hide_index=True, height=380,
                 column_config={"status": st.column_config.TextColumn("Status"), "violations": st.column_config.NumberColumn("Violations")})

section("Model registry", "every training run is versioned; one model is in production")
reg = reg.sort_values("trained_at", ascending=False)
reg["wape"] = [json.loads(m).get("wape") for m in reg["metrics_json"]]
reg["coverage"] = [json.loads(m).get("coverage_p10_p90") for m in reg["metrics_json"]]
st.dataframe(reg[["model_name", "version", "status", "trained_at", "train_rows", "wape", "coverage"]], hide_index=True,
             column_config={"wape": st.column_config.NumberColumn("Backtest WAPE", format="percent"),
                            "coverage": st.column_config.NumberColumn("P10–P90 coverage", format="percent"),
                            "train_rows": st.column_config.NumberColumn("Training rows", format="%,d")})
with st.expander("Full task-run history"):
    st.dataframe(runs.sort_values("finished_at", ascending=False), hide_index=True)
