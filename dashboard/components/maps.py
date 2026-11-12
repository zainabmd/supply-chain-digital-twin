"""Geographic network map: nodes by type (colour + symbol), lanes weighted by planned flow."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.components.theme import MUTED, NODE_STYLE, STATUS, TEAL

FLOW_BUCKETS = [(0.0, 0.12, 1.2), (0.12, 0.35, 2.4), (0.35, 0.7, 3.8), (0.7, 1.01, 5.5)]


def network_map(nodes: pd.DataFrame, edges: pd.DataFrame, flows: pd.DataFrame | None = None, *,
                closed: set[str] | frozenset = frozenset(), size_by: pd.Series | None = None,
                show_supply_links: bool = False, highlight_edges: set[tuple[str, str]] | None = None,
                height: int = 520, title: str | None = None, flow_color: str = TEAL) -> go.Figure:
    pos = nodes.set_index("node_id")[["lat", "lon"]]
    fig = go.Figure()

    def add_lines(pairs, color, width, name, dash=None, opacity=0.8, show=True):
        lat, lon = [], []
        for a, b in pairs:
            if a in pos.index and b in pos.index:
                lat += [pos.at[a, "lat"], pos.at[b, "lat"], None]
                lon += [pos.at[a, "lon"], pos.at[b, "lon"], None]
        if lat:
            fig.add_scattergeo(lat=lat, lon=lon, mode="lines", line=dict(width=width, color=color, dash=dash),
                               opacity=opacity, hoverinfo="skip", name=name, showlegend=show)

    # Supply links (tier-3 → tier-2 → plants): faint, optional
    if show_supply_links:
        sup = edges[edges["edge_type"].isin(["material", "component"])]
        add_lines(zip(sup["src"], sup["dst"]), "#9b8fd6", 0.6, "Supply links", opacity=0.35)

    fg = edges[edges["edge_type"].isin(["sea_leg", "inland", "last_mile"])]
    if flows is not None and len(flows):
        f = flows.groupby(["src", "dst"], as_index=False)["flow"].sum()
        f = f[f["src"].isin(pos.index) & f["dst"].isin(pos.index)]
        unused = fg.merge(f, on=["src", "dst"], how="left")
        unused = unused[unused["flow"].isna()]
        add_lines(zip(unused["src"], unused["dst"]), "#c9c2b1", 0.7, "Available lane (no planned flow)", dash="dot", opacity=0.55)
        rel = f["flow"] / max(f["flow"].max(), 1e-9)
        for i, (lo, hi, w) in enumerate(FLOW_BUCKETS):
            sel = f[(rel >= lo) & (rel < hi)]
            add_lines(zip(sel["src"], sel["dst"]), flow_color, w, "Planned flow (width = volume)" if i == 0 else None,
                      opacity=0.85, show=i == 0)
    else:
        add_lines(zip(fg["src"], fg["dst"]), "#8fb3c9", 1.0, "Lanes", opacity=0.6)
    if highlight_edges:
        add_lines(highlight_edges, STATUS["High"], 3.2, "Rerouted lane", opacity=0.95)

    # Nodes
    for ntype, (label, color, symbol) in NODE_STYLE.items():
        sub = nodes[nodes["node_type"] == ntype]
        if sub.empty or ntype == "waypoint":
            continue
        if size_by is not None:
            sz = 7 + 18 * size_by.reindex(sub["node_id"]).fillna(0).clip(0, 100).to_numpy() / 100
        else:
            sz = np.full(len(sub), 11 if ntype in ("chokepoint", "dc", "plant") else 8)
        fig.add_scattergeo(
            lat=sub["lat"], lon=sub["lon"], mode="markers", name=label,
            marker=dict(size=sz, color=color, symbol=symbol, line=dict(width=1, color="white"), opacity=0.92),
            customdata=list(zip(sub["name"], sub["country"], sub["capacity_per_week"].round(0))),
            hovertemplate="<b>%{customdata[0]}</b><br>" + label + " · %{customdata[1]}<br>capacity %{customdata[2]:,.0f} u/wk<extra></extra>",
        )
    if closed:
        c = nodes[nodes["node_id"].isin(closed)]
        fig.add_scattergeo(lat=c["lat"], lon=c["lon"], mode="markers+text", name="Disrupted node",
                           marker=dict(size=22, color="rgba(208,59,59,0.18)", line=dict(width=2.5, color=STATUS["Critical"]),
                                       symbol="circle"),
                           text=c["name"], textposition="top center", textfont=dict(color=STATUS["Critical"], size=11),
                           hovertemplate="<b>%{text}</b><br>disrupted in this scenario<extra></extra>")
    fig.update_geos(lataxis_range=[-45, 66], lonaxis_range=[-130, 155], projection_type="natural earth",
                    center=dict(lon=15, lat=15))
    fig.update_layout(height=height, title_text=title, margin=dict(l=0, r=0, t=46 if title else 6, b=0),
                      legend=dict(orientation="h", y=-0.02, yanchor="top", x=0, font=dict(size=11, color=MUTED)))
    return fig
