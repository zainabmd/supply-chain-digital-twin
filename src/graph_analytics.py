"""
Network resilience analytics (NetworkX).

For every node in the multi-tier network, three complementary lenses answer
"what happens if this node fails?":

1. **Flow lens (max-flow).** The network is a capacitated flow graph:
   super-source → tier-3 → tier-2 → plants → ports/chokepoints → DCs →
   super-sink (sized by forecast demand). Node capacities are enforced by node
   splitting. Removing a node and re-solving max-flow gives the share of
   deliverable volume lost.
2. **Dependency lens (bill of materials and routing).**
   - Suppliers: which SKUs depend on a component or material that only this
     supplier provides? Single sourcing is invisible to max-flow, because flow is
     fungible.
   - Plants: SKUs with no alternate plant.
   - Logistics nodes: re-route every plant→DC lane without the node, measuring
     unreachable revenue and added transit days.
3. **Time lens (TTR vs TTS, after Simchi-Levi's Risk Exposure Index).**
   - Time-to-recover (TTR): how long the node takes to come back.
   - Time-to-survive (TTS): how long the affected SKU×DCs can keep serving from
     inventory on hand plus in the pipeline.
   - If TTR > TTS, revenue is lost for (TTR − TTS) days. That is the node's
     **risk exposure** in dollars.

Also computed: betweenness centrality on the logistics graph, the network-wide
minimum cut (the binding bottleneck), and supplier concentration (HHI) per
component.
"""
from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pandas as pd

from src.config import get_logger
from src.routing import NetworkState, all_routes, fg_graph
from src.warehouse import read_sql, write_df

log = get_logger("graph")
SRC, SNK = "__SOURCE__", "__SINK__"


def load_inputs() -> dict[str, pd.DataFrame]:
    q = {
        "nodes": "SELECT * FROM raw.nodes", "edges": "SELECT * FROM raw.edges", "bom": "SELECT * FROM raw.bom",
        "sku_plants": "SELECT * FROM raw.sku_plants", "comp_src": "SELECT * FROM raw.component_sources",
        "mat_src": "SELECT * FROM raw.material_sources", "sourcing": "SELECT * FROM raw.sourcing",
        "health": "SELECT * FROM marts.mart_inventory_health",
        "fc": "SELECT sku_id, dc_id, avg(p50) AS weekly_demand FROM ml.forecast_weekly GROUP BY ALL",
        "skus": "SELECT sku_id, unit_price FROM raw.skus",
    }
    return {k: read_sql(v) for k, v in q.items()}


# --------------------------------------------------------------------------- #
# Flow lens
# --------------------------------------------------------------------------- #
def capacity_graph(nodes: pd.DataFrame, edges: pd.DataFrame, dc_demand: dict[str, float]) -> nx.DiGraph:
    """Split every node v into v_in -> v_out with the node's capacity; add super source/sink."""
    G = nx.DiGraph()
    for r in nodes.itertuples():
        G.add_edge(f"{r.node_id}_in", f"{r.node_id}_out", capacity=float(r.capacity_per_week), node=r.node_id)
    for r in edges.itertuples():
        G.add_edge(f"{r.src}_out", f"{r.dst}_in", capacity=float(r.capacity_per_week))
    for r in nodes[nodes["node_type"] == "supplier_t3"].itertuples():
        G.add_edge(SRC, f"{r.node_id}_in", capacity=float(r.capacity_per_week))
    for dc, dem in dc_demand.items():
        G.add_edge(f"{dc}_out", SNK, capacity=float(dem))
    return G


FAILABLE = {"supplier_t3", "supplier_t2", "plant", "port", "chokepoint", "dc"}  # open-ocean waypoints can't "fail"


def flow_lens(nodes, edges, dc_demand) -> tuple[pd.DataFrame, float, list, pd.DataFrame]:
    G = capacity_graph(nodes, edges, dc_demand)
    base, _ = nx.maximum_flow(G, SRC, SNK, flow_func=nx.algorithms.flow.edmonds_karp)
    rows = []
    for r in nodes[nodes["node_type"].isin(FAILABLE)].itertuples():
        e = (f"{r.node_id}_in", f"{r.node_id}_out")
        cap = G.edges[e]["capacity"]
        G.edges[e]["capacity"] = 0.0
        val, _ = nx.maximum_flow(G, SRC, SNK, flow_func=nx.algorithms.flow.edmonds_karp)
        G.edges[e]["capacity"] = cap
        rows.append({"node_id": r.node_id, "flow_loss_pct": max(0.0, 1 - val / base) if base else 0.0})
    # At baseline the binding cut is simply demand (spare capacity everywhere). Stress-test demand to find
    # where the network breaks first: the surge headroom, and the min-cut at the breaking point.
    stress = []
    total = sum(dc_demand.values())
    for mult in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0):
        Gs = capacity_graph(nodes, edges, {k: v * mult for k, v in dc_demand.items()})
        val, _ = nx.maximum_flow(Gs, SRC, SNK, flow_func=nx.algorithms.flow.edmonds_karp)
        stress.append({"demand_multiplier": mult, "demand": total * mult, "max_flow": val, "served_share": val / (total * mult)})
    stress = pd.DataFrame(stress)
    breaking = stress.loc[stress["served_share"] < 0.999, "demand_multiplier"]
    mult = float(breaking.iloc[0]) if len(breaking) else 3.0
    Gs = capacity_graph(nodes, edges, {k: v * mult for k, v in dc_demand.items()})
    _, (S, T) = nx.minimum_cut(Gs, SRC, SNK, flow_func=nx.algorithms.flow.edmonds_karp)
    cut_edges = [(u, v, Gs.edges[u, v]["capacity"], mult) for u in S for v in Gs.successors(u) if v in T]
    return pd.DataFrame(rows), base, cut_edges, stress


# --------------------------------------------------------------------------- #
# Dependency lens
# --------------------------------------------------------------------------- #
def dependency_lens(d: dict[str, pd.DataFrame], weekly_rev: pd.DataFrame) -> pd.DataFrame:
    """Weekly revenue that loses *all* supply if the node fails."""
    rev_sku = weekly_rev.groupby("sku_id")["weekly_revenue"].sum()
    sku_components = d["bom"].groupby("sku_id")["component"].apply(set)
    comp_suppliers = d["comp_src"].groupby("component")["supplier_id"].apply(set)
    mat_suppliers = d["mat_src"].groupby("material")["supplier_id"].apply(set)
    from src.network import COMPONENT_MATERIALS
    rows = []
    # Tier-2 / tier-3 suppliers
    for sup in pd.concat([d["comp_src"]["supplier_id"], d["mat_src"]["supplier_id"]]).unique():
        at_risk = set()
        for sku, comps in sku_components.items():
            for c in comps:
                if comp_suppliers.get(c) == {sup}:
                    at_risk.add(sku)
                for m in COMPONENT_MATERIALS.get(c, []):
                    if mat_suppliers.get(m) == {sup}:
                        at_risk.add(sku)
        rows.append({"node_id": sup, "skus_at_risk": len(at_risk),
                     "revenue_at_risk_weekly": float(rev_sku.reindex(list(at_risk)).fillna(0).sum()), "extra_transit_days": 0.0})
    # Plants
    plants_per_sku = d["sku_plants"].groupby("sku_id")["plant_id"].apply(set)
    for p in d["sku_plants"]["plant_id"].unique():
        at_risk = [s for s, ps in plants_per_sku.items() if ps == {p}]
        rows.append({"node_id": p, "skus_at_risk": len(at_risk),
                     "revenue_at_risk_weekly": float(rev_sku.reindex(at_risk).fillna(0).sum()), "extra_transit_days": 0.0})
    # Logistics nodes: re-route each sourcing lane without the node
    G0 = fg_graph(d["nodes"], d["edges"])
    plants, dcs = sorted(d["sourcing"]["plant_id"].unique()), sorted(d["sourcing"]["dc_id"].unique())
    base = all_routes(G0, plants, dcs).set_index(["plant_id", "dc_id"])
    lane_rev = weekly_rev.merge(d["sourcing"][["sku_id", "dc_id", "plant_id"]], on=["sku_id", "dc_id"]).groupby(
        ["plant_id", "dc_id"])["weekly_revenue"].sum()
    logistics = d["nodes"][d["nodes"]["node_type"].isin(["port", "chokepoint"])]["node_id"]
    for n in logistics:
        alt = all_routes(fg_graph(d["nodes"], d["edges"], NetworkState(closed={n})), plants, dcs).set_index(["plant_id", "dc_id"])
        lost, extra, w = 0.0, 0.0, 0.0
        for key, rev in lane_rev.items():
            if not base.loc[key, "reachable"]:
                continue
            if not alt.loc[key, "reachable"]:
                lost += rev
            else:
                delta = alt.loc[key, "transit_days"] - base.loc[key, "transit_days"]
                extra += delta * rev
                w += rev
        rows.append({"node_id": n, "skus_at_risk": np.nan, "revenue_at_risk_weekly": lost,
                     "extra_transit_days": extra / w if w else 0.0})
    return pd.DataFrame(rows)


def time_lens(d: dict[str, pd.DataFrame], dep: pd.DataFrame) -> pd.DataFrame:
    """Time-to-survive: revenue-weighted days of cover (on-hand + pipeline) of the SKU×DCs a node feeds."""
    h = d["health"].copy()
    h["rev"] = h["avg_daily_demand_28d"].fillna(0) * h["unit_price"]
    tts_network = np.average(h["days_of_cover_incl_pipeline"].fillna(0).clip(upper=120), weights=h["rev"] + 1e-9)
    tts_by_plant = h.groupby("plant_id").apply(
        lambda g: np.average(g["days_of_cover_incl_pipeline"].fillna(0).clip(upper=120), weights=g["rev"] + 1e-9),
        include_groups=False)
    rows = []
    for r in d["nodes"].itertuples():
        if r.node_type == "plant":
            tts = tts_by_plant.get(r.node_id, tts_network)
        elif r.node_type in ("supplier_t2", "supplier_t3"):
            tts = tts_network + 21 + (7 if r.node_type == "supplier_t3" else 0)  # + plant component buffer
        else:
            tts = tts_network
        rows.append({"node_id": r.node_id, "tts_days": float(tts)})
    return pd.DataFrame(rows)


def concentration(d) -> pd.DataFrame:
    nodes = d["nodes"].set_index("node_id")
    rows = []
    for comp, g in d["comp_src"].groupby("component"):
        regions = nodes.loc[g["supplier_id"], "region"].value_counts(normalize=True)
        countries = nodes.loc[g["supplier_id"], "country"].value_counts(normalize=True)
        rows.append({"item": comp, "tier": 2, "n_suppliers": len(g), "hhi_country": float((countries ** 2).sum()),
                     "top_country": countries.index[0], "top_region": regions.index[0]})
    for mat, g in d["mat_src"].groupby("material"):
        countries = nodes.loc[g["supplier_id"], "country"].value_counts(normalize=True)
        regions = nodes.loc[g["supplier_id"], "region"].value_counts(normalize=True)
        rows.append({"item": mat, "tier": 3, "n_suppliers": len(g), "hhi_country": float((countries ** 2).sum()),
                     "top_country": countries.index[0], "top_region": regions.index[0]})
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    t0 = time.perf_counter()
    d = load_inputs()
    weekly_rev = d["fc"].merge(d["skus"], on="sku_id")
    weekly_rev["weekly_revenue"] = weekly_rev["weekly_demand"] * weekly_rev["unit_price"]
    dc_demand = d["fc"].groupby("dc_id")["weekly_demand"].sum().to_dict()

    flow, base_flow, cut, stress = flow_lens(d["nodes"], d["edges"], dc_demand)
    dep = dependency_lens(d, weekly_rev)
    tl = time_lens(d, dep)
    G = fg_graph(d["nodes"], d["edges"])
    btw = nx.betweenness_centrality(G, weight="days", normalized=True)

    failable = d["nodes"][d["nodes"]["node_type"].isin(FAILABLE)]
    m = (failable[["node_id", "name", "node_type", "tier", "region", "country", "lat", "lon", "ttr_days", "risk_score",
                   "capacity_per_week"]]
         .merge(flow, on="node_id", how="left").merge(dep, on="node_id", how="left").merge(tl, on="node_id", how="left"))
    m["betweenness"] = m["node_id"].map(btw).fillna(0.0)
    m["revenue_at_risk_weekly"] = m["revenue_at_risk_weekly"].fillna(0.0)
    m["extra_transit_days"] = m["extra_transit_days"].fillna(0.0)
    # Risk exposure: revenue lost while the node is down and inventory has run out.
    m["uncovered_days"] = (m["ttr_days"] - m["tts_days"]).clip(lower=0)
    m["risk_exposure_usd"] = m["revenue_at_risk_weekly"] / 7 * m["uncovered_days"]
    total_weekly = weekly_rev["weekly_revenue"].sum()
    m["revenue_at_risk_share"] = m["revenue_at_risk_weekly"] / total_weekly
    # Composite criticality (0-100) on *magnitudes*, not ranks: a rank-based blend buried the single-source
    # foundry (92% of revenue exposed) below ports, because its flow loss and centrality are zero.
    def scaled(s):
        s = s.fillna(0).clip(lower=0)
        return s / s.max() if s.max() > 0 else s
    m["criticality_score"] = (100 * (0.20 * scaled(m["flow_loss_pct"]) + 0.30 * scaled(m["revenue_at_risk_share"])
                                     + 0.35 * scaled(np.log1p(m["risk_exposure_usd"])) + 0.15 * scaled(m["betweenness"]))).round(1)
    m["is_single_point_of_failure"] = (m["revenue_at_risk_share"] > 0.05) | (m["flow_loss_pct"].fillna(0) > 0.10)
    m = m.sort_values("criticality_score", ascending=False)

    write_df(m, "ml.graph_node_metrics")
    write_df(pd.DataFrame([{"from_node": u.replace("_out", "").replace("_in", ""), "to_node": v.replace("_in", "").replace("_out", ""),
                            "capacity": c, "is_node_capacity": u.endswith("_in"), "at_demand_multiplier": mlt}
                           for u, v, c, mlt in cut]), "ml.graph_min_cut")
    write_df(stress, "ml.graph_stress_test")
    write_df(concentration(d), "ml.supplier_concentration")
    log.info("Graph analytics: base flow %.0f units/wk, %d SPOFs, min-cut %d arcs (%.1fs)", base_flow,
             int(m["is_single_point_of_failure"].sum()), len(cut), time.perf_counter() - t0)
    return m


if __name__ == "__main__":
    run()
