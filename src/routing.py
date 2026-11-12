"""
Finished-goods routing on the network graph.

One routing engine is shared by the history simulator, the graph analytics
and the scenario simulator, so a disrupted chokepoint produces the *same*
reroute everywhere. Routes are least-cost paths with transit time as a
tiebreaker, through plant → port → sea legs/chokepoints → port → DC
(or direct road for regional plants).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

FG_NODE_TYPES = {"plant", "port", "chokepoint", "waypoint", "dc"}
TIME_VALUE_PER_DAY = 0.012  # USD per unit-day in transit (capital + obsolescence), adds time into the routing cost


@dataclass
class NetworkState:
    """A (possibly disrupted) state of the network."""

    closed: set[str] = field(default_factory=set)                 # nodes unusable
    extra_delay: dict[str, float] = field(default_factory=dict)   # node -> added days when traversed
    capacity_factor: dict[str, float] = field(default_factory=dict)  # node -> remaining capacity share
    lead_time_multiplier: float = 1.0                             # global sensitivity (congestion, buffers)


def fg_graph(nodes: pd.DataFrame, edges: pd.DataFrame, state: NetworkState | None = None) -> nx.DiGraph:
    """Directed finished-goods graph with `days` and `cost` edge weights for a given network state."""
    state = state or NetworkState()
    fg_nodes = nodes[nodes["node_type"].isin(FG_NODE_TYPES)]
    G = nx.DiGraph()
    for r in fg_nodes.itertuples():
        if r.node_id in state.closed:
            continue
        G.add_node(r.node_id, node_type=r.node_type, capacity=r.capacity_per_week * state.capacity_factor.get(r.node_id, 1.0))
    for r in edges.itertuples():
        if r.src in G and r.dst in G:
            # Delay at a node is paid when entering it.
            days = (r.transit_days + state.extra_delay.get(r.dst, 0.0)) * state.lead_time_multiplier
            G.add_edge(r.src, r.dst, days=days, cost=r.cost_per_unit + days * TIME_VALUE_PER_DAY,
                       capacity=r.capacity_per_week, mode=r.mode, edge_type=r.edge_type)
    return G


def route(G: nx.DiGraph, plant: str, dc: str) -> dict | None:
    """Least generalised-cost route plant → DC, or None if the DC is unreachable."""
    if plant not in G or dc not in G:
        return None
    try:
        path = nx.shortest_path(G, plant, dc, weight="cost")
    except nx.NetworkXNoPath:
        return None
    days = sum(G[a][b]["days"] for a, b in zip(path, path[1:]))
    cost = sum(G[a][b]["cost"] for a, b in zip(path, path[1:]))
    chokepoints = [n for n in path if G.nodes[n].get("node_type") == "chokepoint"]
    return {"path": path, "transit_days": days, "route_cost": cost, "chokepoints": chokepoints,
            "is_sea": any(G[a][b]["mode"] == "sea" for a, b in zip(path, path[1:]))}


def all_routes(G: nx.DiGraph, plants, dcs) -> pd.DataFrame:
    rows = []
    for p in plants:
        if p not in G:
            continue
        # One Dijkstra per plant is far cheaper than one per (plant, dc) pair.
        dist, paths = nx.single_source_dijkstra(G, p, weight="cost")
        for d in dcs:
            if d not in paths:
                rows.append({"plant_id": p, "dc_id": d, "reachable": False})
                continue
            path = paths[d]
            days = sum(G[a][b]["days"] for a, b in zip(path, path[1:]))
            rows.append({"plant_id": p, "dc_id": d, "reachable": True, "transit_days": days, "route_cost": dist[d],
                         "path": "→".join(path),
                         "chokepoints": "|".join(n for n in path if G.nodes[n].get("node_type") == "chokepoint")})
    return pd.DataFrame(rows)
