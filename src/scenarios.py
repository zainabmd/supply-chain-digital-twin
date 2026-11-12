"""
Disruption scenario engine: the "what-if" core of the digital twin.

A scenario is a named set of affected nodes plus four dials (the same dials as
the dashboard sliders):

    severity              0-100% capacity lost at affected nodes (≥95% = closed)
    duration_weeks        how long the disruption lasts inside the 8-week outlook
    lead_time_multiplier  global lead-time sensitivity (congestion, carrier slack)
    demand_shock          ± change in demand (panic buying, promotions, recession)

`evaluate()` turns that into:
  1. re-routed lanes (graph shortest paths with nodes closed or congested);
  2. supply availability per SKU through the bill of materials (alternate
     qualified suppliers can flex up 25%);
  3. a network flow plan (SciPy `linprog`, HiGHS): cost-to-serve, node
     utilisation and unmet demand;
  4. per-SKU×DC lead times, fulfilment fractions and pipeline delays, which feed
     the Monte Carlo stockout-risk model in `risk.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from src.generate import PRODUCTION_DAYS
from src.network import COMPONENT_MATERIALS
from src.routing import NetworkState, all_routes, fg_graph

def as_is_closures() -> set[str]:
    """Closures still active at the end of history (e.g. the Red Sea crisis). The baseline is the network
    *as it is today*, not an idealised open network, otherwise scenarios double-count ongoing detours."""
    from src.config import HISTORY_END
    from src.generate import EVENTS
    return {n for e in EVENTS if e.kind == "closure" and e.end >= HISTORY_END for n in e.nodes}


SCENARIOS: dict[str, dict] = {
    "Baseline (no disruption)": {"nodes": [], "delay": 0, "icon": "🟢",
                                 "description": "Network as it is today (Red Sea still diverted via the Cape), forecast demand, "
                                                "learned lead times."},
    "Red Sea reopens (upside)": {"nodes": [], "reopen": True, "delay": 0, "icon": "🚢",
                                 "description": "Carriers return to Suez: Asia–Europe and Asia–US East lanes shorten by ~10 days."},
    "Panama Canal drought": {"nodes": ["CP_PANAMA"], "delay": 7, "icon": "🌵",
                             "description": "Draft limits and transit slots cut canal capacity."},
    "Taiwan Strait blockade": {"nodes": ["CP_TAIWAN", "T2_HSINCHU_FAB", "T2_KAOHSIUNG_RF", "T2_TAICHUNG_OPTICS",
                                         "T3_HSINCHU_WAFER"], "delay": 5, "icon": "⚓",
                               "description": "Shipping through the strait halts and Taiwanese suppliers cannot export."},
    "Strait of Hormuz closure": {"nodes": ["CP_HORMUZ"], "delay": 3, "icon": "🛢️",
                                 "description": "Gulf access cut: Jebel Ali / Dubai DC isolated by sea."},
    "Shanghai port lockdown": {"nodes": ["CNSHA"], "delay": 6, "icon": "🔒",
                               "description": "The world's busiest container port stops handling exports."},
    "US West Coast port strike": {"nodes": ["USLAX"], "delay": 8, "icon": "✊",
                                  "description": "LA/Long Beach terminals stop; transpacific volume must reroute."},
    "Hsinchu foundry outage": {"nodes": ["T2_HSINCHU_FAB"], "delay": 0, "icon": "🔥",
                               "description": "The single-source SoC / Power-IC fab loses output."},
    "Regional supplier outage": {"nodes": "__REGION__", "delay": 0, "icon": "🌏",
                                 "description": "All tier-2/3 suppliers in the selected supplier region are impaired."},
    "Demand surge": {"nodes": [], "delay": 0, "icon": "📈", "default_shock": 0.30,
                     "description": "Network-wide demand spike (use the demand-shock slider)."},
    "Custom node outage": {"nodes": "__CUSTOM__", "delay": 5, "icon": "🎯", "hidden": True,
                           "description": "Any single node taken out (drill-down from the Network Twin)."},
}


@dataclass(frozen=True)
class ScenarioParams:
    scenario: str = "Baseline (no disruption)"
    severity: float = 1.0
    duration_weeks: int = 6
    lead_time_multiplier: float = 1.0
    demand_shock: float = 0.0
    supplier_region: str = "Asia Pacific"
    custom_nodes: tuple[str, ...] = ()


@dataclass
class ScenarioResult:
    params: ScenarioParams
    affected_nodes: list[str]
    series: pd.DataFrame                  # per SKU x DC: lead times, fulfilment, pipeline delay, route
    lanes: pd.DataFrame                   # per plant x DC: baseline vs scenario route
    flows: pd.DataFrame                   # LP edge flows
    node_util: pd.DataFrame               # LP node utilisation
    unmet: pd.DataFrame                   # LP unmet demand per DC
    lp_cost: float
    lp_status: str
    kpis: dict = field(default_factory=dict)
    closed_nodes: set = field(default_factory=set)        # everything closed: as-is disruptions + scenario


def affected(params: ScenarioParams, nodes: pd.DataFrame) -> list[str]:
    spec = SCENARIOS.get(params.scenario, SCENARIOS["Baseline (no disruption)"])
    if spec["nodes"] == "__CUSTOM__":
        return list(params.custom_nodes)
    if spec["nodes"] == "__REGION__":
        sup = nodes[nodes["node_type"].isin(["supplier_t2", "supplier_t3"]) & (nodes["region"] == params.supplier_region)]
        return sup["node_id"].tolist()
    return list(spec["nodes"])


def network_state(params: ScenarioParams, nodes: pd.DataFrame) -> NetworkState:
    spec = SCENARIOS.get(params.scenario, {})
    closed = set() if spec.get("reopen") else as_is_closures()
    st = NetworkState(closed=closed, lead_time_multiplier=params.lead_time_multiplier)
    ntype = nodes.set_index("node_id")["node_type"]
    for n in affected(params, nodes):
        if ntype.get(n) in ("supplier_t2", "supplier_t3"):
            continue  # suppliers act through the BOM, not routing
        if params.severity >= 0.95:
            st.closed.add(n)
        else:
            st.capacity_factor[n] = 1 - params.severity
            st.extra_delay[n] = spec.get("delay", 0) + 10 * params.severity  # congestion grows with severity
    return st


def supply_availability(params: ScenarioParams, d: dict[str, pd.DataFrame]) -> pd.Series:
    """Share of each SKU's normal output still producible, via component and material sourcing."""
    hit = set(affected(params, d["nodes"]))
    cap = d["nodes"].set_index("node_id")["capacity_per_week"]

    def avail(links: pd.DataFrame, key: str) -> pd.Series:
        """Share of each item's requirement still coverable.

        A supplier serving several items splits its capacity across them; capacities are sized at
        1.5x requirement, so requirement = allocated capacity / 1.5. Unaffected alternates can flex +25%.
        """
        n_items = links.groupby("supplier_id")[key].transform("count")
        links = links.assign(alloc=links["supplier_id"].map(cap).fillna(0) / n_items)
        out = {}
        for item, g in links.groupby(key):
            required = g["alloc"].sum() / 1.5
            ok = sum(a * (1.25 if s not in hit else (1 - params.severity)) for s, a in zip(g["supplier_id"], g["alloc"]))
            out[item] = float(min(1.0, ok / required)) if required > 0 else 1.0
        return pd.Series(out)

    comp_av = avail(d["comp_src"], "component")
    mat_av = avail(d["mat_src"], "material")
    comp_total = {c: min(comp_av.get(c, 1.0), *[mat_av.get(m, 1.0) for m in COMPONENT_MATERIALS.get(c, [])] or [1.0])
                  for c in comp_av.index}
    sku = d["bom"].assign(av=d["bom"]["component"].map(comp_total).fillna(1.0)).groupby("sku_id")["av"].min()
    # Plant-level impact (a plant in the affected set)
    plant_hit = d["sku_plants"][d["sku_plants"]["plant_id"].isin(hit)]
    for r in plant_hit.itertuples():
        sku[r.sku_id] = sku.get(r.sku_id, 1.0) * (1 - r.share * params.severity)
    return sku


def flow_plan(G, dc_demand: dict[str, float], plant_supply: dict[str, float], unmet_penalty: float = 50.0):
    """Min-cost multi-source flow on the finished-goods graph (SciPy linprog / HiGHS).

    Variables: one flow per edge + one 'unmet' slack per DC.
    Constraints: flow conservation, node throughput ≤ node capacity, plant output ≤ supply.
    """
    edges = list(G.edges(data=True))
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    ne, nd = len(edges), len(dc_demand)
    dcs = list(dc_demand)
    c = np.array([e[2]["cost"] for e in edges] + [unmet_penalty] * nd)
    A_eq = lil_matrix((len(nodes), ne + nd))
    b_eq = np.zeros(len(nodes))
    A_ub = lil_matrix((len(nodes), ne + nd))
    b_ub = np.zeros(len(nodes))
    for j, (u, v, _) in enumerate(edges):
        A_eq[idx[u], j] -= 1          # outflow
        A_eq[idx[v], j] += 1          # inflow
        A_ub[idx[v], j] += 1          # throughput (inflow) counts against capacity
    for k, dc in enumerate(dcs):
        if dc in idx:
            A_eq[idx[dc], ne + k] = 1  # unmet slack satisfies the DC's balance
    for n in nodes:
        t = G.nodes[n].get("node_type")
        if t == "dc":
            b_eq[idx[n]] = dc_demand.get(n, 0.0)
            b_ub[idx[n]] = 1e12
        elif t == "plant":
            # plants are sources: net outflow ≤ supply  ->  -(in - out) ≤ supply, and balance is inequality
            b_eq[idx[n]] = 0.0
            b_ub[idx[n]] = 1e12
        else:
            b_ub[idx[n]] = G.nodes[n].get("capacity", 1e12)
    # Plants: replace equality with "out - in ≤ supply" (source); drop their equality rows.
    plant_rows = [idx[n] for n in nodes if G.nodes[n].get("node_type") == "plant"]
    keep = [i for i in range(len(nodes)) if i not in set(plant_rows)]
    A_eq_k, b_eq_k = A_eq.tocsr()[keep], b_eq[keep]
    P = lil_matrix((len(plant_rows), ne + nd))
    for r, i in enumerate(plant_rows):
        P[r] = -A_eq.tocsr()[i]
    b_p = np.array([plant_supply.get(nodes[i], 0.0) for i in plant_rows])
    from scipy.sparse import vstack
    A_ub_all = vstack([A_ub.tocsr(), P.tocsr()])
    b_ub_all = np.concatenate([b_ub, b_p])
    res = linprog(c, A_ub=A_ub_all, b_ub=b_ub_all, A_eq=A_eq_k, b_eq=b_eq_k, bounds=(0, None), method="highs")
    if res.status != 0:
        return None, None, None, float("nan"), res.message
    x = res.x
    flows = pd.DataFrame([{"src": u, "dst": v, "flow": x[j], "mode": e.get("mode"), "days": e["days"], "cost_per_unit": e["cost"]}
                          for j, (u, v, e) in enumerate(edges) if x[j] > 1e-6])
    unmet = pd.DataFrame({"dc_id": dcs, "unmet_units": x[ne:], "demand_units": [dc_demand[d] for d in dcs]})
    thr = flows.groupby("dst")["flow"].sum()
    util = pd.DataFrame([{"node_id": n, "node_type": G.nodes[n].get("node_type"), "throughput": float(thr.get(n, 0.0)),
                          "capacity": G.nodes[n].get("capacity", np.nan)} for n in nodes])
    util["utilisation"] = util["throughput"] / util["capacity"].replace(0, np.nan)
    return flows, util, unmet, float(res.fun), "optimal"


def evaluate(params: ScenarioParams, d: dict[str, pd.DataFrame]) -> ScenarioResult:
    """d: nodes, edges, bom, comp_src, mat_src, sku_plants, sourcing, health, fc, skus."""
    hit = affected(params, d["nodes"])
    st = network_state(params, d["nodes"])
    G0 = fg_graph(d["nodes"], d["edges"], NetworkState(closed=as_is_closures()))   # as-is baseline
    G1 = fg_graph(d["nodes"], d["edges"], st)
    plants, dcs = sorted(d["sourcing"]["plant_id"].unique()), sorted(d["sourcing"]["dc_id"].unique())
    base = all_routes(G0, plants, dcs).set_index(["plant_id", "dc_id"])
    scn = all_routes(G1, plants, dcs).set_index(["plant_id", "dc_id"])
    lanes = base[["transit_days", "path", "chokepoints"]].add_prefix("base_").join(
        scn[["reachable", "transit_days", "path", "chokepoints"]].add_prefix("scn_")).reset_index()

    # Lane capacity factor: tightest affected node on the scenario path.
    def path_factor(path) -> float:
        if not isinstance(path, str):
            return 0.0
        return float(min([st.capacity_factor.get(n, 1.0) for n in path.split("→")] + [1.0]))
    lanes["lane_capacity_factor"] = [path_factor(p) if r else 0.0 for p, r in zip(lanes["scn_path"], lanes["scn_reachable"])]
    lanes["delta_days"] = lanes["scn_transit_days"] - lanes["base_transit_days"]

    sku_av = supply_availability(params, d)
    s = d["health"][["sku_id", "dc_id", "plant_id", "unit_price", "unit_cost", "learned_lead_time_days", "lt_mean_26w",
                     "lt_std_26w"]].merge(lanes, on=["plant_id", "dc_id"], how="left")
    s["sku_supply_factor"] = s["sku_id"].map(sku_av).fillna(1.0)
    s["reachable"] = s["scn_reachable"].fillna(False).astype(bool)
    # Lead time = route transit + production + the history-learned slack (queues, customs, consolidation).
    slack = (s["lt_mean_26w"] - (s["base_transit_days"] + PRODUCTION_DAYS)).clip(lower=0).fillna(0)
    s["lt_base"] = (s["base_transit_days"] + PRODUCTION_DAYS + slack)
    s["lt_scenario"] = np.where(s["reachable"], s["scn_transit_days"].fillna(s["base_transit_days"]) + PRODUCTION_DAYS + slack,
                                s["lt_base"] + 7 * params.duration_weeks)   # unreachable: wait out the disruption
    supplier_hit = s["sku_supply_factor"] < 0.999
    s.loc[supplier_hit, "lt_scenario"] += 10 * params.severity            # re-qualification / expediting from alternates
    s["fulfil_factor"] = np.where(s["reachable"], s["lane_capacity_factor"].fillna(1.0), 0.0) * s["sku_supply_factor"]
    # Goods already on the water on a now-affected route are delayed by the reroute difference (or the outage).
    s["pipeline_delay_days"] = np.where(s["reachable"], (s["lt_scenario"] - s["lt_base"]).clip(lower=0),
                                        7 * params.duration_weeks).round()
    s.loc[supplier_hit, "pipeline_delay_days"] = 0.0                       # finished goods in transit are unaffected

    # Network flow plan
    shock = 1 + params.demand_shock
    dc_demand = (d["fc"].groupby("dc_id")["weekly_demand"].sum() * shock).to_dict()
    plant_cap = d["nodes"].set_index("node_id")["capacity_per_week"]
    plant_factor = s.groupby("plant_id")["sku_supply_factor"].mean()
    plant_supply = {p: float(plant_cap.get(p, 0) * plant_factor.get(p, 1.0) * (0 if p in st.closed else 1)) for p in plants}
    flows, util, unmet, cost, status = flow_plan(G1, dc_demand, plant_supply)

    res = ScenarioResult(params, hit, s, lanes, flows if flows is not None else pd.DataFrame(),
                         util if util is not None else pd.DataFrame(), unmet if unmet is not None else pd.DataFrame(),
                         cost, status, closed_nodes=set(st.closed) | set(hit))
    served = 1 - unmet["unmet_units"].sum() / unmet["demand_units"].sum() if unmet is not None else float("nan")
    res.kpis = {
        "affected_nodes": len(hit),
        "lanes_rerouted": int(((lanes["base_path"] != lanes["scn_path"]) & lanes["scn_reachable"].fillna(False)).sum()),
        "lanes_unreachable": int((~lanes["scn_reachable"].fillna(False).astype(bool)).sum()),
        "avg_lead_time_base": float(s["lt_base"].mean()),
        "avg_lead_time_scenario": float(s["lt_scenario"].mean()),
        "network_served_share": float(served),
        # Real logistics cost only (the LP objective also contains the unmet-demand penalty, a modelling device).
        "cost_per_unit": float((flows["flow"] * flows["cost_per_unit"]).sum()
                               / max(1e-9, sum(dc_demand.values()) - unmet["unmet_units"].sum())) if flows is not None else float("nan"),
        "skus_supply_impaired": int((sku_av < 0.999).sum()),
    }
    return res
