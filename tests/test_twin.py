"""
Unit + integration tests for the supply-chain digital twin.   pytest -q

The warehouse integration test runs the real extract → load → DQ gate →
staging → marts chain into a temp DuckDB (~20 s). Analytics modules are tested
on the in-memory network so the suite stays fast.
"""
from __future__ import annotations

import sys
import types

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.network import build_network, haversine_km
from src.routing import NetworkState, all_routes, fg_graph, route


@pytest.fixture(scope="session")
def net():
    return build_network()


# ------------------------------------------------------------------ network
def test_network_shape_and_integrity(net):
    assert net.skus["sku_id"].is_unique and len(net.skus) == 60
    ids = set(net.nodes["node_id"])
    assert set(net.edges["src"]) <= ids and set(net.edges["dst"]) <= ids
    assert (net.edges["capacity_per_week"] > 0).all() and (net.edges["transit_days"] > 0).all()
    # Every BOM component has at least one qualified supplier
    assert set(net.bom["component"]) <= set(net.component_sources["component"])


def test_known_single_source_structure(net):
    counts = net.component_sources.groupby("component")["supplier_id"].nunique()
    assert counts["SoC"] == 1  # the deliberate SPOF the resilience lens must find


def test_haversine():
    assert 9_000 < haversine_km(51.95, 4.05, 1.26, 103.84) < 11_000  # Rotterdam–Singapore great circle ≈ 10.5k km


# ------------------------------------------------------------------ routing
def test_every_dc_reachable_at_baseline(net):
    G = fg_graph(net.nodes, net.edges)
    plants = sorted(net.sku_plants["plant_id"].unique())
    dcs = net.nodes.loc[net.nodes["node_type"] == "dc", "node_id"]
    r = all_routes(G, plants, dcs)
    assert r.groupby("dc_id")["reachable"].any().all()


def test_suez_closure_reroutes_via_cape(net):
    base = route(fg_graph(net.nodes, net.edges), "PL_SHENZHEN", "DC_ROTTERDAM")
    closed = route(fg_graph(net.nodes, net.edges, NetworkState(closed={"CP_SUEZ", "CP_BAB_EL_MANDEB"})), "PL_SHENZHEN", "DC_ROTTERDAM")
    assert "CP_SUEZ" in base["chokepoints"]
    assert "CP_CAPE" in closed["chokepoints"] and "CP_SUEZ" not in closed["chokepoints"]
    assert closed["transit_days"] > base["transit_days"] + 5


def test_delay_and_lead_time_multiplier(net):
    g1 = fg_graph(net.nodes, net.edges, NetworkState(extra_delay={"USLAX": 6}))
    g2 = fg_graph(net.nodes, net.edges, NetworkState(lead_time_multiplier=1.5))
    base = route(fg_graph(net.nodes, net.edges), "PL_SHENZHEN", "DC_INLAND_EMPIRE")["transit_days"]
    assert route(g2, "PL_SHENZHEN", "DC_INLAND_EMPIRE")["transit_days"] == pytest.approx(base * 1.5, rel=1e-6)
    assert route(g1, "PL_SHENZHEN", "DC_INLAND_EMPIRE")["transit_days"] >= base


# ------------------------------------------------------------------ graph analytics
def test_max_flow_drops_when_plant_removed(net):
    from src.graph_analytics import capacity_graph, SRC, SNK
    dem = {d: 5000.0 for d in net.nodes.loc[net.nodes["node_type"] == "dc", "node_id"]}
    G = capacity_graph(net.nodes, net.edges, dem)
    base, _ = nx.maximum_flow(G, SRC, SNK, flow_func=nx.algorithms.flow.edmonds_karp)
    G.edges["PL_SHENZHEN_in", "PL_SHENZHEN_out"]["capacity"] = 0
    after, _ = nx.maximum_flow(G, SRC, SNK, flow_func=nx.algorithms.flow.edmonds_karp)
    assert after <= base


# ------------------------------------------------------------------ scenarios / flow LP
def test_flow_lp_meets_demand_and_respects_capacity(net):
    from src.scenarios import flow_plan
    G = fg_graph(net.nodes, net.edges)
    dcs = net.nodes.loc[net.nodes["node_type"] == "dc", "node_id"]
    dem = {d: 1000.0 for d in dcs}
    supply = {p: 5000.0 for p in net.sku_plants["plant_id"].unique()}
    flows, util, unmet, cost, status = flow_plan(G, dem, supply)
    assert status == "optimal" and unmet["unmet_units"].sum() == pytest.approx(0, abs=1e-6)
    cap = util.dropna(subset=["capacity"])
    assert (cap["throughput"] <= cap["capacity"] * (1 + 1e-6)).all()


def test_flow_lp_reports_unmet_when_supply_short(net):
    from src.scenarios import flow_plan
    G = fg_graph(net.nodes, net.edges)
    dcs = net.nodes.loc[net.nodes["node_type"] == "dc", "node_id"]
    _, _, unmet, _, status = flow_plan(G, {d: 1000.0 for d in dcs}, {p: 100.0 for p in net.sku_plants["plant_id"].unique()})
    assert status == "optimal" and unmet["unmet_units"].sum() > 0


def test_supplier_outage_impairs_soc_skus(net):
    from src.scenarios import ScenarioParams, supply_availability
    d = {"nodes": net.nodes, "comp_src": net.component_sources, "mat_src": net.material_sources, "bom": net.bom,
         "sku_plants": net.sku_plants}
    av = supply_availability(ScenarioParams(scenario="Hsinchu foundry outage", severity=1.0), d)
    soc_skus = set(net.bom.loc[net.bom["component"] == "SoC", "sku_id"])
    assert (av[list(soc_skus)] == 0).all()
    # Power-IC SKUs keep most supply via the qualified alternate fab (Dresden flexes up), but not all of it.
    others = av.drop(list(soc_skus))
    assert (others > 0.85).all() and (others < 1.0).all()


# ------------------------------------------------------------------ risk engine
def _toy_series(lt_scn=20.0, fulfil=1.0, oh=200.0):
    s = pd.DataFrame({"sku_id": ["A", "B"], "dc_id": ["D", "D"], "on_hand_units": [oh, oh], "unit_price": [100.0, 100.0],
                      "lt_base": [10.0, 10.0], "lt_scenario": [lt_scn, lt_scn], "fulfil_factor": [fulfil, fulfil],
                      "pipeline_delay_days": [0.0, 0.0]})
    fc = pd.DataFrame([{"sku_id": k, "dc_id": "D", "horizon": h, "p10": 60, "p50": 70, "p90": 85} for k in "AB" for h in range(1, 9)])
    return s, fc


def test_risk_worse_under_disruption():
    from src.risk import simulate
    s, fc = _toy_series(lt_scn=10.0)
    good = simulate(s, fc, pd.DataFrame(columns=["sku_id", "dc_id", "days_until_arrival", "qty_units"]), sims=200)
    s2, _ = _toy_series(lt_scn=40.0, fulfil=0.3)
    bad = simulate(s2, fc, pd.DataFrame(columns=["sku_id", "dc_id", "days_until_arrival", "qty_units"]), sims=200, duration_days=56)
    assert bad.summary["expected_fill_rate"].mean() < good.summary["expected_fill_rate"].mean()
    assert bad.summary["p_stockout"].mean() >= good.summary["p_stockout"].mean()
    assert bad.traj_p50.shape == (2, 56)


# ------------------------------------------------------------------ optimisation
def test_loss_function_matches_definition():
    from src.optimize import loss
    z = np.linspace(-1, 3, 9)
    assert np.allclose(loss(z), norm.pdf(z) - z * (1 - norm.cdf(z)))


def test_slsqp_respects_space_and_beats_uniform():
    from src.optimize import evaluate_policy, optimise_dc
    rng = np.random.default_rng(0)
    n = 25
    g = pd.DataFrame({"sku_id": [f"S{i}" for i in range(n)], "dc_id": "D", "category": "X",
                      "unit_cost": rng.uniform(10, 500, n), "unit_price": rng.uniform(600, 900, n),
                      "unit_volume_m3": rng.uniform(0.001, 0.01, n), "mu": rng.uniform(1, 50, n), "sigma": rng.uniform(5, 60, n),
                      "L": 20.0, "annual_demand": 0.0})
    g["annual_demand"] = g["mu"] * 365
    g["pi"] = 0.35 * (g["unit_price"] - g["unit_cost"]) + 0.05 * g["unit_price"]
    cur = evaluate_policy(g, np.full(n, 1.65))
    g["storage_m3"] = cur["volume_m3"].sum() * 1.1
    z, status = optimise_dc(g)
    opt = evaluate_policy(g, z)
    assert status == "optimal"
    assert opt["volume_m3"].sum() <= g["storage_m3"].iloc[0] * 1.001
    assert opt["total_cost"].sum() <= cur["total_cost"].sum() + 1e-6


# ------------------------------------------------------------------ pipeline + DAG
def test_task_graph_is_acyclic_and_complete():
    from graphlib import TopologicalSorter
    from src.pipeline import TASKS, downstream
    order = list(TopologicalSorter({k: set(v.deps) for k, v in TASKS.items()}).static_order())
    assert set(order) == set(TASKS)
    assert "publish_control_tower" in downstream("train_and_forecast")


def test_airflow_dag_mirrors_registry():
    """Import the DAG file against a minimal Airflow stub and check every dependency is wired."""
    edges = set()

    class _Op:
        def __init__(self, task_id, **kw):
            self.task_id = task_id

        def __rshift__(self, other):
            edges.add((self.task_id, other.task_id))
            return other

    class _DAG:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    stub = types.ModuleType("airflow")
    stub.DAG = _DAG
    ops = types.ModuleType("airflow.operators")
    empty = types.ModuleType("airflow.operators.empty")
    empty.EmptyOperator = _Op
    py = types.ModuleType("airflow.operators.python")
    py.PythonOperator = _Op
    sys.modules.update({"airflow": stub, "airflow.operators": ops, "airflow.operators.empty": empty, "airflow.operators.python": py})
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("dag", Path(__file__).resolve().parents[1] / "dags" / "supply_chain_digital_twin.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from src.pipeline import TASKS
    for name, t in TASKS.items():
        for d in t.deps:
            assert (d, name) in edges
    assert ("start", "extract_operational_data") in edges and ("publish_control_tower", "done") in edges


@pytest.mark.slow
def test_warehouse_integration_extract_to_marts():
    from src import generate, quality, warehouse
    counts = generate.run()
    assert counts["demand_daily"] == 60 * 12 * 728
    warehouse.load_raw()
    res = quality.run_checks(fail_on_error=True)
    assert not (res["status"] == "fail").any()
    warehouse.run_sql_layer("staging")
    warehouse.run_sql_layer("marts")
    k = warehouse.read_sql("SELECT sum(sales_units)/sum(demand_units) AS fill FROM marts.mart_weekly_sku_dc").iloc[0, 0]
    assert 0.85 < k < 1.0
    ev = warehouse.read_sql("SELECT * FROM marts.mart_disruption_impact WHERE event_id = 'EV05'")
    assert ev["avg_delay_days"].iloc[0] > 2   # Red Sea diversion must show up as real delay
