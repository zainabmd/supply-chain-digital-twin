"""
Operational history simulator: the "extract" step that stands in for ERP, TMS and WMS exports.

Simulates 104 weeks (2023-01-02 → 2024-12-29) of:

* **Daily demand** for 60 SKUs × 12 DCs (720 series, ~525k rows). Includes trend,
  weekday and annual seasonality, Black Friday / Christmas / back-to-school peaks,
  a planned promotion calendar, product launches, and over-dispersed noise
  (gamma-Poisson).
* **Replenishment** under each DC's weekly order-up-to policy. Planners learn lead
  times from what actually arrives (an EMA), so they adapt to a disruption only
  after the late containers turn up, as in real life.
* **Shipments** routed through the network graph. A shipment's lead time is its
  route's transit time, plus production, plus delays from any disruption active
  when it departs. When a chokepoint is closed, the route is recomputed (Red Sea
  → Cape of Good Hope).
* **Inventory** positions per day: on-hand, on-order, sales and lost sales.
* **Disruption log** modelled on real 2023–24 events, plus supplier delivery
  performance.

Everything is vectorised across the 720 series. The only loop is over days.

    python -m src.generate
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import (BASELINE_SERVICE_Z, FORECAST_WEEKS, HISTORY_END, HISTORY_START, LANDING_DIR,
                        REVIEW_PERIOD_DAYS, SEED, get_logger)
from src.network import DCS, Network, build_network
from src.routing import NetworkState, all_routes, fg_graph

log = get_logger("generate")
PRODUCTION_DAYS = 5.0
LT_EMA_ALPHA = 0.35


@dataclass(frozen=True)
class Event:
    event_id: str
    name: str
    category: str
    kind: str                 # "delay" | "closure" | "supplier"
    nodes: tuple[str, ...]
    start: date
    end: date
    delay_days: float = 0.0
    capacity_factor: float = 1.0
    description: str = ""


# Modelled on real 2023-24 disruptions (dates and magnitudes simplified).
EVENTS = [
    Event("EV01", "Shanghai port congestion", "Port congestion", "delay", ("CNSHA",), date(2023, 3, 13), date(2023, 4, 23),
          5, 0.7, "Post-reopening backlog at the world's busiest container port."),
    Event("EV02", "US West Coast labour slowdown", "Labour action", "delay", ("USLAX",), date(2023, 6, 1), date(2023, 7, 14),
          6, 0.75, "ILWU contract dispute slows LA/Long Beach terminal throughput."),
    Event("EV03", "Panama Canal drought restrictions", "Climate", "delay", ("CP_PANAMA",), date(2023, 8, 1), date(2024, 2, 15),
          7, 0.6, "Low Gatun Lake levels cut daily transits and draft limits."),
    Event("EV04", "Typhoon Saola – Shenzhen", "Extreme weather", "delay", ("CNYTN", "PL_SHENZHEN"), date(2023, 9, 1),
          date(2023, 9, 10), 7, 0.3, "Port and plant shut for typhoon; recovery backlog."),
    Event("EV05", "Red Sea crisis – Bab-el-Mandeb unsafe", "Geopolitical", "closure", ("CP_BAB_EL_MANDEB",),
          date(2023, 12, 15), HISTORY_END, 0, 0.0, "Carriers divert Asia–Europe services around the Cape of Good Hope."),
    Event("EV06", "Fire at Hsinchu Foundry", "Supplier failure", "supplier", ("T2_HSINCHU_FAB",), date(2024, 3, 10),
          date(2024, 4, 30), 10, 0.5, "Single-source SoC/Power-IC supplier loses half its output for seven weeks."),
    Event("EV07", "Durban port strike", "Labour action", "delay", ("ZADUR",), date(2024, 8, 5), date(2024, 8, 20),
          8, 0.5, "Transnet strike halts container handling."),
    Event("EV08", "US East & Gulf Coast port strike", "Labour action", "delay", ("USNYC", "USSAV"), date(2024, 10, 1),
          date(2024, 10, 4), 6, 0.0, "Three-day ILA strike; one-week backlog to clear."),
]
SOC_COMPONENTS = {"SoC", "Power IC"}


def _days() -> pd.DatetimeIndex:
    return pd.date_range(HISTORY_START, HISTORY_END, freq="D")


def build_promo_calendar(net: Network, rng: np.random.Generator) -> pd.DataFrame:
    """Planned promotions (SKU × region × week), extending FORECAST_WEEKS into the future: a known covariate."""
    weeks = pd.date_range(HISTORY_START, pd.Timestamp(HISTORY_END) + pd.Timedelta(weeks=FORECAST_WEEKS), freq="W-MON")
    regions = sorted({v[3] for v in DCS.values()})
    rows = []
    for sku in net.skus["sku_id"]:
        for reg in regions:
            flags = rng.random(len(weeks)) < 0.06
            for w in weeks[flags]:
                rows.append({"sku_id": sku, "region": reg, "week_start": w.date(), "uplift": round(float(rng.uniform(1.25, 1.7)), 2)})
    return pd.DataFrame(rows)


def build_demand(net: Network, promos: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    days = _days()
    skus = net.skus.reset_index(drop=True)
    dcs = list(DCS)
    pairs = pd.MultiIndex.from_product([skus["sku_id"], dcs], names=["sku_id", "dc_id"]).to_frame(index=False)
    pairs = pairs.merge(skus, on="sku_id")
    pairs["dc_size"] = pairs["dc_id"].map({k: v[5] for k, v in DCS.items()})
    pairs["region"] = pairs["dc_id"].map({k: v[3] for k, v in DCS.items()})
    affinity = np.exp(rng.normal(0, 0.3, len(pairs)))

    t = np.arange(len(days))
    doy = days.day_of_year.to_numpy()
    weekday = np.array([1.10, 1.08, 1.05, 1.05, 1.12, 0.85, 0.75])[days.day_of_week]
    month, dom = days.month.to_numpy(), days.day.to_numpy()
    holiday = np.ones(len(days))
    holiday[(month == 11) & (dom >= 20)] = 1.35                              # Black Friday / Cyber week
    holiday[(month == 12) & (dom <= 22)] = 1.25                              # Christmas
    back_to_school = ((month == 8) | ((month == 9) & (dom <= 7))).astype(float)

    base = (pairs["base_daily_demand"].to_numpy() * pairs["dc_size"].to_numpy() * affinity)[:, None]
    trend = (1 + pairs["trend_per_year"].to_numpy()[:, None]) ** (t[None, :] / 365.0)
    amp = pairs["seasonality_amp"].to_numpy()[:, None]
    annual = 1 + amp * np.cos(2 * np.pi * (doy[None, :] - 350) / 365.0)
    bts = 1 + 0.25 * back_to_school[None, :] * (pairs["category"].to_numpy() == "Laptops")[:, None]
    launch_day = pairs["launch_week"].to_numpy()[:, None] * 7
    # 8-week adoption ramp for products launched during the window; established products start at full rate.
    ramp = np.where(launch_day > 0, np.clip((t[None, :] - launch_day) / 56.0, 0, 1), 1.0)

    # Promotions: map (sku, region, week) uplift onto each day.
    week_idx = ((days - pd.Timestamp(HISTORY_START)).days // 7).to_numpy()
    promo_mult = np.ones((len(pairs), len(days)))
    if not promos.empty:
        pidx = {(s, r): i for i, (s, r) in enumerate(zip(pairs["sku_id"], pairs["region"]))}
        rows_by_key: dict[tuple, list[int]] = {}
        for i, key in enumerate(zip(pairs["sku_id"], pairs["region"])):
            rows_by_key.setdefault(key, []).append(i)
        for p in promos.itertuples():
            w = (pd.Timestamp(p.week_start) - pd.Timestamp(HISTORY_START)).days // 7
            if 0 <= w <= week_idx.max():
                cols = np.flatnonzero(week_idx == w)
                for i in rows_by_key.get((p.sku_id, p.region), []):
                    promo_mult[i, cols] = p.uplift
        del pidx

    lam = base * trend * weekday[None, :] * holiday[None, :] * annual * bts * ramp * promo_mult
    # Gamma-Poisson (negative binomial) noise: variance > mean, like real order streams.
    shape = 3.0
    demand = rng.poisson(lam * rng.gamma(shape, 1 / shape, lam.shape)).astype(np.int32)

    long = pd.DataFrame({
        "date": np.tile(days.date, len(pairs)),
        "sku_id": np.repeat(pairs["sku_id"].to_numpy(), len(days)),
        "dc_id": np.repeat(pairs["dc_id"].to_numpy(), len(days)),
        "demand_units": demand.ravel(),
        "promo_flag": (promo_mult > 1).ravel(),
    })
    return long, demand, pairs[["sku_id", "dc_id", "category", "region", "unit_cost", "unit_price", "unit_volume_m3"]]


def _active(day: date, ev: Event) -> bool:
    return ev.start <= day <= ev.end


def simulate_supply(net: Network, demand: np.ndarray, pairs: pd.DataFrame, rng: np.random.Generator):
    days = _days()
    n_series, n_days = demand.shape
    plants = sorted(net.sku_plants["plant_id"].unique())
    dcs = list(DCS)

    # ---- Sourcing: the plant with the cheapest baseline route for each SKU x DC ----
    base_routes = all_routes(fg_graph(net.nodes, net.edges), plants, dcs).set_index(["plant_id", "dc_id"])
    cand = net.sku_plants.merge(pd.DataFrame({"dc_id": dcs}), how="cross")
    cand["route_cost"] = [base_routes.loc[(p, d), "route_cost"] if base_routes.loc[(p, d), "reachable"] else np.inf
                          for p, d in zip(cand["plant_id"], cand["dc_id"])]
    cand["score"] = cand["route_cost"] - cand["share"] * 0.5      # prefer the primary plant on near-ties
    sourcing = cand.sort_values("score").groupby(["sku_id", "dc_id"], as_index=False).first()
    sourcing = pairs[["sku_id", "dc_id"]].merge(sourcing[["sku_id", "dc_id", "plant_id"]], on=["sku_id", "dc_id"], how="left")
    if sourcing["plant_id"].isna().any():
        raise RuntimeError("Some SKU x DC pairs have no reachable source plant")
    plant_idx = sourcing["plant_id"].map({p: i for i, p in enumerate(plants)}).to_numpy()
    dc_idx = sourcing["dc_id"].map({d: i for i, d in enumerate(dcs)}).to_numpy()
    uses_soc = pairs["sku_id"].isin(net.bom.loc[net.bom["component"].isin(SOC_COMPONENTS), "sku_id"]).to_numpy()

    # ---- Route tables per network state (cached by active-event set) ----
    state_cache: dict[frozenset, tuple[np.ndarray, np.ndarray]] = {}

    def routes_for(active: frozenset) -> tuple[np.ndarray, np.ndarray]:
        if active not in state_cache:
            st = NetworkState()
            for ev in EVENTS:
                if ev.event_id in active:
                    for n in ev.nodes:
                        if ev.kind == "closure":
                            st.closed.add(n)
                        elif ev.kind == "delay":
                            st.extra_delay[n] = st.extra_delay.get(n, 0) + ev.delay_days
            rt = all_routes(fg_graph(net.nodes, net.edges, st), plants, dcs).set_index(["plant_id", "dc_id"])
            days_m = np.full((len(plants), len(dcs)), 60.0)
            cps = np.empty((len(plants), len(dcs)), dtype=object)
            for (p, d), r in rt.iterrows():
                if r["reachable"]:
                    days_m[plants.index(p), dcs.index(d)] = r["transit_days"]
                    cps[plants.index(p), dcs.index(d)] = r["chokepoints"] + "#" + r["path"]
            state_cache[active] = (days_m, cps)
        return state_cache[active]

    # ---- Policy state ----
    baseline_days, _ = routes_for(frozenset())
    lt_plan = baseline_days[plant_idx, dc_idx] + PRODUCTION_DAYS
    lt_est = lt_plan.copy()
    mu0 = demand[:, :28].mean(axis=1) + 0.5
    sd0 = demand[:, :28].std(axis=1) + 0.5
    horizon = n_days + 160
    arrivals = np.zeros((n_series, horizon), dtype=np.float64)
    # Opening position: two weeks of cover plus safety stock (a steady-state network, not a cold start).
    on_hand = np.round(mu0 * 14 + BASELINE_SERVICE_Z * np.sqrt((lt_plan + 7) * sd0 ** 2 + (mu0 * 0.15 * lt_plan) ** 2))
    for k in range(0, int(lt_plan.max()) + 7, 7):                     # warm pipeline already on the water
        mask = k < lt_plan
        arrivals[mask, k] += np.round(mu0[mask] * 7)

    arrival_learn: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    out_on_hand = np.zeros((n_series, n_days), dtype=np.float32)
    out_on_order = np.zeros((n_series, n_days), dtype=np.float32)
    out_sales = np.zeros((n_series, n_days), dtype=np.float32)
    out_lost = np.zeros((n_series, n_days), dtype=np.float32)
    ship_rows: list[pd.DataFrame] = []
    cum_arrivals = np.cumsum(arrivals, axis=1)  # recomputed lazily below for on-order

    for t, day in enumerate(days.date):
        on_hand += arrivals[:, t]
        if t in arrival_learn:  # planners update lead-time beliefs from what actually arrived
            for idx, lt in arrival_learn.pop(t):
                lt_est[idx] = (1 - LT_EMA_ALPHA) * lt_est[idx] + LT_EMA_ALPHA * lt
        d = demand[:, t]
        sales = np.minimum(d, on_hand)
        on_hand -= sales
        out_sales[:, t], out_lost[:, t], out_on_hand[:, t] = sales, d - sales, on_hand

        if t % REVIEW_PERIOD_DAYS == 0:
            window = demand[:, max(0, t - 28):t] if t >= 7 else demand[:, :28]
            mu, sd = window.mean(axis=1), window.std(axis=1) + 0.5
            cover = lt_est + REVIEW_PERIOD_DAYS
            # Safety stock covers demand AND lead-time uncertainty (planners assume a 15% LT CV).
            sigma = np.sqrt(cover * sd ** 2 + (mu * 0.15 * lt_est) ** 2)
            target = mu * cover + BASELINE_SERVICE_Z * sigma
            on_order = arrivals[:, t + 1:].sum(axis=1)
            qty = np.maximum(0, np.ceil((target - on_hand - on_order) / 10) * 10)
            idx = np.flatnonzero(qty > 0)
            if len(idx):
                active = frozenset(ev.event_id for ev in EVENTS if _active(day, ev) and ev.kind in ("delay", "closure"))
                days_m, cps = routes_for(active)
                transit = days_m[plant_idx[idx], dc_idx[idx]]
                prod = np.full(len(idx), PRODUCTION_DAYS)
                for ev in EVENTS:
                    if ev.kind == "supplier" and _active(day, ev):
                        prod = prod + ev.delay_days * uses_soc[idx]
                    if ev.kind == "delay" and _active(day, ev):
                        prod = prod + ev.delay_days * np.isin(sourcing["plant_id"].to_numpy()[idx], ev.nodes)
                noise = rng.lognormal(0, 0.10, len(idx)) * transit + rng.exponential(1.0, len(idx))
                lt = np.round(noise + prod).astype(int)
                arr_day = t + lt
                np.add.at(arrivals, (idx, np.minimum(arr_day, horizon - 1)), qty[idx])
                for a_day in np.unique(arr_day):
                    sel = arr_day == a_day
                    arrival_learn.setdefault(int(a_day), []).append((idx[sel], lt[sel].astype(float)))
                route_info = cps[plant_idx[idx], dc_idx[idx]]
                ship_rows.append(pd.DataFrame({
                    "order_date": day, "sku_id": pairs["sku_id"].to_numpy()[idx], "dc_id": pairs["dc_id"].to_numpy()[idx],
                    "plant_id": sourcing["plant_id"].to_numpy()[idx], "qty_units": qty[idx].astype(int),
                    "planned_lead_time_days": np.round(lt_plan[idx], 1), "actual_lead_time_days": lt,
                    "arrival_date": [days[0].date() + timedelta(days=int(x)) for x in arr_day],
                    "chokepoints": [str(r).split("#")[0] for r in route_info],
                    "route_path": [str(r).split("#")[1] if "#" in str(r) else "" for r in route_info],
                    "active_events": "|".join(sorted(active)) if active else "",
                }))
        out_on_order[:, t] = arrivals[:, t + 1:].sum(axis=1)
    del cum_arrivals

    n = n_series * n_days
    inv = pd.DataFrame({
        "date": np.tile(days.date, n_series),
        "sku_id": np.repeat(pairs["sku_id"].to_numpy(), n_days),
        "dc_id": np.repeat(pairs["dc_id"].to_numpy(), n_days),
        "on_hand_units": out_on_hand.ravel(), "on_order_units": out_on_order.ravel(),
        "sales_units": out_sales.ravel(), "lost_sales_units": out_lost.ravel(),
    })
    assert len(inv) == n
    shipments = pd.concat(ship_rows, ignore_index=True)
    shipments.insert(0, "shipment_id", [f"SH{i:07d}" for i in range(1, len(shipments) + 1)])
    shipments["delay_days"] = shipments["actual_lead_time_days"] - shipments["planned_lead_time_days"]
    shipments["on_time"] = shipments["delay_days"] <= 3

    # Pipeline still in transit at the end of history (needed for forward-looking risk).
    end = n_days - 1
    pipe_idx, pipe_day = np.nonzero(arrivals[:, end + 1:] > 0)
    pipeline = pd.DataFrame({"sku_id": pairs["sku_id"].to_numpy()[pipe_idx], "dc_id": pairs["dc_id"].to_numpy()[pipe_idx],
                             "days_until_arrival": pipe_day + 1, "qty_units": arrivals[pipe_idx, end + 1 + pipe_day]})
    sourcing = sourcing.assign(planned_lead_time_days=np.round(lt_plan, 1), learned_lead_time_days=np.round(lt_est, 1))
    return inv, shipments, sourcing, pipeline


def supplier_performance(net: Network, rng: np.random.Generator) -> pd.DataFrame:
    """Weekly tier-2 deliveries into plants with on-time / quality outcomes."""
    weeks = pd.date_range(HISTORY_START, HISTORY_END, freq="W-MON")
    edges = net.edges[net.edges["edge_type"] == "component"]
    risk = net.nodes.set_index("node_id")["risk_score"]
    rows = []
    for e in edges.itertuples():
        base_otd = 0.97 - 0.25 * risk.get(e.src, 0.3)
        for w in weeks:
            p = base_otd
            for ev in EVENTS:
                if e.src in ev.nodes and ev.start <= w.date() <= ev.end:
                    p *= 0.35
            on_time = rng.random() < p
            rows.append({"week_start": w.date(), "supplier_id": e.src, "plant_id": e.dst,
                         "on_time": bool(on_time), "days_late": 0 if on_time else int(rng.integers(2, 15)),
                         "quality_ppm": int(rng.gamma(2, 120 + 600 * risk.get(e.src, 0.3)))})
    return pd.DataFrame(rows)


def run(seed: int = SEED) -> dict[str, int]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    net = build_network(seed)
    promos = build_promo_calendar(net, rng)
    log.info("Simulating demand for %d SKU x DC series over %d days", len(net.skus) * len(DCS), len(_days()))
    demand_long, demand, pairs = build_demand(net, promos, rng)
    log.info("Simulating replenishment, shipments and inventory")
    inv, shipments, sourcing, pipeline = simulate_supply(net, demand, pairs, rng)
    supp = supplier_performance(net, rng)
    events = pd.DataFrame([{**e.__dict__, "nodes": "|".join(e.nodes)} for e in EVENTS])

    tables = {
        "nodes": net.nodes, "edges": net.edges, "skus": net.skus, "bom": net.bom, "sku_plants": net.sku_plants,
        "component_sources": net.component_sources, "material_sources": net.material_sources,
        "promo_calendar": promos, "demand_daily": demand_long, "inventory_daily": inv, "shipments": shipments,
        "sourcing": sourcing, "pipeline_open": pipeline, "disruption_events": events, "supplier_deliveries": supp,
    }
    counts = {}
    for name, df in tables.items():
        df.to_parquet(LANDING_DIR / f"{name}.parquet", index=False)
        counts[name] = len(df)
    manifest = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "seed": seed, "rows": counts,
                "history": [str(HISTORY_START), str(HISTORY_END)], "seconds": round(time.perf_counter() - t0, 1)}
    (LANDING_DIR / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Landing written: %s rows in %.1fs", f"{sum(counts.values()):,}", time.perf_counter() - t0)
    return counts


if __name__ == "__main__":
    run()
