"""
Monte Carlo stockout-risk engine (8-week outlook, all 720 SKU × DC series at once).

For each simulated future (default 400 per series):

1. **Demand.** Weekly draws come from a log-normal fitted to the forecast's P50 and
   P90, then split into days with intra-week noise. A demand shock scales the median.
2. **Supply.** Inventory already in transit arrives on schedule, or later by the
   scenario's re-route delay. New orders follow the weekly order-up-to policy,
   using either the current uniform service level or the optimised per-SKU one.
   - Order lead times are drawn around the scenario lead time while the
     disruption lasts, and the baseline lead time afterwards.
   - Order quantities are scaled by the fulfilment factor (lane capacity ×
     component availability).
3. **Selling.** Each day: sell min(stock, demand) and record lost sales.

Outputs per series:
- P(stockout within the horizon)
- expected fill rate
- expected lost units and revenue
- median day of first stockout
- projected on-hand P10 / P50 / P90 trajectories (used by the drill-down charts)

Everything is vectorised as (sims × series) NumPy arrays. The only Python loop is
over the 56 days, so a full-network run takes about a second.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import BASELINE_SERVICE_Z, MC_SIMS, REVIEW_PERIOD_DAYS, RISK_HORIZON_DAYS, SEED


@dataclass
class RiskResult:
    summary: pd.DataFrame          # one row per SKU x DC
    traj_p10: np.ndarray           # (series, days) projected on-hand percentiles
    traj_p50: np.ndarray
    traj_p90: np.ndarray
    reorder_point: np.ndarray      # (series,) policy target (for chart reference lines)


def simulate(series: pd.DataFrame, forecast: pd.DataFrame, pipeline: pd.DataFrame, *, z: np.ndarray | None = None,
             duration_days: int = 0, demand_shock: float = 0.0, lead_time_multiplier: float = 1.0,
             sims: int = MC_SIMS, horizon: int = RISK_HORIZON_DAYS, seed: int = SEED) -> RiskResult:
    """
    series   : one row per SKU x DC with columns sku_id, dc_id, on_hand_units, unit_price, lt_base, lt_scenario,
               fulfil_factor, pipeline_delay_days  (from scenarios.evaluate + mart_inventory_health)
    forecast : sku_id, dc_id, horizon (weeks 1..), p10, p50, p90
    pipeline : sku_id, dc_id, days_until_arrival, qty_units (open in-transit orders)
    z        : per-series safety factor (default: current uniform policy)
    """
    rng = np.random.default_rng(seed)
    s = series.reset_index(drop=True)
    n = len(s)
    key = pd.MultiIndex.from_frame(s[["sku_id", "dc_id"]])
    weeks = int(np.ceil(horizon / 7))

    fc = forecast[forecast["horizon"] <= weeks].pivot_table(index=["sku_id", "dc_id"], columns="horizon",
                                                             values=["p10", "p50", "p90"])
    p50 = fc["p50"].reindex(key).fillna(0).to_numpy() * (1 + demand_shock)           # (n, weeks)
    p90 = fc["p90"].reindex(key).fillna(0).to_numpy() * (1 + demand_shock)
    sigma_w = np.clip(np.log((p90 + 1) / (p50 + 1)) / 1.2816, 0.05, 1.2)              # weekly log-sd
    mu_daily = p50 / 7.0

    z = np.full(n, BASELINE_SERVICE_Z) if z is None else np.asarray(z, dtype=float)
    lt_base = s["lt_base"].to_numpy(dtype=float) * lead_time_multiplier
    lt_scn = s["lt_scenario"].to_numpy(dtype=float) * lead_time_multiplier
    fulfil = s["fulfil_factor"].to_numpy(dtype=float)
    on_hand = np.repeat(s["on_hand_units"].to_numpy(dtype=np.float32)[None, :], sims, axis=0)

    # In-transit pipeline (shifted by scenario delays), identical across sims.
    arrivals = np.zeros((sims, n, horizon), dtype=np.float32)
    if len(pipeline):
        pos = pd.Series(np.arange(n), index=key)
        p = pipeline.copy()
        p["i"] = pos.reindex(pd.MultiIndex.from_frame(p[["sku_id", "dc_id"]])).to_numpy()
        p = p.dropna(subset=["i"])
        p["i"] = p["i"].astype(int)
        p["day"] = (p["days_until_arrival"] - 1 + s["pipeline_delay_days"].to_numpy()[p["i"]]).astype(int)
        p = p[p["day"] < horizon]
        base_pipe = np.zeros((n, horizon), dtype=np.float32)
        np.add.at(base_pipe, (p["i"].to_numpy(), p["day"].to_numpy()), p["qty_units"].to_numpy(dtype=np.float32))
        arrivals += base_pipe[None, :, :]

    # Demand paths: weekly log-normal multiplier x daily noise.
    wk_mult = np.exp(rng.standard_normal((sims, n, weeks)) * sigma_w[None] - 0.5 * sigma_w[None] ** 2)
    day_week = np.minimum(np.arange(horizon) // 7, weeks - 1)
    daily_noise = rng.lognormal(-0.03, 0.25, (sims, n, horizon)).astype(np.float32)
    demand = (mu_daily[None, :, day_week] * wk_mult[:, :, day_week] * daily_noise).astype(np.float32)
    demand = rng.poisson(demand).astype(np.float32)
    del daily_noise, wk_mult  # free ~65 MB before the day loop

    lost = np.zeros((sims, n), dtype=np.float32)
    total_dem = demand.sum(axis=2)
    first_stockout = np.full((sims, n), horizon + 1, dtype=np.int16)
    traj = np.zeros((sims, n, horizon), dtype=np.float32)
    target_ref = np.zeros(n)

    for t in range(horizon):
        on_hand += arrivals[:, :, t]
        if t % REVIEW_PERIOD_DAYS == 0:
            w = min(t // 7, weeks - 1)
            in_disruption = t < duration_days
            lt_plan = np.where(in_disruption, lt_scn, lt_base)
            cover = lt_plan + REVIEW_PERIOD_DAYS
            # Weekly sd ≈ weekly mean x log-sd; spread over 7 independent days -> daily sd = mu_d x sigma_w x sqrt(7).
            sd_daily = mu_daily[:, w] * sigma_w[:, w] * np.sqrt(7) + 0.5
            target = mu_daily[:, w] * cover + z * np.sqrt(cover * sd_daily ** 2 + (mu_daily[:, w] * 0.15 * lt_plan) ** 2)
            if t == 0:
                target_ref = target
            position = on_hand + arrivals[:, :, t + 1:].sum(axis=2)
            qty = np.maximum(0, target[None, :] - position) * np.where(in_disruption, fulfil, 1.0)[None, :]
            lt_draw = lt_plan[None, :] * rng.lognormal(0, 0.12, (sims, n))
            arr = t + np.round(lt_draw).astype(int)
            ok = arr < horizon
            si, ni = np.nonzero(ok & (qty > 0))
            np.add.at(arrivals, (si, ni, arr[si, ni]), qty[si, ni].astype(np.float32))
        d = demand[:, :, t]
        sold = np.minimum(d, on_hand)
        short = d - sold
        lost += short
        newly = (short > 0) & (first_stockout > horizon)
        first_stockout[newly] = t
        on_hand -= sold
        traj[:, :, t] = on_hand

    fill = 1 - lost / np.maximum(total_dem, 1e-6)
    price = s["unit_price"].to_numpy()
    fs = first_stockout.astype(float)
    fs[fs > horizon] = np.nan
    summary = s[["sku_id", "dc_id"]].copy()
    # "Material" stockout: at least one full day of demand lost (a single lost unit on a slow mover isn't a crisis).
    day_demand = np.maximum(1.0, total_dem.mean(axis=0) / horizon)
    summary["p_stockout"] = (lost >= day_demand[None, :]).mean(axis=0)
    summary["p_any_stockout"] = (first_stockout <= horizon).mean(axis=0)
    summary["expected_fill_rate"] = fill.mean(axis=0)
    summary["expected_lost_units"] = lost.mean(axis=0)
    summary["expected_lost_revenue"] = lost.mean(axis=0) * price
    summary["expected_demand_units"] = total_dem.mean(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # series that never stock out -> all-NaN column
        summary["median_days_to_stockout"] = np.nanmedian(fs, axis=0)
    summary["projected_end_on_hand_p50"] = np.median(traj[:, :, -1], axis=0)
    summary["risk_band"] = pd.cut(summary["p_stockout"], [-0.01, 0.2, 0.5, 0.8, 1.01],
                                  labels=["Low", "Watch", "High", "Critical"]).astype(str)
    q = np.percentile(traj, [10, 50, 90], axis=0)
    return RiskResult(summary, q[0], q[1], q[2], np.asarray(target_ref))
