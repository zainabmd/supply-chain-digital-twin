"""
Inventory policy optimisation (SciPy).

Problem (one per DC, solved with SLSQP and analytic gradients):

    choose a safety factor z_i for every SKU at the DC to minimise

        Σ_i  h·c_i·σ_i·z_i                 annual holding cost of safety stock
           + 52·σ_i·G(z_i)·π_i            expected annual shortage cost (weekly review)

    subject to the DC's storage capacity:

        Σ_i  v_i·(σ_i·z_i + μ_i·R/2)  ≤  capacity_m3,      0 ≤ z_i ≤ 4

    where
        σ_i  = √((L_i+R)·σ_d² + μ_i²·σ_L²)  demand uncertainty over lead time + review period
        G(z) = φ(z) − z·(1−Φ(z))            standard-normal loss function
        π_i  = 35% × unit margin + 5% × price   expected cost of a stockout unit
               (most stockouts substitute or backorder; the rest are lost)

The current policy uses the same z = 1.65 for every SKU, whatever its margin,
cost or variability. The optimiser instead buys service where it is cheap and
valuable (high-margin, low-cost, volatile SKUs), and economises where it isn't.

An **efficient frontier** (inventory investment vs expected fill rate) comes from
the Lagrangian dual of the budget-constrained problem. Root-finding with
`scipy.optimize.brentq` yields the best attainable fill rate at the current
policy's budget.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import norm

from src.config import BASELINE_SERVICE_Z, HOLDING_RATE, REVIEW_PERIOD_DAYS, get_logger
from src.warehouse import read_sql, write_df

log = get_logger("optimize")
LOST_SALE_SHARE = 0.35   # share of stockouts that become truly lost sales (rest substitute or backorder)
GOODWILL = 0.05          # long-run customer-goodwill cost as share of price
SPACE_HEADROOM = 1.10    # allocated DC space for this product family = current footprint + 10%
Z_MAX = 4.0


def loss(z):
    return norm.pdf(z) - z * (1 - norm.cdf(z))


def load_items() -> pd.DataFrame:
    h = read_sql("SELECT * FROM marts.mart_inventory_health")
    fc = read_sql("SELECT sku_id, dc_id, avg(p50) AS p50_w, avg(p10) AS p10_w, avg(p90) AS p90_w "
                  "FROM ml.forecast_weekly WHERE horizon <= 8 GROUP BY ALL")
    cap = read_sql("SELECT node_id AS dc_id, storage_m3 FROM raw.nodes WHERE node_type = 'dc'")
    df = h.merge(fc, on=["sku_id", "dc_id"], how="left").merge(cap, on="dc_id")
    df["mu"] = (df["p50_w"].fillna(df["avg_daily_demand_28d"] * 7) / 7).clip(lower=0.01)
    df["sd_d"] = ((df["p90_w"] - df["p10_w"]).fillna(0) / 2.563 / np.sqrt(7)).clip(lower=0.3)
    df["L"] = df["lt_mean_26w"].fillna(df["learned_lead_time_days"])
    df["sd_L"] = df["lt_std_26w"].fillna(2.0)
    R = REVIEW_PERIOD_DAYS
    df["sigma"] = np.sqrt((df["L"] + R) * df["sd_d"] ** 2 + df["mu"] ** 2 * df["sd_L"] ** 2)
    df["pi"] = LOST_SALE_SHARE * (df["unit_price"] - df["unit_cost"]) + GOODWILL * df["unit_price"]
    df["annual_demand"] = df["mu"] * 365
    return df


def evaluate_policy(df: pd.DataFrame, z: np.ndarray) -> pd.DataFrame:
    R = REVIEW_PERIOD_DAYS
    out = df[["sku_id", "dc_id", "category", "unit_cost", "unit_price", "unit_volume_m3", "mu", "sigma", "L"]].copy()
    out["z"] = z
    out["safety_stock"] = z * df["sigma"]
    out["reorder_point"] = df["mu"] * (df["L"] + R) + out["safety_stock"]
    out["ss_value"] = out["safety_stock"] * df["unit_cost"]
    out["holding_cost"] = HOLDING_RATE * out["ss_value"]
    out["expected_short_units_yr"] = 52 * df["sigma"] * loss(z)
    out["shortage_cost"] = out["expected_short_units_yr"] * df["pi"]
    out["fill_rate"] = 1 - out["expected_short_units_yr"] / df["annual_demand"].clip(lower=1e-6)
    out["volume_m3"] = df["unit_volume_m3"] * (out["safety_stock"] + df["mu"] * R / 2)
    out["total_cost"] = out["holding_cost"] + out["shortage_cost"]
    return out


def optimise_dc(g: pd.DataFrame) -> tuple[np.ndarray, str]:
    hc = HOLDING_RATE * g["unit_cost"].to_numpy() * g["sigma"].to_numpy()
    sc = 52 * g["sigma"].to_numpy() * g["pi"].to_numpy()
    vol = g["unit_volume_m3"].to_numpy()
    sig, mu = g["sigma"].to_numpy(), g["mu"].to_numpy()
    cap = float(g["storage_m3"].iloc[0])
    cycle_vol = float((vol * mu * REVIEW_PERIOD_DAYS / 2).sum())

    # Warm start from the unconstrained newsvendor optimum: Φ(z*) = 1 - h c / (52 π)
    z0 = np.clip(norm.ppf(np.clip(1 - hc / np.maximum(sc, 1e-9), 1e-4, 1 - 1e-6)), 0, Z_MAX)
    # Normalise objective and constraint to O(1): with raw dollars (~$1M) and m³ SLSQP's line search stalls.
    f_scale = 1.0 / max(float(hc @ z0 + sc @ loss(z0)), 1.0)
    c_scale = 1.0 / max(cap, 1e-9)

    def f(z):
        return float(hc @ z + sc @ loss(z)) * f_scale

    def grad(z):
        return (hc + sc * (norm.cdf(z) - 1)) * f_scale

    cons = [{"type": "ineq", "fun": lambda z: (cap - cycle_vol - float((vol * sig) @ z)) * c_scale,
             "jac": lambda z: -(vol * sig) * c_scale}]
    if cycle_vol + float((vol * sig) @ np.full(len(z0), 0.0)) > cap:  # infeasible even with zero safety stock
        z0 = np.zeros_like(z0)
    res = minimize(f, z0, jac=grad, bounds=[(0, Z_MAX)] * len(z0), constraints=cons, method="SLSQP",
                   options={"maxiter": 300, "ftol": 1e-9})
    return np.clip(res.x, 0, Z_MAX), ("optimal" if res.success else f"approx: {res.message}")


def frontier(df: pd.DataFrame, points: int = 40) -> pd.DataFrame:
    """Fill-rate frontier: min Σ units short s.t. Σ c σ z ≤ B. Lagrangian optimum: Φ(z_i) = 1 − λ·c_i / 52.

    (Units, not margin: the frontier answers "best fill rate per inventory dollar"; cheap SKUs get service first.)
    """
    c, sig, A = (df[k].to_numpy() for k in ("unit_cost", "sigma", "annual_demand"))

    def at(lam):
        z = np.clip(norm.ppf(np.clip(1 - lam * c / 52, 1e-6, 1 - 1e-9)), 0, Z_MAX)
        budget = float((c * sig * z).sum())
        fill = 1 - float((52 * sig * loss(z)).sum() / A.sum())
        return budget, fill, z

    rows = [dict(zip(("ss_investment", "fill_rate"), at(lam)[:2]), lam=lam) for lam in np.logspace(-4, 1.5, points)]
    return pd.DataFrame(rows).sort_values("ss_investment")


def run() -> dict:
    t0 = time.perf_counter()
    df = load_items()
    current = evaluate_policy(df, np.full(len(df), BASELINE_SERVICE_Z))
    # Space actually allocated to this family is the current footprint plus headroom: the optimiser must
    # reallocate scarce shelf space, not just buy more stock.
    df["storage_m3"] = df["dc_id"].map(current.groupby("dc_id")["volume_m3"].sum() * SPACE_HEADROOM)
    z_opt = np.zeros(len(df))
    status = {}
    for dc, g in df.groupby("dc_id"):
        z, st = optimise_dc(g)
        z_opt[g.index.to_numpy()] = z
        status[dc] = st
    optimal = evaluate_policy(df, z_opt)

    # Same-budget comparison: best fill achievable for exactly the current safety-stock investment.
    c, pi = df["unit_cost"].to_numpy(), df["pi"].to_numpy()
    sig, A = df["sigma"].to_numpy(), df["annual_demand"].to_numpy()
    cur_budget = float(current["ss_value"].sum())

    def budget_gap(lam):
        z = np.clip(norm.ppf(np.clip(1 - lam * c / 52, 1e-6, 1 - 1e-9)), 0, Z_MAX)
        return float((c * sig * z).sum()) - cur_budget
    lam = brentq(budget_gap, 1e-7, 1e4)
    z_same = np.clip(norm.ppf(np.clip(1 - lam * c / 52, 1e-6, 1 - 1e-9)), 0, Z_MAX)
    same_budget = evaluate_policy(df, z_same)

    policy = current[["sku_id", "dc_id", "category"]].copy()
    for name, p in (("current", current), ("optimal", optimal)):
        for col in ("z", "safety_stock", "reorder_point", "ss_value", "holding_cost", "shortage_cost", "total_cost",
                    "fill_rate", "volume_m3"):
            policy[f"{name}_{col}"] = p[col].to_numpy()
    policy["mu_daily"] = df["mu"].to_numpy()
    policy["sigma_lt"] = df["sigma"].to_numpy()
    policy["unit_cost"] = df["unit_cost"].to_numpy()
    policy["unit_margin"] = (df["unit_price"] - df["unit_cost"]).to_numpy()
    write_df(policy, "ml.inventory_policy")

    dc_summary = (policy.groupby("dc_id").agg(current_ss_value=("current_ss_value", "sum"), optimal_ss_value=("optimal_ss_value", "sum"),
                                              current_total_cost=("current_total_cost", "sum"), optimal_total_cost=("optimal_total_cost", "sum"),
                                              optimal_volume_m3=("optimal_volume_m3", "sum"), current_volume_m3=("current_volume_m3", "sum"))
                  .join(df.groupby("dc_id")["storage_m3"].first()).reset_index())
    dc_summary["solver_status"] = dc_summary["dc_id"].map(status)
    write_df(dc_summary, "ml.inventory_policy_dc")
    fr = frontier(df)
    write_df(fr, "ml.inventory_frontier")

    w = df["annual_demand"].to_numpy()
    summary = {
        "current_ss_value": cur_budget, "optimal_ss_value": float(optimal["ss_value"].sum()),
        "current_total_cost": float(current["total_cost"].sum()), "optimal_total_cost": float(optimal["total_cost"].sum()),
        "current_fill": float(np.average(current["fill_rate"], weights=w)), "optimal_fill": float(np.average(optimal["fill_rate"], weights=w)),
        "same_budget_fill": float(np.average(same_budget["fill_rate"], weights=w)),
    }
    write_df(pd.DataFrame([summary]), "ml.inventory_policy_summary")
    log.info("Policy optimised in %.1fs · total cost %.2fM → %.2fM (%.1f%%) · fill %.2f%% → %.2f%% · same-budget fill %.2f%%",
             time.perf_counter() - t0, summary["current_total_cost"] / 1e6, summary["optimal_total_cost"] / 1e6,
             100 * (1 - summary["optimal_total_cost"] / summary["current_total_cost"]), 100 * summary["current_fill"],
             100 * summary["optimal_fill"], 100 * summary["same_budget_fill"])
    return summary


if __name__ == "__main__":
    run()
