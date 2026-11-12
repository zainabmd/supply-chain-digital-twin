"""
Weekly demand feature store (ml.feature_store_weekly).

Point-in-time correctness: features for origin week `o` use only data up to
and including `o`. A model forecasting week `o + h` can never see the
future, so offline training matches online inference exactly.

Known-in-advance covariates (planned promotions, calendar events) live in a
separate table keyed by *target* week (ml.future_covariates), because they're
legitimately known at forecast time.

Each refresh stamps `feature_version` and `computed_at`, so a model can be
reproduced against the exact feature snapshot it was trained on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FORECAST_WEEKS, HISTORY_START, get_logger
from src.warehouse import read_sql, write_df

log = get_logger("features")
FEATURE_VERSION = "fs-weekly-v1"
LAGS = [0, 1, 2, 3, 7, 12, 51]   # lag 0 = the origin week itself; 51 → same week last year for h=1


def calendar_flags(week_start: pd.Series) -> pd.DataFrame:
    ws = pd.to_datetime(week_start)
    woy = ws.dt.isocalendar().week.astype(int).clip(upper=52)
    return pd.DataFrame({
        "woy_sin": np.sin(2 * np.pi * woy / 52), "woy_cos": np.cos(2 * np.pi * woy / 52),
        "is_black_friday": ((ws.dt.month == 11) & (ws.dt.day >= 17)).astype(int),
        "is_christmas": ((ws.dt.month == 12) & (ws.dt.day <= 22)).astype(int),
        "is_back_to_school": ((ws.dt.month == 8) | ((ws.dt.month == 9) & (ws.dt.day <= 7))).astype(int),
    }, index=week_start.index)


def build_feature_store() -> pd.DataFrame:
    w = read_sql("SELECT week_start, sku_id, dc_id, category, dc_region, demand_units, promo_days, stockout_days "
                 "FROM marts.mart_weekly_sku_dc ORDER BY sku_id, dc_id, week_start")
    w["week_start"] = pd.to_datetime(w["week_start"])
    g = w.groupby(["sku_id", "dc_id"], sort=False)["demand_units"]
    feats = w[["week_start", "sku_id", "dc_id", "category", "dc_region"]].copy()
    for lag in LAGS:
        feats[f"lag_{lag}"] = g.shift(lag)
    # Rolling statistics end at the origin week (inclusive) – no future leakage.
    feats["roll_mean_4"] = g.transform(lambda s: s.rolling(4, min_periods=2).mean())
    feats["roll_mean_13"] = g.transform(lambda s: s.rolling(13, min_periods=4).mean())
    feats["roll_std_13"] = g.transform(lambda s: s.rolling(13, min_periods=4).std())
    feats["roll_max_13"] = g.transform(lambda s: s.rolling(13, min_periods=4).max())
    feats["trend_ratio"] = feats["roll_mean_4"] / feats["roll_mean_13"].replace(0, np.nan)
    feats["stockout_days_4"] = w.groupby(["sku_id", "dc_id"], sort=False)["stockout_days"].transform(
        lambda s: s.rolling(4, min_periods=1).sum())  # censored demand signal
    feats["weeks_since_start"] = ((feats["week_start"] - pd.Timestamp(HISTORY_START)).dt.days // 7).astype(int)
    feats["feature_version"] = FEATURE_VERSION
    feats["computed_at"] = pd.Timestamp.now(tz="UTC")
    write_df(feats, "ml.feature_store_weekly")

    cov = build_future_covariates()
    log.info("Feature store refreshed: %s origin rows, %s covariate rows (%s)", f"{len(feats):,}", f"{len(cov):,}",
             FEATURE_VERSION)
    return feats


def build_future_covariates() -> pd.DataFrame:
    """Target-week covariates known in advance: planned promotions + calendar."""
    pairs = read_sql("SELECT DISTINCT d.sku_id, d.dc_id, n.region FROM raw.sourcing d JOIN raw.nodes n ON n.node_id = d.dc_id")
    hist_weeks = read_sql("SELECT DISTINCT week_start FROM marts.mart_weekly_sku_dc ORDER BY 1")["week_start"]
    last = pd.to_datetime(hist_weeks).max()
    weeks = pd.DatetimeIndex(pd.to_datetime(hist_weeks)).append(
        pd.date_range(last + pd.Timedelta(weeks=1), periods=FORECAST_WEEKS, freq="W-MON"))
    grid = pairs.merge(pd.DataFrame({"week_start": weeks}), how="cross")
    promos = read_sql("SELECT sku_id, region, CAST(week_start AS TIMESTAMP) AS week_start, uplift FROM raw.promo_calendar")
    grid = grid.merge(promos, on=["sku_id", "region", "week_start"], how="left")
    grid["promo_uplift"] = grid["uplift"].fillna(1.0)
    grid["promo_flag"] = (grid["promo_uplift"] > 1).astype(int)
    grid = pd.concat([grid.drop(columns=["uplift", "region"]), calendar_flags(grid["week_start"])], axis=1)
    grid["feature_version"] = FEATURE_VERSION
    write_df(grid, "ml.future_covariates")
    return grid


if __name__ == "__main__":
    build_feature_store()
