"""
Probabilistic demand forecasting: a global, direct multi-horizon quantile model.

* **Global:** one model across all 720 SKU × DC series. It learns shared seasonality
  and promo response, and copes with sparse or new series.
* **Direct multi-horizon:** one training row per (origin week, horizon 1..13). The
  horizon is a feature, so there is no recursive error compounding.
* **Probabilistic:** separate gradient-boosted models for the P10, P50 and P90
  quantiles (pinball loss). The P10–P90 band feeds the Monte Carlo stockout-risk
  model, which needs a distribution, not just a point.
* **Honest evaluation:** a time-based backtest holds out the last 13 weeks,
  scored against two baselines a planner would actually use: seasonal-naive
  (same week last year) and a 4-week moving average.

Targets are modelled in log1p space (quantiles are invariant to monotone transforms).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from src.config import BACKTEST_WEEKS, FORECAST_WEEKS, MODEL_DIR, get_logger
from src.warehouse import connect, read_sql, write_df

log = get_logger("forecast")
QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
CAT_COLS = ["category", "dc_id", "dc_region"]
NUM_COLS = ["horizon", "lag_0", "lag_1", "lag_2", "lag_3", "lag_7", "lag_12", "lag_51", "roll_mean_4", "roll_mean_13",
            "roll_std_13", "roll_max_13", "trend_ratio", "stockout_days_4", "log_price",
            "promo_uplift", "promo_flag", "woy_sin", "woy_cos", "is_black_friday", "is_christmas", "is_back_to_school"]
FEATURES = NUM_COLS + CAT_COLS


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feats = read_sql("SELECT * FROM ml.feature_store_weekly")
    cov = read_sql("SELECT * EXCLUDE (feature_version) FROM ml.future_covariates")
    dem = read_sql("SELECT week_start, sku_id, dc_id, demand_units FROM marts.mart_weekly_sku_dc")
    price = read_sql("SELECT sku_id, ln(unit_price) AS log_price FROM staging.stg_skus")
    for df in (feats, cov, dem):
        df["week_start"] = pd.to_datetime(df["week_start"])
    feats = feats.merge(price, on="sku_id")
    return feats, cov, dem


def make_rows(feats: pd.DataFrame, cov: pd.DataFrame, dem: pd.DataFrame, horizons=range(1, FORECAST_WEEKS + 1),
              origins: pd.Series | None = None) -> pd.DataFrame:
    """Expand origin features to (origin, horizon) rows and attach target-week covariates + actuals."""
    base = feats if origins is None else feats[feats["week_start"].isin(origins)]
    frames = []
    for h in horizons:
        f = base.copy()
        f["horizon"] = h
        f["target_week"] = f["week_start"] + pd.Timedelta(weeks=h)
        frames.append(f)
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.merge(cov.rename(columns={"week_start": "target_week"}), on=["sku_id", "dc_id", "target_week"], how="left")
    rows = rows.merge(dem.rename(columns={"week_start": "target_week", "demand_units": "y"}),
                      on=["sku_id", "dc_id", "target_week"], how="left")
    seasonal = dem.assign(target_week=dem["week_start"] + pd.Timedelta(weeks=52)).rename(columns={"demand_units": "seasonal_naive"})
    rows = rows.merge(seasonal[["sku_id", "dc_id", "target_week", "seasonal_naive"]], on=["sku_id", "dc_id", "target_week"], how="left")
    for c in CAT_COLS:
        rows[c] = rows[c].astype("category")
    return rows


def _model(q: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, learning_rate=0.06, max_iter=300,
                                        max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.1,
                                        categorical_features="from_dtype", random_state=11)


def fit_quantiles(train: pd.DataFrame) -> dict[str, HistGradientBoostingRegressor]:
    X, y = train[FEATURES], np.log1p(train["y"].to_numpy())
    models = {}
    for name, q in QUANTILES.items():
        t = time.perf_counter()
        models[name] = _model(q).fit(X, y)
        log.info("  fitted %s on %s rows in %.1fs", name, f"{len(X):,}", time.perf_counter() - t)
    return models


def predict(models: dict, rows: pd.DataFrame) -> pd.DataFrame:
    preds = np.column_stack([np.expm1(models[k].predict(rows[FEATURES])) for k in QUANTILES])
    preds = np.sort(np.clip(preds, 0, None), axis=1)          # enforce monotone quantiles (no crossing)
    out = rows[["sku_id", "dc_id", "week_start", "target_week", "horizon"]].rename(columns={"week_start": "origin_week"}).copy()
    out[["p10", "p50", "p90"]] = preds
    return out


def score(df: pd.DataFrame) -> dict:
    y, p = df["y"].to_numpy(), df["p50"].to_numpy()
    tot = y.sum()
    pin = np.mean([np.mean(np.maximum(q * (y - df[k]), (q - 1) * (y - df[k]))) for k, q in QUANTILES.items()])
    out = {"wape": float(np.abs(y - p).sum() / tot), "bias": float((p - y).sum() / tot),
           "coverage_p10_p90": float(((y >= df["p10"]) & (y <= df["p90"])).mean()), "mean_pinball": float(pin)}
    if "seasonal_naive" in df:
        sn = df["seasonal_naive"].fillna(df["roll_mean_4"]).to_numpy()
        out["wape_seasonal_naive"] = float(np.abs(y - sn).sum() / tot)
    if "roll_mean_4" in df:
        out["wape_ma4"] = float(np.abs(y - df["roll_mean_4"].to_numpy()).sum() / tot)
    return out


def run() -> dict:
    t0 = time.perf_counter()
    feats, cov, dem = _load()
    weeks = np.sort(feats["week_start"].unique())
    last_week = pd.Timestamp(weeks[-1])
    cutoff = last_week - pd.Timedelta(weeks=BACKTEST_WEEKS)
    eligible = pd.Series(weeks[12:])                            # need ~a quarter of history for lags

    rows = make_rows(feats, cov, dem, origins=eligible)
    rows = rows[rows["y"].notna()]

    # ---- Backtest: train on targets <= cutoff, forecast from the cutoff origin ----
    log.info("Backtest: training on targets up to %s", cutoff.date())
    train_bt = rows[rows["target_week"] <= cutoff]
    models_bt = fit_quantiles(train_bt)
    test = rows[(rows["week_start"] == cutoff) & (rows["target_week"] > cutoff)]
    bt = predict(models_bt, test)
    bt = bt.merge(test[["sku_id", "dc_id", "target_week", "y", "seasonal_naive", "roll_mean_4", "category"]],
                  on=["sku_id", "dc_id", "target_week"])
    metrics = score(bt)
    metrics["by_horizon"] = {int(h): score(g)["wape"] for h, g in bt.groupby("horizon")}
    metrics["by_category"] = {str(c): score(g)["wape"] for c, g in bt.groupby("category", observed=True)}
    log.info("Backtest WAPE %.3f (seasonal-naive %.3f, MA4 %.3f) · bias %+.3f · P10-P90 coverage %.2f",
             metrics["wape"], metrics["wape_seasonal_naive"], metrics["wape_ma4"], metrics["bias"], metrics["coverage_p10_p90"])

    # Permutation importance of the P50 model on the backtest window (what drives the forecast?).
    sample = test.sample(min(4000, len(test)), random_state=3)
    pi = permutation_importance(models_bt["p50"], sample[FEATURES], np.log1p(sample["y"]), n_repeats=3, random_state=3,
                                scoring="neg_mean_absolute_error")
    importance = pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean}).sort_values("importance", ascending=False)

    # ---- Production model: all history, forecast 13 weeks from the last week ----
    log.info("Final fit on all history; forecasting %d weeks from %s", FORECAST_WEEKS, last_week.date())
    models = fit_quantiles(rows)
    future = make_rows(feats, cov, dem, origins=pd.Series([last_week]))
    fc = predict(models, future)

    version = "hgb-quantile-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    fc["model_version"] = version
    joblib.dump({"models": models, "features": FEATURES, "version": version, "metrics": metrics}, MODEL_DIR / "demand_forecaster.joblib")
    write_df(fc, "ml.forecast_weekly")
    write_df(bt, "ml.forecast_backtest")
    write_df(importance, "ml.forecast_feature_importance")
    reg = pd.DataFrame([{"model_name": "demand_forecaster", "version": version, "trained_at": pd.Timestamp.now(tz="UTC"),
                         "train_rows": len(rows), "metrics_json": json.dumps(metrics), "status": "production"}])
    with connect() as con:
        con.register("_reg", reg)
        con.execute("CREATE TABLE IF NOT EXISTS ops.model_registry AS SELECT * FROM _reg WHERE false")
        con.execute("UPDATE ops.model_registry SET status = 'archived' WHERE model_name = 'demand_forecaster'")
        con.execute("INSERT INTO ops.model_registry SELECT * FROM _reg")
    log.info("Forecasting done in %.1fs (%s)", time.perf_counter() - t0, version)
    return metrics


if __name__ == "__main__":
    from src.features import build_feature_store
    build_feature_store()
    run()
