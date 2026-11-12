"""
Central configuration for the supply-chain digital twin.

Environment variables override defaults so the same code runs locally,
inside the Airflow containers and in CI.
"""
from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("SCT_DATA_DIR", PROJECT_ROOT / "data"))
LANDING_DIR = DATA_DIR / "landing"            # extracted files (what an ERP/TMS export would drop)
WAREHOUSE_PATH = Path(os.getenv("SCT_WAREHOUSE", DATA_DIR / "warehouse.duckdb"))
MODEL_DIR = PROJECT_ROOT / "models"
SQL_DIR = PROJECT_ROOT / "sql"

for _d in (LANDING_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = int(os.getenv("SCT_SEED", "7"))
HISTORY_START = date(2023, 1, 2)          # a Monday: weekly buckets align with ISO weeks
HISTORY_END = date(2024, 12, 29)          # a Sunday: 104 full weeks
FORECAST_WEEKS = 13                       # one quarter ahead
BACKTEST_WEEKS = 13

# Inventory economics
HOLDING_RATE = 0.25                       # annual carrying cost as share of unit cost
REVIEW_PERIOD_DAYS = 7                    # weekly replenishment review
BASELINE_SERVICE_Z = 1.65                 # current policy: ~95% cycle service everywhere

# Monte Carlo
MC_SIMS = int(os.getenv("SCT_MC_SIMS", "300"))
RISK_HORIZON_DAYS = 56                    # 8-week stockout-risk outlook

LOG_LEVEL = os.getenv("SCT_LOG_LEVEL", "INFO")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s"))
        logger.addHandler(h)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False
    return logger
