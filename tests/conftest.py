"""Test isolation: point the warehouse and landing zone at a temp dir BEFORE any src module is imported."""
import os
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="sct_test_"))
os.environ["SCT_DATA_DIR"] = str(_TMP)
os.environ["SCT_WAREHOUSE"] = str(_TMP / "warehouse.duckdb")
os.environ["SCT_MC_SIMS"] = "60"
os.environ.setdefault("SCT_LOG_LEVEL", "WARNING")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
